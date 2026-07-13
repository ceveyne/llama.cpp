#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <clocale>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

#if defined(_MSC_VER)
#    pragma warning(disable : 4244 4267)  // possible loss of data
#endif

using json = nlohmann::ordered_json;

// Prompt construction uses the model's own GGUF-baked "rerank" chat template
// (written by conversion/qwen.py for Qwen3-(VL-)Reranker checkpoints), fetched
// via llama_model_chat_template(model, "rerank") and filled in with simple
// {instruction}/{query}/{document} substitution - the exact same mechanism
// tools/server/server-common.cpp's format_prompt_rerank() uses. This is the
// single source of truth for the reranker prompt; do not reimplement prompt
// construction here. If the prompt is wrong, fix the template in
// conversion/qwen.py (and re-convert), not this file.

namespace {

const char *      DEFAULT_INSTRUCTION =
    "Given a search query, retrieve relevant candidates that answer the query.";
const std::string QWEN_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>";

struct vl_side {
    std::optional<std::string> text;
    std::optional<std::string> image;
};

struct vl_rerank_request {
    std::string          instruction;
    vl_side               query;
    std::vector<vl_side> documents;
};

bool has_image(const vl_side & side) {
    return side.image.has_value() && !side.image->empty();
}

bool has_text(const vl_side & side) {
    return side.text.has_value() && !side.text->empty();
}

vl_side parse_side(const json & obj, const char * label) {
    if (!obj.is_object()) {
        throw std::runtime_error(string_format("%s must be a JSON object", label));
    }

    vl_side side;

    if (obj.contains("text") && !obj.at("text").is_null()) {
        if (!obj.at("text").is_string()) {
            throw std::runtime_error(string_format("%s.text must be a string", label));
        }
        side.text = obj.at("text").get<std::string>();
    }

    if (obj.contains("image") && !obj.at("image").is_null()) {
        if (!obj.at("image").is_string()) {
            throw std::runtime_error(string_format("%s.image must be a string", label));
        }
        side.image = obj.at("image").get<std::string>();
    }

    if (obj.contains("video") && !obj.at("video").is_null()) {
        throw std::runtime_error(
            "video input is not supported in vl-rerank yet; mtmd public API currently does not expose video "
            "ingestion");
    }

    return side;
}

vl_rerank_request parse_request(const std::string & inputs_json) {
    json parsed = json::parse(inputs_json);
    if (!parsed.is_object()) {
        throw std::runtime_error("--inputs/--inputs-file must be a JSON object with \"query\" and \"documents\"");
    }

    if (!parsed.contains("query")) {
        throw std::runtime_error("inputs.query is required");
    }
    if (!parsed.contains("documents") || !parsed.at("documents").is_array()) {
        throw std::runtime_error("inputs.documents must be an array");
    }

    vl_rerank_request request;
    request.instruction = DEFAULT_INSTRUCTION;
    if (parsed.contains("instruction") && !parsed.at("instruction").is_null()) {
        if (!parsed.at("instruction").is_string()) {
            throw std::runtime_error("inputs.instruction must be a string");
        }
        request.instruction = parsed.at("instruction").get<std::string>();
    }

    request.query = parse_side(parsed.at("query"), "inputs.query");

    const auto & documents = parsed.at("documents");
    request.documents.reserve(documents.size());
    for (size_t i = 0; i < documents.size(); ++i) {
        request.documents.push_back(parse_side(documents.at(i), string_format("inputs.documents[%zu]", i).c_str()));
    }

    if (request.documents.empty()) {
        throw std::runtime_error("inputs.documents must not be empty");
    }

    return request;
}

// Builds the {query}/{document} substitution value for one side of the pair:
// an optional image marker followed by text, or the literal "NULL" fallback
// when neither text nor image is present - matching Qwen3VLReranker.format_mm_content()
// in the reference script.
std::string build_side_text(const vl_side & side) {
    std::string result;
    if (has_image(side)) {
        result += QWEN_IMAGE_MARKER;
    }
    if (has_text(side)) {
        result += *side.text;
    } else if (!has_image(side)) {
        result += "NULL";
    }
    return result;
}

std::string format_pair_prompt(const llama_model * model, const std::string & instruction, const vl_side & query,
                               const vl_side & doc) {
    const char * rerank_template = llama_model_chat_template(model, "rerank");
    if (rerank_template == nullptr) {
        throw std::runtime_error("model does not provide a 'rerank' chat template; is this a reranker GGUF?");
    }

    std::string prompt = rerank_template;
    string_replace_all(prompt, "{instruction}", instruction);
    string_replace_all(prompt, "{query}", build_side_text(query));
    string_replace_all(prompt, "{document}", build_side_text(doc));
    return prompt;
}

std::string rewrite_multimodal_markers(std::string prompt) {
    string_replace_all(prompt, QWEN_IMAGE_MARKER, mtmd_default_marker());
    return prompt;
}

mtmd::bitmap load_bitmap_from_path(mtmd_context * ctx_vision, const std::string & path) {
    if (string_starts_with(path, "http://") || string_starts_with(path, "https://") ||
        string_starts_with(path, "oss://")) {
        throw std::runtime_error("remote media URLs are not supported by vl-rerank yet");
    }

    std::string local_path = path;
    if (string_starts_with(local_path, "file://")) {
        local_path = local_path.substr(7);
    }

    mtmd::bitmap bitmap(mtmd_helper_bitmap_init_from_file(ctx_vision, local_path.c_str(), false).bitmap);
    if (!bitmap.ptr) {
        throw std::runtime_error(string_format("failed to load media file '%s'", local_path.c_str()));
    }
    return bitmap;
}

void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

// Returns the rerank score (softmax "yes" probability, already computed inside
// the graph's build_pooling() for LLAMA_POOLING_TYPE_RANK on QWEN3/QWEN3VL arch).
float extract_rank_score(llama_context * ctx) {
    const float * embd = llama_get_embeddings_seq(ctx, 0);
    GGML_ASSERT(embd != NULL && "failed to get rerank score, is --pooling rank supported by this model?");
    return embd[0];
}

float score_text_only(llama_context * ctx, const llama_vocab * vocab, const std::string & prompt, int32_t n_batch) {
    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, true, true);
    if ((int32_t) tokens.size() > n_batch) {
        throw std::runtime_error(
            string_format("number of tokens in prompt (%lld) exceeds batch size (%d), increase batch size and re-run",
                          (long long) tokens.size(), n_batch));
    }

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    batch_add_seq(batch, tokens, 0);
    if (llama_decode(ctx, batch) < 0) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to process prompt");
    }
    llama_batch_free(batch);

    return extract_rank_score(ctx);
}

float score_multimodal(mtmd_context * ctx_vision, llama_context * ctx, const std::string & prompt,
                       const std::vector<std::string> & images, int32_t n_batch) {
    mtmd::bitmaps bitmaps;
    for (const auto & image : images) {
        bitmaps.entries.push_back(load_bitmap_from_path(ctx_vision, image));
    }

    std::string prompt_with_marker = rewrite_multimodal_markers(prompt);

    mtmd_input_text text;
    text.text          = prompt_with_marker.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto               bitmaps_c_ptr = bitmaps.c_ptr();

    llama_memory_clear(llama_get_memory(ctx), true);

    const int32_t tok_res =
        mtmd_tokenize(ctx_vision, chunks.ptr.get(), &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (tok_res != 0) {
        throw std::runtime_error(string_format("unable to tokenize multimodal prompt, res = %d", tok_res));
    }

    llama_pos     new_n_past = 0;
    const int32_t eval_res =
        mtmd_helper_eval_chunks(ctx_vision, ctx, chunks.ptr.get(), 0, 0, n_batch, true, &new_n_past);
    if (eval_res != 0) {
        throw std::runtime_error(string_format("unable to eval multimodal prompt, res = %d", eval_res));
    }

    return extract_rank_score(ctx);
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    params.embedding = true;

    vl_rerank_request request;
    try {
        if (params.inputs_json.empty()) {
            throw std::runtime_error("--inputs or --inputs-file is required, see --help");
        }
        request = parse_request(params.inputs_json);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to parse inputs: %s\n", __func__, e.what());
        return 1;
    }

    const bool needs_vision = has_image(request.query) ||
                              std::any_of(request.documents.begin(), request.documents.end(), has_image);
    if (needs_vision && params.mmproj.path.empty()) {
        LOG_ERR("%s: multimodal inputs require --mmproj\n", __func__);
        return 1;
    }

    // Rerank prompts are short (one query + one document per call, one
    // sequence at a time - never batched/parallel across documents). Do not
    // fall through to the model's full training context (n_ctx=0 means "use
    // n_ctx_train", which is 262144 for Qwen3-VL): that alone allocates a
    // KV cache in the tens of GB for a prompt that is typically a few
    // hundred tokens. --ctx-size still overrides this explicitly.
    if (params.n_ctx == 0) {
        params.n_ctx = 8192;
    }
    if (params.n_batch < params.n_ctx) {
        params.n_batch = params.n_ctx;
    }
    if (params.attention_type != LLAMA_ATTENTION_TYPE_CAUSAL) {
        params.n_ubatch = params.n_batch;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL || ctx == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_RANK) {
        LOG_ERR("%s: model/context pooling type is not 'rank'; this is not a reranker GGUF "
                "(missing pooling_type=rank / cls_out in the converted model)\n",
                __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    mtmd::context_ptr ctx_vision;
    if (needs_vision) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu             = params.mmproj_use_gpu;
        mparams.print_timings       = true;
        mparams.n_threads           = params.cpuparams.n_threads;
        mparams.flash_attn_type     = params.flash_attn_type;
        mparams.warmup              = params.warmup;
        mparams.image_min_tokens    = params.image_min_tokens;
        mparams.image_max_tokens    = params.image_max_tokens;
        ctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("%s: failed to load multimodal projector from %s\n", __func__, params.mmproj.path.c_str());
            return 1;
        }
    }

    std::vector<float> scores;
    scores.reserve(request.documents.size());

    try {
        for (const auto & doc : request.documents) {
            const std::string prompt = format_pair_prompt(model, request.instruction, request.query, doc);

            if (params.verbose_prompt) {
                LOG_INF("%s: formatted prompt: %s\n", __func__, prompt.c_str());
            }

            const bool pair_needs_vision = has_image(request.query) || has_image(doc);
            if (pair_needs_vision) {
                std::vector<std::string> images;
                if (has_image(request.query)) {
                    images.push_back(*request.query.image);
                }
                if (has_image(doc)) {
                    images.push_back(*doc.image);
                }
                scores.push_back(score_multimodal(ctx_vision.get(), ctx, prompt, images, params.n_batch));
            } else {
                scores.push_back(score_text_only(ctx, vocab, prompt, params.n_batch));
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        llama_backend_free();
        return 1;
    }

    if (params.embd_out == "json" || params.embd_out == "array") {
        json out = json::array();
        for (float score : scores) {
            out.push_back(score);
        }
        LOG("%s\n", out.dump().c_str());
    } else {
        for (size_t i = 0; i < scores.size(); ++i) {
            LOG("rerank score %zu: %8.6f\n", i, scores[i]);
        }
    }

    llama_backend_free();
    return 0;
}
