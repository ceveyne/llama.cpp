#include "arg.h"
#include "chat.h"
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

// Reference prompt/scoring behavior verified against:
//   - Qwen3-VL-Reranker-8B/chat_template.jinja (default template, no <think> block)
//   - Qwen3-VL-Reranker-8B/scripts/qwen3_vl_reranker.py (format_mm_instruction, process)
//   - Qwen3-VL-Reranker-8B/1_LogitScore/config.json ("yes"/"no" token ids)
// Do not change the system prompt, the "<Instruct>:"/"<Query>:"/"<Document>:"
// markers, or the NULL fallback text without re-checking that reference.

namespace {

const char *      DEFAULT_INSTRUCTION =
    "Given a search query, retrieve relevant candidates that answer the query.";
const char *      SYSTEM_PROMPT =
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\".";
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

// Builds the exact (system, user) message pair the reference Python implementation
// constructs in Qwen3VLReranker.format_mm_instruction(), rendered through the
// model's own default chat template (mirrors vl-embedding.cpp's build_conversation,
// not the separately baked "rerank" named chat template, which adds an unused
// <think> block for this VL variant).
std::vector<common_chat_msg> build_pair_messages(const std::string & instruction, const vl_side & query,
                                                 const vl_side & doc) {
    std::vector<common_chat_msg> messages;

    common_chat_msg system;
    system.role    = "system";
    system.content = SYSTEM_PROMPT;
    messages.push_back(std::move(system));

    common_chat_msg user;
    user.role = "user";

    user.content_parts.push_back({ "text", "<Instruct>: " + instruction });
    user.content_parts.push_back({ "text", "<Query>:" });
    if (has_image(query)) {
        user.content_parts.push_back({ "media_marker", QWEN_IMAGE_MARKER });
    }
    if (has_text(query)) {
        user.content_parts.push_back({ "text", *query.text });
    } else if (!has_image(query)) {
        user.content_parts.push_back({ "text", "NULL" });
    }

    user.content_parts.push_back({ "text", "\n<Document>:" });
    if (has_image(doc)) {
        user.content_parts.push_back({ "media_marker", QWEN_IMAGE_MARKER });
    }
    if (has_text(doc)) {
        user.content_parts.push_back({ "text", *doc.text });
    } else if (!has_image(doc)) {
        user.content_parts.push_back({ "text", "NULL" });
    }

    messages.push_back(std::move(user));
    return messages;
}

std::string format_pair_prompt(const common_chat_templates * tmpls, const std::string & instruction,
                               const vl_side & query, const vl_side & doc) {
    common_chat_templates_inputs chat_inputs;
    chat_inputs.use_jinja             = true;
    chat_inputs.messages              = build_pair_messages(instruction, query, doc);
    chat_inputs.add_generation_prompt = true;
    chat_inputs.add_bos               = true;
    // Must NOT append an eos token: the classifier score is read from the hidden
    // state of the very last prompt token (right after "assistant\n"), matching
    // Qwen3VLReranker.compute_scores() in the reference script.
    chat_inputs.add_eos = false;
    return common_chat_templates_apply(tmpls, chat_inputs).prompt;
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

    const int n_seq_max = llama_max_parallel_sequences();
    if (params.n_parallel == 1) {
        params.kv_unified = true;
        params.n_parallel = n_seq_max;
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

    common_chat_templates_ptr tmpls;
    try {
        tmpls = common_chat_templates_init(model, "");
    } catch (const std::exception &) {
        LOG_ERR("%s: failed to initialize chat template for the model\n", __func__);
        return 1;
    }

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
            const std::string prompt = format_pair_prompt(tmpls.get(), request.instruction, request.query, doc);

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
