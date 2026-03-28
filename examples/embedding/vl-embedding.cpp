#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <ctime>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

#if defined(_MSC_VER)
#    pragma warning(disable : 4244 4267)  // possible loss of data
#endif

using json = nlohmann::ordered_json;

namespace {

const char *      DEFAULT_INSTRUCTION = "Represent the user's input.";
const std::string QWEN_IMAGE_MARKER   = "<|vision_start|><|image_pad|><|vision_end|>";
const std::string QWEN_VIDEO_MARKER   = "<|vision_start|><|video_pad|><|vision_end|>";

struct vl_video_item {
    std::optional<std::string> path;
    std::vector<std::string>   frames;
};

struct vl_input_item {
    std::vector<std::string>   texts;
    std::vector<std::string>   images;
    std::vector<vl_video_item> videos;
    std::optional<std::string> instruction;
    std::optional<float>       fps;
    std::optional<int>         max_frames;
};

struct embedding_result {
    std::vector<float> values;
    int                rows = 0;
};

std::vector<std::string> split_lines(const std::string & s, const std::string & separator = "\n") {
    std::vector<std::string> lines;
    size_t                   start = 0;
    size_t                   end   = s.find(separator);

    while (end != std::string::npos) {
        lines.push_back(s.substr(start, end - start));
        start = end + separator.length();
        end   = s.find(separator, start);
    }

    lines.push_back(s.substr(start));

    return lines;
}

bool ends_with_punctuation(const std::string & s) {
    if (s.empty()) {
        return false;
    }

    const unsigned char c = (unsigned char) s.back();
    if (std::ispunct(c)) {
        return true;
    }

    auto ends_with_literal = [&](const char * suffix) {
        const size_t suffix_len = std::strlen(suffix);
        return s.size() >= suffix_len && s.compare(s.size() - suffix_len, suffix_len, suffix) == 0;
    };

    return ends_with_literal("。") || ends_with_literal("！") || ends_with_literal("？");
}

std::string normalize_instruction(const std::optional<std::string> & instruction) {
    std::string result = instruction.value_or(DEFAULT_INSTRUCTION);
    result             = string_strip(result);

    if (!result.empty() && !ends_with_punctuation(result)) {
        result += '.';
    }

    return result;
}

bool is_image_path(const std::string & path) {
    static const std::vector<std::string> exts = { ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif" };

    std::string clean_path = path;
    if (string_starts_with(clean_path, "file://")) {
        clean_path = clean_path.substr(7);
    }

    std::string lower = clean_path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char) std::tolower(c); });

    for (const auto & ext : exts) {
        if (lower.size() >= ext.size() && lower.compare(lower.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }

    return false;
}

std::vector<vl_input_item> parse_json_inputs(const std::string & inputs_json) {
    json parsed = json::parse(inputs_json);
    if (!parsed.is_array()) {
        throw std::runtime_error("inputs must be a JSON array");
    }

    std::vector<vl_input_item> inputs;
    inputs.reserve(parsed.size());

    for (size_t i = 0; i < parsed.size(); ++i) {
        const auto & item = parsed.at(i);
        if (!item.is_object()) {
            throw std::runtime_error(string_format("inputs[%zu] must be a JSON object", i));
        }

        vl_input_item input;

        auto parse_string_or_string_array = [&](const char * key) -> std::vector<std::string> {
            if (!item.contains(key) || item.at(key).is_null()) {
                return {};
            }

            const auto & value = item.at(key);
            if (value.is_string()) {
                return { value.get<std::string>() };
            }

            if (!value.is_array()) {
                throw std::runtime_error(string_format("inputs[%zu].%s must be a string or array of strings", i, key));
            }

            std::vector<std::string> result;
            result.reserve(value.size());
            for (size_t j = 0; j < value.size(); ++j) {
                if (!value.at(j).is_string()) {
                    throw std::runtime_error(string_format("inputs[%zu].%s[%zu] must be a string", i, key, j));
                }
                result.push_back(value.at(j).get<std::string>());
            }
            return result;
        };

        auto parse_video_items = [&]() -> std::vector<vl_video_item> {
            if (!item.contains("video") || item.at("video").is_null()) {
                return {};
            }

            const auto & value = item.at("video");
            if (value.is_string()) {
                return {
                    vl_video_item{ value.get<std::string>(), {} }
                };
            }

            if (!value.is_array()) {
                throw std::runtime_error(string_format(
                    "inputs[%zu].video must be a string, an array of frame paths, or an array of videos", i));
            }

            if (value.empty()) {
                return {};
            }

            if (value.at(0).is_string()) {
                std::vector<std::string> strings;
                strings.reserve(value.size());
                for (size_t j = 0; j < value.size(); ++j) {
                    if (!value.at(j).is_string()) {
                        throw std::runtime_error(
                            string_format("inputs[%zu].video[%zu] must be a string when video is a flat array", i, j));
                    }
                    strings.push_back(value.at(j).get<std::string>());
                }

                if (is_image_path(strings.front())) {
                    return {
                        vl_video_item{ std::nullopt, std::move(strings) }
                    };
                }

                std::vector<vl_video_item> videos;
                videos.reserve(strings.size());
                for (auto & s : strings) {
                    videos.push_back({ std::move(s), {} });
                }
                return videos;
            }

            std::vector<vl_video_item> videos;
            videos.reserve(value.size());
            for (size_t j = 0; j < value.size(); ++j) {
                const auto & entry = value.at(j);
                if (entry.is_string()) {
                    videos.push_back({ entry.get<std::string>(), {} });
                    continue;
                }

                if (!entry.is_array()) {
                    throw std::runtime_error(
                        string_format("inputs[%zu].video[%zu] must be a string or an array of frame paths", i, j));
                }

                vl_video_item video;
                video.frames.reserve(entry.size());
                for (size_t k = 0; k < entry.size(); ++k) {
                    if (!entry.at(k).is_string()) {
                        throw std::runtime_error(
                            string_format("inputs[%zu].video[%zu][%zu] must be a string", i, j, k));
                    }
                    video.frames.push_back(entry.at(k).get<std::string>());
                }
                videos.push_back(std::move(video));
            }

            return videos;
        };

        input.texts  = parse_string_or_string_array("text");
        input.images = parse_string_or_string_array("image");
        input.videos = parse_video_items();

        if (item.contains("instruction") && !item.at("instruction").is_null()) {
            if (!item.at("instruction").is_string()) {
                throw std::runtime_error(string_format("inputs[%zu].instruction must be a string", i));
            }
            input.instruction = item.at("instruction").get<std::string>();
        }

        if (item.contains("fps") && !item.at("fps").is_null()) {
            if (!item.at("fps").is_number()) {
                throw std::runtime_error(string_format("inputs[%zu].fps must be a number", i));
            }
            input.fps = item.at("fps").get<float>();
        }

        if (item.contains("max_frames") && !item.at("max_frames").is_null()) {
            if (!item.at("max_frames").is_number_integer()) {
                throw std::runtime_error(string_format("inputs[%zu].max_frames must be an integer", i));
            }
            input.max_frames = item.at("max_frames").get<int>();
        }

        inputs.push_back(std::move(input));
    }

    return inputs;
}

std::vector<vl_input_item> build_inputs_from_params(const common_params & params) {
    if (!params.inputs_json.empty()) {
        return parse_json_inputs(params.inputs_json);
    }

    std::vector<vl_input_item> inputs;
    for (const auto & prompt : split_lines(params.prompt, params.embd_sep)) {
        vl_input_item item;
        item.texts.push_back(prompt);
        inputs.push_back(std::move(item));
    }
    return inputs;
}

bool has_multimodal_content(const vl_input_item & input) {
    return !input.images.empty() || !input.videos.empty();
}

std::string summarize_input_label(const vl_input_item & input) {
    if (!input.texts.empty()) {
        return input.texts.front();
    }
    if (!input.images.empty()) {
        return "[image]";
    }
    if (!input.videos.empty()) {
        return "[video]";
    }
    return "[empty]";
}

std::vector<common_chat_msg> build_conversation(const vl_input_item & input) {
    std::vector<common_chat_msg> messages;

    common_chat_msg system;
    system.role = "system";
    system.content_parts.push_back({ "text", normalize_instruction(input.instruction) });
    messages.push_back(std::move(system));

    common_chat_msg user;
    user.role = "user";

    std::string merged_text;
    for (const auto & text : input.texts) {
        merged_text += text;
    }

    if (input.images.empty() && input.videos.empty()) {
        user.content = merged_text.empty() ? "NULL" : merged_text;
        messages.push_back(std::move(user));
        return messages;
    }

    for (size_t i = 0; i < input.videos.size(); ++i) {
        user.content_parts.push_back({ "media_marker", QWEN_VIDEO_MARKER });
    }

    for (size_t i = 0; i < input.images.size(); ++i) {
        user.content_parts.push_back({ "media_marker", QWEN_IMAGE_MARKER });
    }

    if (!merged_text.empty()) {
        user.content_parts.push_back({ "text", std::move(merged_text) });
    }

    if (user.content_parts.empty()) {
        user.content_parts.push_back({ "text", "NULL" });
    }

    messages.push_back(std::move(user));
    return messages;
}

std::string format_input_prompt(const common_chat_templates * tmpls, const vl_input_item & input) {
    common_chat_templates_inputs chat_inputs;
    chat_inputs.use_jinja             = true;
    chat_inputs.messages              = build_conversation(input);
    chat_inputs.add_generation_prompt = true;
    chat_inputs.add_bos               = true;
    chat_inputs.add_eos               = true;
    return common_chat_templates_apply(tmpls, chat_inputs).prompt;
}

std::string rewrite_multimodal_markers(std::string prompt) {
    string_replace_all(prompt, QWEN_IMAGE_MARKER, mtmd_default_marker());
    string_replace_all(prompt, QWEN_VIDEO_MARKER, mtmd_default_marker());
    return prompt;
}

mtmd::bitmap load_bitmap_from_path(mtmd_context * ctx_vision, const std::string & path) {
    if (string_starts_with(path, "http://") || string_starts_with(path, "https://") ||
        string_starts_with(path, "oss://")) {
        throw std::runtime_error("remote media URLs are not supported by vl-embedding yet");
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

void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_embd_out, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    llama_memory_clear(llama_get_memory(ctx), true);

    LOG_INF("%s: n_tokens = %d\n", __func__, batch.n_tokens);
    if (llama_decode(ctx, batch) < 0) {
        throw std::runtime_error("failed to process prompt");
    }

    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int i = 0; i < batch.n_tokens; i++) {
            const float * embd = llama_get_embeddings_ith(ctx, i);
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
            common_embd_normalize(embd, output + i * n_embd_out, n_embd_out, embd_norm);
        }
    } else {
        const float * embd = llama_get_embeddings_seq(ctx, 0);
        GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        common_embd_normalize(embd, output, n_embd_out, embd_norm);
    }
}

embedding_result encode_text_only(llama_context *     ctx,
                                  const llama_vocab * vocab,
                                  const std::string & prompt,
                                  int32_t             n_batch,
                                  int                 n_embd_out,
                                  int                 embd_normalize,
                                  bool                verbose_prompt) {
    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, true, true);
    if ((int32_t) tokens.size() > n_batch) {
        throw std::runtime_error(
            string_format("number of tokens in input (%lld) exceeds batch size (%d), increase batch size and re-run",
                          (long long) tokens.size(), n_batch));
    }

    if (verbose_prompt) {
        LOG_INF("%s: formatted prompt: '%s'\n", __func__, prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, tokens.size());
        for (size_t j = 0; j < tokens.size(); ++j) {
            LOG("%6d -> '%s'\n", tokens[j], common_token_to_piece(ctx, tokens[j]).c_str());
        }
        LOG("\n");
    }

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const int                     rows         = pooling_type == LLAMA_POOLING_TYPE_NONE ? (int) tokens.size() : 1;

    embedding_result result;
    result.rows = rows;
    result.values.resize((size_t) rows * n_embd_out);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    batch_add_seq(batch, tokens, 0);
    batch_decode(ctx, batch, result.values.data(), n_embd_out, embd_normalize);
    llama_batch_free(batch);

    return result;
}

embedding_result encode_multimodal(mtmd_context *        ctx_vision,
                                   llama_context *       ctx,
                                   const vl_input_item & input,
                                   const std::string &   prompt,
                                   int32_t               n_batch,
                                   int                   n_embd_out,
                                   int                   embd_normalize) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        throw std::runtime_error("multimodal input currently requires --pooling mean/cls/last/rank, not none");
    }

    if (!input.videos.empty()) {
        throw std::runtime_error(
            "video input is not supported in vl-embedding yet; mtmd public API currently does not expose video "
            "ingestion");
    }

    mtmd::bitmaps bitmaps;
    for (const auto & image : input.images) {
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

    const float * embd = llama_get_embeddings_seq(ctx, 0);
    GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");

    embedding_result result;
    result.rows = 1;
    result.values.resize(n_embd_out);
    common_embd_normalize(embd, result.values.data(), n_embd_out, embd_normalize);
    return result;
}

void print_raw_embeddings(const float *           emb,
                          int                     n_embd_count,
                          int                     n_embd,
                          const llama_model *     model,
                          enum llama_pooling_type pooling_type,
                          int                     embd_normalize) {
    const uint32_t n_cls_out = llama_model_n_cls_out(model);
    const bool     is_rank   = (pooling_type == LLAMA_POOLING_TYPE_RANK);
    const int      cols      = is_rank ? std::min<int>(n_embd, (int) n_cls_out) : n_embd;

    for (int j = 0; j < n_embd_count; ++j) {
        for (int i = 0; i < cols; ++i) {
            if (embd_normalize == 0) {
                LOG("%1.0f%s", emb[j * n_embd + i], (i + 1 < cols ? " " : ""));
            } else {
                LOG("%1.7f%s", emb[j * n_embd + i], (i + 1 < cols ? " " : ""));
            }
        }
        LOG("\n");
    }
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

    std::vector<vl_input_item> inputs;
    try {
        inputs = build_inputs_from_params(params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to parse inputs: %s\n", __func__, e.what());
        return 1;
    }

    const bool needs_vision = std::any_of(inputs.begin(), inputs.end(), has_multimodal_content);
    if (needs_vision && params.mmproj.path.empty()) {
        LOG_ERR("%s: multimodal inputs require --mmproj\n", __func__);
        return 1;
    }

    const int n_seq_max = llama_max_parallel_sequences();

    if (params.n_parallel == 1) {
        LOG_INF("%s: n_parallel == 1 -> unified KV cache is enabled\n", __func__);
        params.kv_unified = true;
        params.n_parallel = n_seq_max;
    }

    if (params.n_batch < params.n_ctx) {
        LOG_WRN("%s: setting batch size to %d\n", __func__, params.n_ctx);
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

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        LOG_ERR("%s: computing embeddings in encoder-decoder models is not supported\n", __func__);
        return 1;
    }

    const llama_vocab *           vocab        = llama_model_get_vocab(model);
    const int                     n_ctx_train  = llama_model_n_ctx_train(model);
    const int                     n_ctx        = llama_n_ctx(ctx);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train,
                n_ctx);
    }

    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

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

    const int                     n_embd_out = llama_model_n_embd_out(model);
    std::vector<embedding_result> results;
    std::vector<std::string>      labels;
    results.reserve(inputs.size());
    labels.reserve(inputs.size());

    try {
        for (const auto & input : inputs) {
            const std::string prompt = format_input_prompt(tmpls.get(), input);
            labels.push_back(summarize_input_label(input));

            if (params.verbose_prompt) {
                LOG_INF("%s: formatted prompt: %s\n", __func__, prompt.c_str());
            }

            if (has_multimodal_content(input)) {
                results.push_back(encode_multimodal(ctx_vision.get(), ctx, input, prompt, params.n_batch, n_embd_out,
                                                    params.embd_normalize));
            } else {
                results.push_back(encode_text_only(ctx, vocab, prompt, params.n_batch, n_embd_out,
                                                   params.embd_normalize, params.verbose_prompt));
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        llama_backend_free();
        return 1;
    }

    int n_embd_count = 0;
    for (const auto & result : results) {
        n_embd_count += result.rows;
    }

    std::vector<float> embeddings((size_t) n_embd_count * n_embd_out, 0.0f);
    int                row_offset = 0;
    for (const auto & result : results) {
        std::copy(result.values.begin(), result.values.end(), embeddings.begin() + (size_t) row_offset * n_embd_out);
        row_offset += result.rows;
    }
    float * emb = embeddings.data();

    if (params.embd_out.empty()) {
        LOG("\n");

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            for (int j = 0; j < n_embd_count; j++) {
                LOG("embedding %d: ", j);
                for (int i = 0; i < std::min(3, n_embd_out); i++) {
                    if (params.embd_normalize == 0) {
                        LOG("%6.0f ", emb[j * n_embd_out + i]);
                    } else {
                        LOG("%9.6f ", emb[j * n_embd_out + i]);
                    }
                }
                LOG(" ... ");
                for (int i = n_embd_out - 3; i < n_embd_out; i++) {
                    if (params.embd_normalize == 0) {
                        LOG("%6.0f ", emb[j * n_embd_out + i]);
                    } else {
                        LOG("%9.6f ", emb[j * n_embd_out + i]);
                    }
                }
                LOG("\n");
            }
        } else if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
            const uint32_t           n_cls_out = llama_model_n_cls_out(model);
            std::vector<std::string> cls_out_labels;

            for (uint32_t i = 0; i < n_cls_out; i++) {
                const char *      label = llama_model_cls_label(model, i);
                const std::string label_i(label == nullptr ? "" : label);
                cls_out_labels.emplace_back(label_i.empty() ? std::to_string(i) : label_i);
            }

            for (int j = 0; j < n_embd_count; j++) {
                for (uint32_t i = 0; i < n_cls_out; i++) {
                    if (n_cls_out == 1) {
                        LOG("rerank score %d: %8.3f\n", j, emb[j * n_embd_out]);
                    } else {
                        LOG("rerank score %d: %8.3f [%s]\n", j, emb[j * n_embd_out + i], cls_out_labels[i].c_str());
                    }
                }
            }
        } else {
            for (int j = 0; j < (int) results.size(); j++) {
                LOG("embedding %d: ", j);
                for (int i = 0; i < ((int) results.size() > 1 ? std::min(16, n_embd_out) : n_embd_out); i++) {
                    if (params.embd_normalize == 0) {
                        LOG("%6.0f ", emb[j * n_embd_out + i]);
                    } else {
                        LOG("%9.6f ", emb[j * n_embd_out + i]);
                    }
                }
                LOG("\n");
            }

            if (results.size() > 1) {
                LOG("\n");
                LOG("cosine similarity matrix:\n\n");
                for (const auto & label : labels) {
                    LOG("%6.6s ", label.c_str());
                }
                LOG("\n");
                for (int i = 0; i < (int) results.size(); i++) {
                    for (int j = 0; j < (int) results.size(); j++) {
                        float sim = common_embd_similarity_cos(emb + i * n_embd_out, emb + j * n_embd_out, n_embd_out);
                        LOG("%6.2f ", sim);
                    }
                    LOG("%1.10s", labels[i].c_str());
                    LOG("\n");
                }
            }
        }
    }

    if (params.embd_out == "json" || params.embd_out == "json+" || params.embd_out == "array") {
        const bool notArray = params.embd_out != "array";

        LOG(notArray ? "{\n  \"object\": \"list\",\n  \"data\": [\n" : "[");
        for (int j = 0;;) {
            if (notArray) {
                LOG("    {\n      \"object\": \"embedding\",\n      \"index\": %d,\n      \"embedding\": ", j);
            }
            LOG("[");
            for (int i = 0;;) {
                LOG(params.embd_normalize == 0 ? "%1.0f" : "%1.7f", emb[j * n_embd_out + i]);
                i++;
                if (i < n_embd_out) {
                    LOG(",");
                } else {
                    break;
                }
            }
            LOG(notArray ? "]\n    }" : "]");
            j++;
            if (j < n_embd_count) {
                LOG(notArray ? ",\n" : ",");
            } else {
                break;
            }
        }
        LOG(notArray ? "\n  ]" : "]\n");

        if (params.embd_out == "json+" && results.size() > 1 && pooling_type != LLAMA_POOLING_TYPE_NONE) {
            LOG(",\n  \"cosineSimilarity\": [\n");
            for (int i = 0;;) {
                LOG("    [");
                for (int j = 0;;) {
                    float sim = common_embd_similarity_cos(emb + i * n_embd_out, emb + j * n_embd_out, n_embd_out);
                    LOG("%6.2f", sim);
                    j++;
                    if (j < (int) results.size()) {
                        LOG(", ");
                    } else {
                        break;
                    }
                }
                LOG(" ]");
                i++;
                if (i < (int) results.size()) {
                    LOG(",\n");
                } else {
                    break;
                }
            }
            LOG("\n  ]");
        }

        if (notArray) {
            LOG("\n}\n");
        }
    } else if (params.embd_out == "raw") {
        print_raw_embeddings(emb, n_embd_count, n_embd_out, model, pooling_type, params.embd_normalize);
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();
    return 0;
}
