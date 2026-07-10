#include "models.h"

#include <cstring>

ggml_cgraph * clip_graph_qwen3vl::build() {
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size = 1;
    const int n_pos      = n_patches;
    const int rope_dim   = d_head / 2;

    norm_type norm_t = NORM_TYPE_NORMAL;
    const char * debug_tensor = std::getenv("MTMD_DEBUG_TENSOR");

    auto maybe_keep_output = [debug_tensor](ggml_tensor * t, const char * name) {
        if (debug_tensor != nullptr && strcmp(debug_tensor, name) == 0) {
            ggml_set_output(t);
        }
    };

    ggml_tensor * learned_pos_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_pos);
    ggml_set_name(learned_pos_embd, "learned_pos_embd");
    ggml_set_input(learned_pos_embd);

    ggml_tensor * rope_cos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rope_dim, n_pos);
    ggml_set_name(rope_cos, "rope_cos");
    ggml_set_input(rope_cos);

    ggml_tensor * rope_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rope_dim, n_pos);
    ggml_set_name(rope_sin, "rope_sin");
    ggml_set_input(rope_sin);

    ggml_tensor * ln_one = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(ln_one, "ln_one");
    ggml_set_input(ln_one);

    ggml_tensor * ln_eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(ln_eps, "ln_eps");
    ggml_set_input(ln_eps);

    ggml_tensor * inp = build_inp_with_temporal_merge();

    // spatial merge
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // add patch bias
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
        maybe_keep_output(inp, "patch_bias");
    }

    // add absolute position embedding (interpolated on the host, see
    // qwen3vl_fast_pos_embed_interpolate in clip.cpp)
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "inp_pos_emb", -1);
    maybe_keep_output(inp, "inp_pos_emb");

    ggml_tensor * inpL = inp;
    ggml_tensor * rope_cos_3d = ggml_reshape_3d(ctx0, rope_cos, rope_dim, 1, n_pos);
    ggml_tensor * rope_sin_3d = ggml_reshape_3d(ctx0, rope_sin, rope_dim, 1, n_pos);

    // HF LayerNorm computed manually via ln_eps/ln_one inputs, to avoid
    // relying on the generic build_norm() path for this model.
    auto apply_norm = [&](ggml_tensor * cur, ggml_tensor * mw, ggml_tensor * mb, int il) -> ggml_tensor * {
        if (norm_t == NORM_TYPE_RMS) {
            return build_norm(cur, mw, mb, norm_t, eps, il);
        }

        ggml_tensor * mean = ggml_mean(ctx0, cur);
        ggml_tensor * centered = ggml_sub(ctx0, cur, mean);
        ggml_tensor * var = ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, centered)), 1.0f / cur->ne[0]);
        ggml_tensor * denom = ggml_sqrt(ctx0, ggml_add1(ctx0, var, ln_eps));
        ggml_tensor * inv_std = ggml_div(ctx0, ggml_repeat(ctx0, ln_one, denom), denom);
        cur = ggml_mul(ctx0, centered, inv_std);

        if (mw) {
            cur = ggml_mul(ctx0, cur, mw);
            cb(cur, "norm_w", il);
        }

        if (mb) {
            cur = ggml_add(ctx0, cur, mb);
            cb(cur, "norm_b", il);
        }

        return cur;
    };

    // vision RoPE applied via precomputed cos/sin tables (see
    // qwen3vl_build_vision_rope_tables in clip.cpp), not ggml_rope_multi.
    auto apply_vision_rope = [&](ggml_tensor * cur, const char * name, int il) -> ggml_tensor * {
        ggml_tensor * first = ggml_view_3d(ctx0, cur, rope_dim, n_head, n_pos,
                /* nb1    */ cur->nb[1],
                /* nb2    */ cur->nb[2],
                /* offset */ 0);
        ggml_tensor * second = ggml_view_3d(ctx0, cur, rope_dim, n_head, n_pos,
                /* nb1    */ cur->nb[1],
                /* nb2    */ cur->nb[2],
                /* offset */ ggml_row_size(cur->type, rope_dim));

        ggml_tensor * cos = ggml_repeat(ctx0, rope_cos_3d, first);
        ggml_tensor * sin = ggml_repeat(ctx0, rope_sin_3d, first);

        ggml_tensor * first_rot = ggml_sub(ctx0,
                ggml_mul(ctx0, first,  cos),
                ggml_mul(ctx0, second, sin));
        ggml_tensor * second_rot = ggml_add(ctx0,
                ggml_mul(ctx0, first,  sin),
                ggml_mul(ctx0, second, cos));

        ggml_tensor * out = ggml_concat(ctx0, first_rot, second_rot, 0);
        out = ggml_cont_3d(ctx0, out, d_head, n_head, n_pos);
        cb(out, name, il);
        return out;
    };

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = apply_norm(inpL, model.pre_ln_w, model.pre_ln_b, -1);
    }

    // deepstack features (stack along the feature dimension), [n_embd * len(deepstack_layers), n_patches_x * n_patches_y, batch_size]
    ggml_tensor * deepstack_features = nullptr;
    const int merge_factor = hparams.n_merge > 0 ? hparams.n_merge * hparams.n_merge : 4; // default 2x2=4 for qwen3vl

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = apply_norm(cur, layer.ln_1_w, layer.ln_1_b, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            cur = build_mm(layer.qkv_w, cur);
            cur = ggml_add(ctx0, cur, layer.qkv_b);
            cb(cur, "qkv", il);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ 0);

            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, n_embd));

            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, 2 * n_embd));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // apply vision RoPE (precomputed cos/sin, rotate-half style; see
            // apply_vision_rope above), not the generic ggml_rope_multi path
            Qcur = apply_vision_rope(Qcur, "Qcur_rope", il);
            Kcur = apply_vision_rope(Kcur, "Kcur_rope", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = apply_norm(cur, layer.ln_2_w, layer.ln_2_b, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn (not gated for qwen3vl vision tower, matching HF)
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        if (layer.has_deepstack()) {
            ggml_tensor * feat = ggml_reshape_3d(ctx0, cur, n_embd * merge_factor, n_pos / merge_factor, batch_size);
            feat = apply_norm(feat, layer.deepstack_norm_w, layer.deepstack_norm_b, il);
            feat = build_ffn(feat,
                layer.deepstack_fc1_w, layer.deepstack_fc1_b,
                nullptr, nullptr,
                layer.deepstack_fc2_w, layer.deepstack_fc2_b,
                ffn_op_type::FFN_GELU, il);

            if(!deepstack_features) {
                deepstack_features = feat;
            } else {
                // concat along the feature dimension
                deepstack_features = ggml_concat(ctx0, deepstack_features, feat, 0);
            }
        }

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = apply_norm(inpL, model.post_ln_w, model.post_ln_b, n_layer);
    }

    // multimodal projection
    ggml_tensor * embeddings = inpL;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * merge_factor, n_pos / merge_factor, batch_size);

    embeddings = build_ffn(embeddings,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        ffn_op_type::FFN_GELU, n_layer);

    if (deepstack_features) {
        embeddings = ggml_concat(ctx0, embeddings, deepstack_features, 0);
    } // concat along the feature dimension

    embeddings = ggml_cont(ctx0, embeddings);

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
