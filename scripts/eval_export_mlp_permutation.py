from __future__ import annotations

import argparse
import copy
import io
import os
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact export-side MLP hidden-unit permutation on a raw checkpoint.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-unique-blocks", type=int, required=True)
    parser.add_argument("--model-dim", type=int, required=True)
    parser.add_argument("--embed-dim", type=int, default=0)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--tie-embeddings", type=int, default=1)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--eval-seq-len", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--sw-eval-batch", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=32768)
    parser.add_argument("--roundtrip-val-max-tokens", type=int, default=262144)
    parser.add_argument("--int8-axis-mode", default="auto")
    parser.add_argument("--int8-residual-rank", type=int, default=1)
    parser.add_argument("--int8-residual-budget-bytes", type=int, default=65536)
    parser.add_argument("--weight-quantization-bits", type=int, default=8)
    parser.add_argument("--embed-quantization-bits", type=int, default=8)
    parser.add_argument("--log-path", default="")
    return parser.parse_args()


def set_env_from_args(args: argparse.Namespace) -> None:
    os.environ["DATA_PATH"] = args.data_path
    os.environ["TOKENIZER_PATH"] = args.tokenizer_path
    os.environ["VOCAB_SIZE"] = str(args.vocab_size)
    os.environ["NUM_LAYERS"] = str(args.num_layers)
    os.environ["NUM_UNIQUE_BLOCKS"] = str(args.num_unique_blocks)
    os.environ["MODEL_DIM"] = str(args.model_dim)
    os.environ["EMBED_DIM"] = str(args.embed_dim)
    os.environ["NUM_HEADS"] = str(args.num_heads)
    os.environ["NUM_KV_HEADS"] = str(args.num_kv_heads)
    os.environ["MLP_MULT"] = str(args.mlp_mult)
    os.environ["TIE_EMBEDDINGS"] = str(int(args.tie_embeddings))
    os.environ["ROPE_BASE"] = str(args.rope_base)
    os.environ["QK_GAIN_INIT"] = str(args.qk_gain_init)
    os.environ["WINDOW_SIZE"] = str(args.window_size)
    os.environ["TRAIN_SEQ_LEN"] = str(args.train_seq_len)
    os.environ["EVAL_SEQ_LEN"] = str(args.eval_seq_len)
    os.environ["EVAL_STRIDE"] = str(args.eval_stride)
    os.environ["SW_EVAL_BATCH"] = str(args.sw_eval_batch)
    os.environ["VAL_BATCH_SIZE"] = str(args.val_batch_size)
    os.environ["ROUNDTRIP_VAL_MAX_TOKENS"] = str(args.roundtrip_val_max_tokens)
    os.environ["INT8_AXIS_MODE"] = args.int8_axis_mode
    os.environ["INT8_RESIDUAL_RANK"] = str(args.int8_residual_rank)
    os.environ["INT8_RESIDUAL_BUDGET_BYTES"] = str(args.int8_residual_budget_bytes)
    os.environ["WEIGHT_QUANTIZATION_BITS"] = str(args.weight_quantization_bits)
    os.environ["EMBED_QUANTIZATION_BITS"] = str(args.embed_quantization_bits)


def build_log_fn(log_path: str):
    log_file = Path(log_path) if log_path else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg)
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as f:
                print(msg, file=f)

    return log


def sample_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    if count <= 1:
        return [0]
    step = (length - 1) / (count - 1)
    return [round(i * step) for i in range(count)]


def build_mlp_permutation(fc: torch.Tensor, proj: torch.Tensor, tg) -> torch.Tensor:
    quant_bits = min(int(os.environ.get("WEIGHT_QUANTIZATION_BITS", "8")), 8)
    q_fc, s_fc = tg.quantize_float_tensor_axis(fc, axis=0, num_bits=quant_bits)
    q_proj_t, s_proj = tg.quantize_float_tensor_axis(proj.t().contiguous(), axis=0, num_bits=quant_bits)
    fc_sample_idx = sample_indices(q_fc.shape[1], min(12, q_fc.shape[1]))
    proj_sample_idx = sample_indices(q_proj_t.shape[1], min(12, q_proj_t.shape[1]))
    q_fc_cpu = q_fc.cpu()
    q_proj_cpu = q_proj_t.cpu()
    s_fc_cpu = s_fc.float().cpu()
    s_proj_cpu = s_proj.float().cpu()
    keys: list[tuple[int, ...]] = []
    for i in range(q_fc_cpu.shape[0]):
        keys.append(
            (
                int(round(float(s_fc_cpu[i].item()) * 1024.0)),
                int(round(float(s_proj_cpu[i].item()) * 1024.0)),
                int(q_fc_cpu[i].abs().sum().item()),
                int(q_proj_cpu[i].abs().sum().item()),
                *[int(q_fc_cpu[i, j].item()) for j in fc_sample_idx],
                *[int(q_proj_cpu[i, j].item()) for j in proj_sample_idx],
            )
        )
    perm = sorted(range(len(keys)), key=keys.__getitem__)
    return torch.tensor(perm, dtype=torch.long)


def apply_mlp_permutation(state_dict: dict[str, torch.Tensor], tg) -> tuple[dict[str, torch.Tensor], int]:
    out = {name: tensor.detach().clone() for name, tensor in state_dict.items()}
    changed = 0
    prefixes = sorted(
        name[: -len(".mlp.fc.weight")]
        for name in state_dict
        if name.endswith(".mlp.fc.weight")
    )
    for prefix in prefixes:
        fc_name = f"{prefix}.mlp.fc.weight"
        proj_name = f"{prefix}.mlp.proj.weight"
        if proj_name not in state_dict:
            continue
        fc = state_dict[fc_name].detach().cpu()
        proj = state_dict[proj_name].detach().cpu()
        perm = build_mlp_permutation(fc, proj, tg)
        out[fc_name] = fc.index_select(0, perm).contiguous()
        out[proj_name] = proj.index_select(1, perm).contiguous()
        changed += 1
    return out, changed


def quant_blob_size(quant_obj: dict[str, object]) -> int:
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    return len(zlib.compress(buf.getvalue(), level=9))


def evaluate_roundtrip_variant(label: str, raw_state_dict: dict[str, torch.Tensor], model: torch.nn.Module, args_eval, eval_inputs, tg, device: torch.device, log) -> tuple[float, int]:
    quant_obj, quant_stats = tg.quantize_state_dict_int8(raw_state_dict)
    artifact_bytes = quant_blob_size(quant_obj) + len(Path("train_gpt.py").read_text(encoding="utf-8").encode("utf-8"))
    model.load_state_dict(tg.dequantize_state_dict_int8(quant_obj), strict=True)
    torch.cuda.empty_cache()
    val_loss, val_bpb = tg.eval_val(
        args_eval,
        model,
        0,
        1,
        device,
        8,
        eval_inputs["val_tokens"],
        eval_inputs["base_bytes_lut"],
        eval_inputs["has_leading_space_lut"],
        eval_inputs["is_boundary_token_lut"],
    )
    log(
        f"{label}: final_int8_zlib_roundtrip_exact val_loss:{val_loss:.8f} "
        f"val_bpb:{val_bpb:.8f} artifact_bytes:{artifact_bytes} "
        f"residual_tensors:{quant_stats['num_rank1_residual_tensors']}"
    )
    return val_bpb, artifact_bytes


def main() -> None:
    args = parse_args()
    set_env_from_args(args)
    log = build_log_fn(args.log_path)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import train_gpt as tg

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for export permutation evaluation")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_tokens = tg.load_validation_tokens(
        str(Path(args.data_path) / "fineweb_val_*.bin"),
        eval_seq_len,
        args.roundtrip_val_max_tokens,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    eval_args = SimpleNamespace(
        train_seq_len=args.train_seq_len,
        eval_seq_len=args.eval_seq_len,
        eval_stride=args.eval_stride,
        sw_eval_batch=args.sw_eval_batch,
        val_batch_size=args.val_batch_size,
        eval_cache_mix_weight=0.0,
        eval_bigram_mix_weight=0.0,
        eval_cache_size=0,
    )
    model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_blocks=args.num_unique_blocks,
        model_dim=args.model_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=bool(args.tie_embeddings),
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        window_size=args.window_size,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, tg.CastedLinear):
            module.float()
    tg.restore_low_dim_params_to_fp32(model)

    raw_state = torch.load(checkpoint_path, map_location="cpu")
    log(f"loaded_checkpoint:{checkpoint_path} tensors:{len(raw_state)}")
    baseline_bpb, baseline_artifact = evaluate_roundtrip_variant(
        "baseline_export",
        raw_state,
        model,
        eval_args,
        {
            "val_tokens": val_tokens,
            "base_bytes_lut": base_bytes_lut,
            "has_leading_space_lut": has_leading_space_lut,
            "is_boundary_token_lut": is_boundary_token_lut,
        },
        tg,
        device,
        log,
    )
    permuted_state, changed = apply_mlp_permutation(raw_state, tg)
    log(f"mlp_permutation:blocks_changed:{changed}")
    perm_bpb, perm_artifact = evaluate_roundtrip_variant(
        "mlp_permuted_export",
        permuted_state,
        model,
        eval_args,
        {
            "val_tokens": val_tokens,
            "base_bytes_lut": base_bytes_lut,
            "has_leading_space_lut": has_leading_space_lut,
            "is_boundary_token_lut": is_boundary_token_lut,
        },
        tg,
        device,
        log,
    )
    delta_bpb = perm_bpb - baseline_bpb
    delta_bytes = perm_artifact - baseline_artifact
    log(f"mlp_permutation_delta: val_bpb:{delta_bpb:+.8f} artifact_bytes:{delta_bytes:+d}")


if __name__ == "__main__":
    main()
