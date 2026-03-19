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
    parser = argparse.ArgumentParser(description="Measure per-tensor roundtrip sensitivity by selectively protecting tensors from quantization.")
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
    parser.add_argument("--val-batch-size", type=int, default=32768)
    parser.add_argument("--roundtrip-val-max-tokens", type=int, default=262144)
    parser.add_argument("--int8-axis-mode", default="auto")
    parser.add_argument("--int8-residual-rank", type=int, default=1)
    parser.add_argument("--int8-residual-budget-bytes", type=int, default=65536)
    parser.add_argument("--weight-quantization-bits", type=int, default=8)
    parser.add_argument("--embed-quantization-bits", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--log-path", default="")
    return parser.parse_args()


def set_env_from_args(args: argparse.Namespace) -> None:
    env_map = {
        "DATA_PATH": args.data_path,
        "TOKENIZER_PATH": args.tokenizer_path,
        "VOCAB_SIZE": args.vocab_size,
        "NUM_LAYERS": args.num_layers,
        "NUM_UNIQUE_BLOCKS": args.num_unique_blocks,
        "MODEL_DIM": args.model_dim,
        "EMBED_DIM": args.embed_dim,
        "NUM_HEADS": args.num_heads,
        "NUM_KV_HEADS": args.num_kv_heads,
        "MLP_MULT": args.mlp_mult,
        "TIE_EMBEDDINGS": int(args.tie_embeddings),
        "ROPE_BASE": args.rope_base,
        "QK_GAIN_INIT": args.qk_gain_init,
        "WINDOW_SIZE": args.window_size,
        "TRAIN_SEQ_LEN": args.train_seq_len,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "ROUNDTRIP_VAL_MAX_TOKENS": args.roundtrip_val_max_tokens,
        "INT8_AXIS_MODE": args.int8_axis_mode,
        "INT8_RESIDUAL_RANK": args.int8_residual_rank,
        "INT8_RESIDUAL_BUDGET_BYTES": args.int8_residual_budget_bytes,
        "WEIGHT_QUANTIZATION_BITS": args.weight_quantization_bits,
        "EMBED_QUANTIZATION_BITS": args.embed_quantization_bits,
    }
    for key, value in env_map.items():
        os.environ[key] = str(value)


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


def quant_blob_size(quant_obj: dict[str, object]) -> int:
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    return len(zlib.compress(buf.getvalue(), level=9))


def build_model(tg, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
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
    return model


def build_eval_inputs(tg, args: argparse.Namespace, device: torch.device):
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = tg.load_validation_tokens(
        str(Path(args.data_path) / "fineweb_val_*.bin"),
        args.train_seq_len,
        args.roundtrip_val_max_tokens,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    eval_args = SimpleNamespace(
        train_seq_len=args.train_seq_len,
        eval_seq_len=0,
        eval_stride=0,
        sw_eval_batch=32,
        val_batch_size=args.val_batch_size,
        eval_cache_mix_weight=0.0,
        eval_bigram_mix_weight=0.0,
        eval_cache_size=0,
    )
    return eval_args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def eval_quant_obj(label: str, quant_obj: dict[str, object], model: torch.nn.Module, tg, device: torch.device, eval_args, eval_inputs, code_bytes: int, log) -> tuple[float, int]:
    artifact_bytes = quant_blob_size(quant_obj) + code_bytes
    model.load_state_dict(tg.dequantize_state_dict_int8(quant_obj), strict=True)
    torch.cuda.empty_cache()
    val_loss, val_bpb = tg.eval_val(
        eval_args,
        model,
        0,
        1,
        device,
        8,
        eval_inputs[0],
        eval_inputs[1],
        eval_inputs[2],
        eval_inputs[3],
    )
    log(f"{label}: val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} artifact_bytes:{artifact_bytes}")
    return val_bpb, artifact_bytes


def keep_float_variant(baseline_quant_obj: dict[str, object], raw_state: dict[str, torch.Tensor], name: str, tg) -> dict[str, object]:
    quant_obj = copy.deepcopy(baseline_quant_obj)
    tensor = raw_state[name].detach().cpu().contiguous()
    if name in quant_obj["quantized"]:
        del quant_obj["quantized"][name]
        del quant_obj["scales"][name]
        del quant_obj["dtypes"][name]
        if "qmeta" in quant_obj and name in quant_obj["qmeta"]:
            del quant_obj["qmeta"][name]
        if "rank1_residual" in quant_obj and name in quant_obj["rank1_residual"]:
            del quant_obj["rank1_residual"][name]
        passthrough_orig_dtypes = quant_obj.setdefault("passthrough_orig_dtypes", {})
        kept = tg.keep_float_tensor(name, tensor, passthrough_orig_dtypes)
        quant_obj["passthrough"][name] = kept
    return quant_obj


def main() -> None:
    args = parse_args()
    set_env_from_args(args)
    log = build_log_fn(args.log_path)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import train_gpt as tg

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    raw_state = torch.load(args.checkpoint_path, map_location="cpu")
    model = build_model(tg, args, device)
    eval_args, *eval_inputs = build_eval_inputs(tg, args, device)
    code_bytes = len((root / "train_gpt.py").read_text(encoding="utf-8").encode("utf-8"))

    baseline_quant_obj, _ = tg.quantize_state_dict_int8(raw_state)
    baseline_bpb, baseline_artifact = eval_quant_obj(
        "baseline_export",
        baseline_quant_obj,
        model,
        tg,
        device,
        eval_args,
        eval_inputs,
        code_bytes,
        log,
    )

    candidates = [
        (name, tensor.numel())
        for name, tensor in raw_state.items()
        if name in baseline_quant_obj["quantized"] and tensor.ndim == 2
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    candidates = candidates[: args.max_candidates]
    log(f"sensitivity_candidates:{len(candidates)}")
    for name, numel in candidates:
        variant = keep_float_variant(baseline_quant_obj, raw_state, name, tg)
        val_bpb, artifact_bytes = eval_quant_obj(
            f"protect:{name}",
            variant,
            model,
            tg,
            device,
            eval_args,
            eval_inputs,
            code_bytes,
            log,
        )
        delta_bpb = baseline_bpb - val_bpb
        delta_bytes = artifact_bytes - baseline_artifact
        per_kb = (delta_bpb / max(delta_bytes, 1)) * 1024.0 if delta_bytes > 0 else 0.0
        log(
            f"protect_delta:{name} numel:{numel} val_bpb:{delta_bpb:+.8f} "
            f"artifact_bytes:{delta_bytes:+d} bpb_per_kb:{per_kb:+.8f}"
        )


if __name__ == "__main__":
    main()
