# Wan2.2 T2V 14B quantized attention

These four deployment configurations select Dense MXFP8/MXFP4 or BSA FP8/MXFP4.
They require Ascend NPU, a matching TorchNPU/CANN stack and the MindIE-SD build
that exports `quant_attention` and `get_bsa_supported_precisions`.
The Dense API must include both MXFP8 and MXFP4 implementations; the FP8-only
entry point is insufficient. The BSA capability query remains a required
dependency of these configurations, including its matching native support.

Run from the repository root, replacing the model path and configuration name:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model /models/Wan2.2-T2V-A14B \
  --deploy-config examples/offline_inference/text_to_video/wan22_quant_attention/fa_mxfp8.yaml \
  --num-inference-steps 50 --num-frames 81 --height 720 --width 1280 \
  --prompt "A cat walking through a sunlit garden" --seed 42 \
  --enable-cpu-offload --vae-use-tiling --enforce-eager
```

Use `fa_mxfp4.yaml`, `bsa_fp8.yaml` or `bsa_mxfp4.yaml` for the other paths.
The example uses one NPU and CPU offload; adjust placement/offload for available
memory. The command is a hardware qualification recipe, not a measured result.

`skip_layers: "0,39"` selects the same zero-based block indices in both experts.
`skip_steps: "0,49"` selects the first and last of the 50 global denoising steps;
the counter does not reset at expert switching. These illustrative selectors
need model quality validation. BSA uses unquantized sparse attention on a skip;
its sparsity remains 0.8. Cross-attention is always unquantized.

The fallback list is deliberately empty so unsupported quantization cannot mask
a missing kernel during qualification. For production, explicitly configure a
chain such as `[mxfp8, float]` for Dense MXFP4 or `[fp8, float]` for BSA MXFP4.
Kernel execution errors always propagate. BSA precision support is queried from
MindIE-SD's native capability API before dispatch; sparse fallback remains sparse.

CPU contract tests exercise configuration and dispatch, not NPU numerical
correctness, graph compatibility, latency, video quality or VBench acceptance.
Validate those separately on the target hardware with identical sampling inputs.
