# Wan2.2 T2V 14B 量化 Attention 环境验证操作报告

## 1. 本次交付与验证边界

两仓分支均为 `codex/quant-attention-cleanup`：

| 仓库 | 地址 | 功能提交 |
|---|---|---|
| MindIE-SD | https://gitcode.com/zqxu/MindIE-SD.git | `f5be942` |
| vLLM-Omni | https://github.com/De-cs/vllm-omni.git | `a2d859f1` |

Omni 分支在功能提交之后包含本操作报告。没有提交上游 PR。
本次讨论的“将旋转校验提前到 RFv3 入口”只是建议，尚未修改；当前仍在旋转 helper 中校验。

本地已完成 CPU 接口验证：MindIE 24 passed（另有 23 subtests），Omni 联合测试 97 passed、10 skipped。使用了绕开本地缺失 NPU/vLLM 初始化的测试引导脚本；这些数字不代表下述原生环境命令已执行。两仓串联测试调用真实公开 Python 函数，底层 NPU 算子使用替身。已检查修改的 Python 代码 Ruff 和 diff whitespace。

尚未完成：真实 NPU 算子执行、C++ 编译、图模式、视频质量、VBench、性能收益。以下是待执行步骤，不是验收通过报告。

## 2. 获取代码与安装

在已能运行 Wan2.2 T2V-A14B 的 Linux Ascend 环境中操作，保留当前已匹配的驱动、CANN、Torch/TorchNPU、vLLM/vLLM-Ascend 组合。量化算子必须受当前硬件及软件栈支持；不通过随意升级某一个包修复 ABI 不匹配。

使用独立目录，避免覆盖环境已有工作树：

```bash
set -euo pipefail
export VALIDATION_ROOT="$PWD/wan22-quant-validation"
mkdir -p "$VALIDATION_ROOT"
cd "$VALIDATION_ROOT"
git clone --branch codex/quant-attention-cleanup https://gitcode.com/zqxu/MindIE-SD.git mindiesd-src
git clone --branch codex/quant-attention-cleanup https://github.com/De-cs/vllm-omni.git omni-src
git -C mindiesd-src rev-parse HEAD
git -C omni-src rev-parse HEAD
mkdir -p results
npu-smi info > results/npu-smi.txt
python -m pip freeze > results/pip-freeze-before.txt
```

私仓认证使用环境已有的 Git 凭据，不将 token 写进命令或日志。重复执行时不要重新 clone，进入已有目录后 fetch 并使用 `git pull --ff-only`，先保留未提交修改。

MindIE 必须重编：本次增加了原生 `block_sparse_attention_version` 查询，只替换 Python 文件不够。下面路径按实际 CANN 安装位置调整：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$VALIDATION_ROOT/mindiesd-src"
unset MINDIESD_SKIP_OPS_BUILD SKIP_ALL_OPS_PLUGIN_BUILD
python -m pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
python setup.py bdist_wheel 2>&1 | tee "$VALIDATION_ROOT/results/mindiesd-build.log"
# 独立 clone 的 dist 下应只有本次构建的 wheel。
python -m pip install --force-reinstall --no-deps dist/mindiesd-*.whl
cd "$VALIDATION_ROOT/omni-src"
VLLM_OMNI_TARGET_DEVICE=npu python -m pip install -v -e . --no-build-isolation 2>&1 | tee "$VALIDATION_ROOT/results/omni-install.log"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m pip check | tee "$VALIDATION_ROOT/results/pip-check.txt"
```

如果依赖安装改变了原有版本，重新确认匹配关系。构建或安装失败时先处理日志中的第一处错误，不跳过原生编译继续测模型。

## 3. 检查实际加载版本和能力

从 Omni 目录运行，避免 MindIE 源码目录遮蔽刚安装的 wheel：

```bash
cd "$VALIDATION_ROOT/omni-src"
python - <<'PY' | tee "$VALIDATION_ROOT/results/capabilities.txt"
import torch
import torch_npu
import mindiesd
import vllm_omni
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('mindiesd:', mindiesd.__file__)
print('omni:', vllm_omni.__file__)
print('npu available:', torch.npu.is_available())
assert torch.npu.is_available()
assert callable(mindiesd.quant_attention_forward)
print('BSA native version:', torch.ops.mindiesd.block_sparse_attention_version())
supported = mindiesd.get_bsa_supported_precisions()
print('BSA precisions:', supported)
for name in ('quant_flash_attn', 'quant_flash_attn_metadata'):
    print(name, getattr(torch.ops.mindiesd, name).default._schema)
assert callable(torch_npu.npu_fused_infer_attention_score_v2)
assert callable(torch_npu.npu_dynamic_mx_quant)
PY
```

完整 BSA 两路径验证要求结果包含 `fp8` 和 `mxfp4`。只有 `bf16, fp8` 时不能声称 MXFP4 可用；空元组表示没有确认到能力，先排查插件加载和 native 库。该查询只检查 native 版本与 schema，不运行 tensor kernel，也不是硬件数值验证。Dense 接口存在同样不等于执行成功。

## 4. 先跑接口和回退测试

在依赖完整的环境中执行以下测试并保存日志；不能把关键测试被 skip 当成通过：

```bash
cd "$VALIDATION_ROOT/omni-src"
python -m pytest \
  tests/diffusion/attention/test_mindie_runtime.py \
  tests/diffusion/attention/test_rainfusion_plan.py \
  tests/diffusion/attention/test_mindie_public_integration.py \
  tests/diffusion/models/wan2_2/test_wan22_mindie_metadata.py \
  tests/platforms/npu/quant/test_kv_quant_npu.py \
  -q -ra -o addopts='' 2>&1 | tee "$VALIDATION_ROOT/results/contracts.log"
```

重点检查：两个 expert 的全局 step 连续；各 expert 内 layer 索引；layer/step 单独与同时命中；cross 不量化；请求间状态隔离；BSA 能力不足按配置回退；算子执行异常直接传播、不重试。`test_mindie_public_integration.py` 还覆盖 Dense 非整块长度和 BSA 65/600 长度，但依然使用算子替身。

## 5. 四条路径真实模型运行

以下使用单卡、CPU offload、VAE tiling、eager 模式。按设备内存调整部署，保持四条路径与基线的设置一致。模型必须是 Wan2.2 **T2V-A14B**，不是 I2V 或 5B。

```bash
cd "$VALIDATION_ROOT/omni-src"
export WAN_MODEL=/models/Wan2.2-T2V-A14B
for path in fa_mxfp8 fa_mxfp4 bsa_fp8 bsa_mxfp4; do
  python examples/offline_inference/text_to_video/text_to_video.py \
    --model "$WAN_MODEL" \
    --deploy-config "examples/offline_inference/text_to_video/wan22_quant_attention/${path}.yaml" \
    --num-inference-steps 50 --num-frames 81 --height 720 --width 1280 \
    --prompt "A cat walking through a sunlit garden" --seed 42 \
    --enable-cpu-offload --vae-use-tiling --enforce-eager \
    --output "$VALIDATION_ROOT/results/${path}.mp4" \
    2>&1 | tee "$VALIDATION_ROOT/results/${path}.log"
done
```

配置 `fallback: []`，不支持目标精度应明确失败。示例跳过 layer 0、39 和全局 step 0、49，其余符合条件的 self-attention 执行目标量化。BSA sparsity 为 0.8，命中 skip 后仍是浮点 BSA。cross 始终为浮点 Dense。

不要只检查 mp4 是否存在。用环境已有 profiler 或临时调用记录，确认非 skip 的 self-attention 实际到达目标算子，并保留证据：

| 路径 | 应观察到的调用 |
|---|---|
| Dense MXFP8 | TorchNPU `npu_fused_infer_attention_score_v2`，MX 量化输入 |
| Dense MXFP4 | MindIE `quant_flash_attn_metadata`、`quant_flash_attn` |
| BSA FP8 | MindIE `block_sparse_attention`，FP8 输入/scale |
| BSA MXFP4 | MindIE `block_sparse_attention`，MXFP4 quant_mode 与 dtype/scale codes |

Profiler 显示的 kernel 名可能不同，结合算子参数确认；尤其两种 BSA 共用公开算子名，不能只凭名字判定精度。记录 expert、global step、layer、role 与最终 precision，核对阶段切换前后 step 未归零。临时观测不计入性能数据。

## 6. 回退、基线与非整块输入

复制 YAML 到 results 目录后修改，不覆盖原始四个严格配置：

| 用例 | quant 配置变化 | 预期 |
|---|---|---|
| 无 skip | 两个 skip 字段均设为空字符串 | 所有可量化 self-attention 使用目标精度 |
| 只测 layer | skip_layers="0,39"，skip_steps="" | 两 expert 同一局部 block 选择器生效 |
| 只测 step | skip_layers=""，skip_steps="0,49" | 完整 50 步的首尾回退 |
| 同时命中 | 使用原始示例 | layer 或 step 任一命中即回退 |
| Dense 浮点基线 | self.quant.method=float | 浮点 Dense FA |
| BSA 浮点基线 | self.quant.method=float，保留 backend/sparsity | 浮点 BSA，稀疏度不变 |
| 显式精度链 | Dense MXFP4 fallback=[mxfp8,float]；BSA MXFP4 fallback=[fp8,float] | 仅预检不支持时选择下一候选 |

环境能力齐全时不会触发“不支持”的精度链；用第 4 节替身测试验证该分支，不通过故意破坏已安装插件来制造失败。执行中算子异常应直接暴露。

重复进程运行不能证明连续请求隔离。接口测试验证该行为；真实服务验证需在同一个 engine 内连续提交两次请求，检查第二次 global step 从 0 开始，cross 和各层策略无上次残留。

真实非整块测试需单独记录量化前原始序列长度、物理 padding 长度和裁剪后长度。使用模型支持的分辨率/帧数，使实际 attention 序列不整除对应对齐单位；不要仅凭图像宽高推断。在 Dense MXFP4 检查原始有效长度传到算子，BSA 保持 64 对齐，最终输出长度恢复。CPU 的 130、65、600 长度测试不能替代该项真实算子检查。

性能与质量比较保持模型、prompt、seed、采样器、steps、尺寸、offload 和并行配置一致。先预热，再重复测量；分别记录 Attention 耗时、端到端耗时与峰值显存，CPU offload 可能掩盖 Attention 收益。四条路径分别对比对应浮点基线；质量/VBench 门槛由评测方确认。eager 通过后，部署需要图模式时再去掉 `--enforce-eager` 单独验证。

## 7. 回传记录

| 项目 | 结果/证据（执行后填写） |
|---|---|
| 两仓完整 commit、实际 import 路径 | |
| 芯片、驱动、CANN、Torch/TorchNPU、vLLM/Ascend | |
| MindIE 构建与 BSA capability | |
| 接口测试 passed/failed/skipped 及原因 | |
| Dense MXFP8 / MXFP4 算子执行 | |
| BSA FP8 / MXFP4 算子执行 | |
| 两阶段、layer/step、cross、连续请求 | |
| 非整块输入、有效长度与输出裁剪 | |
| 浮点基线与四路径耗时/显存 | |
| 视频质量 / VBench | |
| 图模式（如需要） | |

出现失败时回传完整首个异常堆栈、对应 YAML、commit、capabilities.txt 和软件版本。将“接口通过”“NPU 执行通过”“质量/性能验收通过”分别记录。
