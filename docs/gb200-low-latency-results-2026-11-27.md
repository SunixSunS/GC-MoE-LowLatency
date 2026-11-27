# GC-MoE GB200 小 batch 低延迟结果

截至 2026-11-27，真 EP16 在 16 和 32 token/rank 的最低实测 p50 分别为 **66.560 µs** 和 **68.704 µs**。真 EP32 对应为 **49.600 µs** 和 **52.864 µs**。常规配对中，真 EP32 的 32-token 档最大降幅为 **20.71%**。

## 测试口径

真 EP16 和 EP32 分别使用 16 和 32 张 GB200，跨节点走 NVLink fabric。用 GLM 5.2/5.3 目标形状的合成张量和均匀路由，H6144、I2048、256 个 routed 专家、Top-8、1 个 shared 专家。激活为 FP8，routed 权重为 FP4 加 K32 scale，shared 权重为 FP8 加 K128 scale。CUDA Graph 在目标层前执行三组独立地址的权重，逐窗取最慢 rank，再算 p50/p95。每项都和同轮高吞吐原版 `stock` 配对，所有 rank 的 BF16 输出逐位一致。

## 最低绝对 p50

| 真卡档位 | 原版 → 候选 p50（µs） | 原版 → 候选 p95（µs） | 同轮降幅 | 窗口 | 配置与原始记录 |
|---|---:|---:|---:|---:|---|
| EP16，16 token/rank | 79.952 → **66.560** | 83.968 → 70.400 | 16.75% | 64 | D32、16 Dispatch warp、回填 24 CTA、窄 BM32、分散 cluster、W1 bulk、CTA 共享路由计数。P329 |
| EP16，32 token/rank | 79.920 → **68.704** | 84.096 → 72.352 | 14.03% | 160 | D24、16 Dispatch warp、回填 24 CTA、窄 BM32、W1K4/W2K2、分散 cluster。P252 |
| EP32，16 token/rank | 59.584 → **49.600** | 63.488 → 51.648 | 16.76% | 110 | D24、零回填、普通权重、提前 sentinel Combine。T16 final-trace 轮的未插桩计时窗 |
| EP32，32 token/rank | 65.632 → **52.864** | 69.728 → 55.968 | 19.45% | 60 | D24、回填 16 CTA、BM48、8 Combine warp、单 CTA 清理、W1 bulk、CTA 共享路由计数。P335 |

D24/D32 表示 Dispatch 分配的 SM 数。EP32 T16 的 **49.632 µs** 在三轮复现，同轮原版 **61.472 µs**，降幅 **19.26%**。EP32 T32 的三轮低值为 **52.896、52.928、52.864 µs**，观察到的差值在 0.064 µs 内。

## 最大常规配对降幅

下表取各档独立的正常计时对照。用于 rank 对齐或 Nsight 采集的诊断轮、以及高波动专项对照单独解读。

| 真卡档位 | 同轮原版 → 候选 p50（µs） | 同轮降幅 | 窗口 | 记录 |
|---|---:|---:|---:|---|
| EP16，16 token/rank | 80.448 → 66.656 | **17.14%** | 20 | DeepSeek 晚间 B，窄 BM32、精简描述符和分散 cluster |
| EP16，32 token/rank | 81.920 → 69.344 | **15.35%** | 145 | P257，分散 cluster 配 D32、回填 24 CTA |
| EP32，16 token/rank | 63.472 → 51.200 | **19.34%** | 128 | P107，回填 8 CTA、VMM all 和 16 KiB W1 bulk |
| EP32，32 token/rank | 67.616 → 53.616 | **20.71%** | 100 | P2，回填 16 CTA、VMM all、W1K4/W2K2、提前 sentinel Combine |

真 EP32 T32 在 P327 使用普通权重和 BM48，加入 Dispatch 侧 64 KiB W1 bulk、静态 W1 和启动同步调整后，同轮候选从 **54.176 降至 52.928 µs**，p95 从 **56.928 降至 55.808 µs**。相对同轮高吞吐原版 **66.656 µs**，p50 下降 **20.60%**。这个组合复现了 20% 级别的改善。 P304 的 rank 对齐冷读专项测到 **20.92%**，对应 p50 **55.296 µs**。

## 实测有效的改动

- **让回传和归约进入 W2 的流水。** W2 逐块发布 sentinel，Combine 逐块消费；清理工作交给单个 CTA。真 EP32 T32 匹配 BM48 对照中，单 CTA 清理使 p50 **54.784 → 53.952 µs**。真 EP16 T32 另一组 Nsight 中，低延迟路径的计算后尾部由 **9.31 降至 3.01 µs**。记录 P295、P239。
- **按每专家的实际行数选分块。** EP16 小 batch 的窄 BM32 搭配 W2 tile-major 激活、描述符精简及分散 cluster。EP32 T32 的均匀路由样本中，最多一个专家有 46 行，BM48 可用一个 M tile 覆盖该专家，减少 BM32 可能发出的重复权重请求。记录 P242、P252。
- **把启动预取和路由计数分配到合适的 CTA。** Dispatch 侧 W1 bulk 与静态 W1 的组合在真 EP32 T32 省 **1.248 µs**。真 EP16 D32 上，让每个 Dispatch CTA 只用一个 warp 收齐路由计数，再从 SMEM 广播给同 CTA 的其他 warp，T16/T32 的 p50 各省 **0.720 / 1.008 µs**。真 EP32 T32 叠加到已有 W1 bulk 方案仅省 **0.064 µs**。记录 P327、P329、P335。

以上数据分别来自各自的匹配控制，组合收益随 CTA 布局和 token 数变化。

## 关键路径与边界

真 EP16 T32 中，首个采样 Shared W1 MMA 操作数在 Dispatch 启动后 **4.37 µs** 就绪。首个 Routed W1 权重 TMA 约 **9.42 µs** 发射，激活 TMA 约 **15.50 µs** 发射，首个 Routed W1 MMA 操作数约 **15.87 µs** 才齐。首批 Routed 工作受整块激活交付限制。真 EP32 T32 对应首个 Routed W1 MMA 为 **17.42 µs**，相对同 rank 最后的已采样 Dispatch 拉取完成标记提前 **3.71 µs**。记录 P325、P335。

EP16/EP32 的逻辑权重流量为 **358.91 / 198.48 MB 每 rank**，独立本地 TMA 冷读为 **54.62 / 32.11 µs**。按权重连续搬运并覆盖绝大部分 Dispatch/Combine 的假设，T32 单层估算约 **61–62 / 40–41 µs**。真实吞吐还受激活就绪、W1→W2 依赖、权重请求分布和慢 rank 影响。当前证据为微基准和设备时间戳，实际 HBM 流量与总线空闲比例仍待计数器验证。

曾尝试将 Dispatch 直接接到 W1 mailbox、拆 W2 Split-K、把 Combine 融进计算 CTA 退出、从 Dispatch 尾部预取少量 W2 权重。较强的同轮基线下，这些方案均未给出稳定净收益。最新的 W1 等量聚焦预取依赖 Dispatch 配比，保留为四卡诊断结果。记录 P334、P345、P348。

## 复现索引

实验源码入口是 `benchmarks/benchmark_decode_critical_path.py`。源码实验分支 `experiment/decode-structure-resume` 的本地提交 `40ac39e` 加入 CTA 共享计数，`ca43ace` 拆分 W1/W2 前缀。晚间 BF16 输出校验和候选配置保留在逐 rank 样本中。公共参数为 `--num-experts 256 --tokens 16 32 --capacity 256 --intervening-layers 3`，原版计时使用 `GC_MOE_PRIMARY_CTX=0 GC_MOE_DISPATCH_SMS=16`。

| 记录 | 原始样本文件 |
|---|---|
| P252 / P257 | `ep16-flat-cluster-spread-placement-252.json` / `ep16-narrow-dispatch-residency-257.json` |
| P329 / P325 | `true-ep16-cta-shared-route-counts-329.json` / `true-ep16-flat-bulk-startup-325.json` |
| EP32 T16 最低 / 复现 | `ep32-sentinel-final-trace-t16.json` / `ep32-sentinel-green-confirm.json` |
| P335 / P327 | `true-ep32-bm48-shared-counts-335.json` / `true-ep32-bm48-dispatch-327.json` |
| P107 / P2 | `ep32-overlap-t16-pruned-w1-stack-rerun-107.json` / `ep32-overlap-t32-2.json` |
| DeepSeek B | `true-ep16b.json` |
