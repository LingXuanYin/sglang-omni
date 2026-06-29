# Higgs TTS Voice Timbre Fusion — sglang-omni PR 设计与实现计划

## 目标
给 sglang-omni 的 Higgs TTS 加"音色融合":一次合成可同时条件化 N 个参考音色,按权重在
**解码输出分布层**加权融合(不是 prompt 拼接),得到一个稳定的"中间音色"。

## 机制(横向扩展,非逆架构)
一个融合请求 = N 个 sibling batch 行,每行独立 prefill 出一个参考音色的 KV 上下文。
解码每一步:`modality_head.generate() -> logits_BNV [B,8,1026]` 之后、`batched_step` 之前,
对同组 N 行做**加权概率归约**(同组拿到同一融合分布、同 seed 抽同一帧),N 条上下文锁步演化,
仅 leader 行输出音频。组内共享 `generation_done` 做"同生同灭"屏障(最小侵入,不碰调度内核)。

## 与现有架构的契合点(已逐行核对真实源码)
| 机制 | 现成切点 | 文件:行 |
|---|---|---|
| 归约钩子 | `batched_step_direct` 前拿到 `logits_BNV` | model.py decode_codebooks_batch_cg |
| 组屏障 | `_cg_active_generation_done` scatter 归约 | model_runner.py:185 / model.py |
| 输出去重 | `_decode_collect_host` 非 leader `continue` | model_runner.py:289-312 |
| 参考注入 | 逐请求 `-100` 占位替换,N 行各注入各的 | model_runner.py:338-359 |
| rid 派生 | `_rid_to_row` 纯字符串键,无格式约束 | model.py:167 |

## 必须新增(无现成机制)
1. **拆分 (A)**:1 融合 payload → N 条 `HiggsSGLangRequestData`,rid 派生 `rid#0..N-1`,
   共享 `fusion_group_id`,leader=`#0`。需扩展 builder 返回 list + 入队链。
2. **CG 兼容的组归约 (C+B)**:用预分配 buffer `_cg_fusion_group_id_B` / `_cg_fusion_weight_B`,
   全张量算子(scatter_add + index + log),无 host 分支。默认单行组退化为恒等。

## 文件改动清单
1. `payload_types.py` — `HiggsTtsState` 加 `fusion_refs: list[dict]|None`(每项 codes_delayed/weight/reference_text)
2. `stages.py` — preprocessing/audio_encoder 处理带 weight 的 `references[]`,逐个编码
3. `request_builders.py` — `HiggsSGLangRequestData` 加 fusion 字段;新增 `build_fusion_requests` 返回 list;builder 分发
4. `omni_scheduler.py` — `_enqueue_built_request` 接受 list(N 条同时入队)
5. `model.py` — `HiggsGenParams` 加 fusion 字段;`__init__` 加 `_cg_fusion_*` buffer + `_fusion_*` map;
   `decode_codebooks_batch` / `decode_codebooks_batch_cg` 插入归约;`set_fusion_group` API
6. `model_runner.py` — `_populate_cg_buffers` 填 fusion buffer;`_decode_collect_host` 跳过非 leader
7. **API 层** — `references[].weight` 透传(扩展现有 references 数组,向后兼容)
8. **tests** — 单测:归约数值正确性、单行组退化为恒等、组屏障同步、输出去重

## CG 兼容的组归约(核心算法)
```python
# group_id_B [B]: 同组共享 id;非融合行 id=自身索引(独占组)
# weight_B   [B]: 组内权重;非融合行 = 1.0
probs_BNV = (logits_BNV / temp).softmax(-1)                    # [B,N,V]
w = weight_B[:, None, None]                                    # [B,1,1]
idx = group_id_B[:, None, None].expand(B, N, V)               # [B,N,V]
fused = torch.zeros_like(probs_BNV).scatter_add_(0, idx, probs_BNV * w)
fused_logits = (fused[group_id_B] + 1e-30).log()             # 广播回各行 → batched_step
# 单行组: fused[b]=probs[b]*1.0 → log(softmax) → 采样等价于原 logits/temp(argmax/multinomial 不变)
```
组屏障同理:`done_any = zeros.scatter_add_(0, group_id_B, done.float())>0; new_done = done_any[group_id_B]`

## 已知限制(写入 PR 描述)
- N sibling 行需在同一 decode batch。它们同时入 `waiting_que`、prompt 等长,PrefillAdder 高概率同批准入;
  极端情况(池满分批)下退化为各自独立——加准入校验(组成员要么全准入要么全等待)。
- sampler pool 容量 = max_running_requests+1;1 融合请求占 N 行,需按 N 计费或抬高上限。
- 仅 logit 融合(非 prompt 拼接);CFG 外推留作 follow-up。

## 对抗审查结论(2026-06,已据此修正)
核心数值算法(`fusion.py`)经本机纯 torch 单测 11/11 通过;温度只应用一次、was_done 顺序、
单行组退化为恒等均正确。审查抓到的待修项:
1. **CG buffer 填充(B)**:`_cg_fusion_group`/`_cg_fusion_weight` 必须由 `model_runner._populate_cg_buffers`
   每步用 batch-local group id + weight 重填,padding 行回退为 own-slot/weight-1。否则 CG 生产路径静默不融合。
2. **CG follower 去重(D)**:`_decode_collect_host` 必须跳过 fusion follower 行(否则每个 follower 重复发音频)。
3. **bug #4 — ramp skew(已定方案:源头对齐)**:屏障只同步 generation_done 不够;根因是同组参考长度不等
   → delay ramp 错位 → 共享 seed 不再同帧。**修法:prefill 注入时把同组 N 个参考用 BOC 左填充对齐到组内最长**,
   使 delay_count/eoc_countdown 天然同步,屏障只需同步 generation_done。对齐在 prefill 构造处做。
4. **端到端激活**:builder fan-out + `_gen_params_for_batch` 透传 fusion 字段,见下方文件清单(A)。

