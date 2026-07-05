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


## 调度同批约束(2026-06,第二轮审查结论 — 关键)
深挖确认:higgs 走 `OmniScheduler` + 上游 sglang `get_next_batch_to_run` 驱动;仓库内
`PrefillManager`/`DecodeManager` 在此路径上是死代码。**同组同批 / 同组 retract 完全由上游驱动,
无 fusion 感知**——拆批、prefill→decode 过渡窗口、KV 压力下 retract 子集,均会静默破坏融合。
这是无法靠"写得仔细"消除、且不应 hack 上游调度器的硬约束。

**生产级对策(只动我们自己的 OmniScheduler + model 层,不碰上游):**
1. **原子入队 + 组计费**:N 个 sibling 作为一个单元一起入 `waiting_que`(已实现);sampler pool 与
   `max_running_requests` 须按"1 融合请求 = N 行"计费,避免组被池容量从中切断。
2. **运行时 fail-loud 断言(核心防御)**:在 `fuse_group_logits` 归约点,若本 step batch 未包含某组的
   全部成员(即组被上游拆散/retract),**明确抛错而非静默产出未融合音频**。宁可整请求失败、可观测、可重试,
   也不交付错误结果。这把"上游不保证同批"从隐性正确性 bug 转成显式契约违例。
3. PR 描述里把该约束列为已知限制,并建议上游加 fusion-group-aware 调度作为 follow-up。

## 第三轮对抗审查(2026-06)— 致命问题,需架构调整

全链路审查(顺数据流 7 文件)抓到的真实 bug,按严重度:

### BUG A(致命,必中)— follower 永不 finished → 自己的 fail-loud 把整批炸了
三个 collect 路径(`_decode_collect_host` / `_collect_step_outputs` / eager append loop)
里 follower-skip 的 `continue` 放在了 `_mark_sampler_finished` **之前**。`_mark_sampler_finished`
是 Higgs 路径上唯一设置 `Req.finished_reason` 的地方。后果链:
leader EOC → 被上游移出 batch;follower 无 finished_reason → 滞留 →
下一 decode step 组只剩 follower → `_populate_fusion_buffers` 的完整性检查
`present(1) != expected(2)` → RuntimeError → `_handle_batch_failure` abort **整批**
(含无关并发请求)。**每个正常完成的融合请求都会触发。** 是我自己的防御性断言被自己引爆。

### BUG B(架构级)— 同批不变量不可强制,fail-loud 把常规调度决策变成致命错误
完整性检查假设 N 个 sibling 每一步都同批。连续批处理引擎不保证:不等长参考 →
prefill 时长不同 → 短的先进 decode、长的还在 prefill → present<expected → 崩 + 连累整批。
"fail loud" 把一次普通调度决策升级成不可重试的致命错。共享 seed 锁步模型与引擎的
独立行调度根本对抗。

### BUG C(契约违反,静默)— CG 路径未 gate is_fused,污染所有非融合请求
`decode_codebooks_batch_cg` 无条件调 `fuse_group_logits`(eager 路径有 `is_fused` gate)。
单行组经 `log(softmax+1e-30)` 后分布尾部与 baseline 不再 byte 一致。违反"非融合请求行为
完全不变"的契约。多被 top_k/top_p 掩盖,但仍是契约违反 + 每步额外 softmax+scatter+log。

### BUG D — fusion registry 跨线程无锁
`_fusion_group_of/_fusion_weight_of/_fusion_group_size` 由 build-executor 线程写、
GPU-worker 线程每步读,无锁。registration/release 与 decode 交错时,完整性检查可能读到
部分状态 → 虚假 "group split" 崩 / 错误 weight。概率性。

### BUG F — _fusion_group_size 跨 retry 膨胀
size 只在全部成员清除时才减。retry 复用 request_id → 同 gid;若上次因 A/B 中途崩、
follower row 未释放 → 旧计数残留 → expected 永远偏大 → 该 gid 永久崩。与 A 复合。

### 审查确认 OK 的部分
- payload_types fusion_refs(含 torch.Tensor waveform)经 relay extract/restore_tensors 正确 round-trip
- 共享 seed 同帧抽样(**前提是同批**)
- batch-local group index 在 [0,B) 内,scatter_add_/index_select CG-safe
- abort 级联无无限递归/双重清理

### 根因
"fail-loud 断言 + 共享 seed 锁步"与连续批处理引擎的独立行调度对抗。要真正生产级,
co-batching 必须由调度器**强制保证**(把 sibling 钉在同一 batch + 同生同灭 retract),
而非在 decode 时**断言**。这正是第二轮审查就指出、但当时选择了"fail-loud 兜底"的硬约束——
事实证明 fail-loud 不是兜底,而是把问题从"静默错误"变成"必崩"。

## 方案1 实现:两层 co-batching 保证(2026-06,第三轮审查后定稿)
第三轮端到端审查(数据流 API→preprocess→builder→scheduler→model)抓到致命 bug A
(follower 跳过了 _mark_sampler_finished → leader 退出后 follower 滞留 → 自己的 fail-loud
断言把整批炸掉)及 B/C/D/F。结论:"fail-loud 断言 + 共享 seed 锁步" 单靠断言与连续批处理
引擎对抗,必须由调度器主动保证同批。已据此重构为两层:

**第一层 — 调度器组原子准入(正常路径,使断言永不触发):**
- `OmniScheduler._enqueue_built_request`:N siblings 一起入队,并在 `_fusion_group_members`
  记录完整组成员关系。
- `OmniScheduler.get_next_batch_to_run`(override 上游):上游选完 prefill batch 后,若某
  fusion 组只有部分成员在 batch 内,把这部分**整体退回 waiting_queue 队首**,使全组下一轮
  一起 prefill。decode batch 与非融合 batch 零开销透传。

**第二层 — decode 时 fail-loud 兜底(极端 KV 压力):**
- `fuse_group_logits` 前的组完整性检查(model.py `_batch_local_fusion` / runner
  `_populate_fusion_buffers`):若某组在本 step 缺员(被上游 KV-retract 拆散),抛
  RuntimeError 而非静默产出未融合音频。宁可可观测失败、可重试。

**bug 修复(第三轮):**
- A:follower 现在**先**经 `_mark_sampler_finished` 标记完成(与 leader 同步退出),**再**跳过音频
  append/emit。三个 collect 路径(CG decode / async resolve / prefill)全部修正。
- C:`fuse_group_logits` 对单成员组返回 `logits/T`(下游 softmax 后**字节级**等于 baseline),
  仅对真融合组返回 `log(blend)`,用按行 `torch.where` 选择 → CG 安全且非融合请求零影响。
  新增单测断言此字节级 identity。
- D:fusion registry 三个 dict 加 `_fusion_lock`;decode 步用 `fusion_membership_snapshot`
  取一致快照,杜绝读到半注册状态。
- F:`expected_fusion_group_size` 实时从成员派生(不再单独累加计数),retry 复用 rid 不再泄漏。

**已知待 Linux 实测验证项(本机 Windows 无法跑 sglang 引擎):**
1. **KV 回收(已按既有惯用法修复,仍需真实引擎验证)**:`get_next_batch_to_run` 摘除
   deferred siblings 前,现在会对每个 sibling 调用 `_release_request_kv_cache`(与
   `abort`/`_release_immediate_request_resources` 同一条路径,`req.req_pool_idx is None`
   时是 no-op)。这消除了"事后摘除但从不释放"的确定性泄漏,但**没有在真实引擎上跑过"反复
   触发部分组 defer"的压测**,不能确认 pool 计数在高频 defer 下确实归零、也不能确认在这个
   调用时机(`_Upstream.get_next_batch_to_run` 返回之后)`release_kv_cache` 需要的其他状态
   (tree_cache 簿记等)是否齐备。**这仍是阻塞上 mainline 前必须做的真实验证,不是"已解决"。**
2. **死锁前提**:组原子准入假设 KV 总量 ≥ 一个组的 prefill 占用。若 `max_running_requests`
   或 KV 池太小容不下整组,组会被无限退回 → 死锁。部署须按"1 融合请求 = N 行"给 KV/并发计费。
3. **prefill→decode 过渡**:全组同批 prefill 后是否同步进入 running_batch 首个 decode step
   (而非某行先 decode 一步)。依赖上游 prefill-vs-decode 优先策略,需实测。
4. CUDA graph 捕获下 `_cg_fusion_group`/`_cg_fusion_weight` 每步重填是否与图重放兼容。
