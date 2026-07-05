# Higgs TTS Voice Timbre Fusion — 设计说明

## 目标
给 sglang-omni 的 Higgs TTS 加"音色融合":一次合成可同时条件化 N 个参考音色,按权重在
**解码输出分布层**加权融合(不是 prompt 拼接),得到一个稳定的"中间音色"。

## 机制(横向扩展,非侵入调度内核)
一个融合请求 = N 个 sibling batch 行,每行独立 prefill 出一个参考音色的 KV 上下文。
解码每一步:`modality_head.generate() -> logits_BNV [B,8,1026]` 之后、`batched_step` 之前,
对同组 N 行做**加权概率归约**(同组拿到同一融合分布、同 seed 抽同一帧),N 条上下文锁步演化,
仅 leader 行输出音频。组内共享 `generation_done` 做"同生同灭"屏障。

## 核心组件
| 机制 | 落点 |
|---|---|
| 归约算法 | `fusion.py::fuse_group_logits` / `fuse_group_generation_done`(纯 torch,无 sglang 依赖,可独立单测) |
| 归约钩子 | `model.py` `decode_codebooks_batch` / `decode_codebooks_batch_cg`,在 `batched_step` 前替换 logits |
| 融合注册表 | `fusion.py::FusionRegistry`(线程安全,build 线程写、GPU worker 线程读);`model.py` 的 `set_fusion_group`/`has_any_fusion`/`is_fusion_follower` 等是对它的薄委托 |
| 请求拆分 | `request_builders.py::build_fusion_sibling_requests`:1 融合 payload → N 条 `HiggsSGLangRequestData`,共享 `fusion_group_id` + 一个具体 seed,leader=第一个 sibling |
| 组原子准入 | `omni_scheduler.py::OmniScheduler.get_next_batch_to_run`(override):prefill batch 若只含某融合组部分成员,把这些成员整体退回 waiting_queue,直到全组能一起入批 |
| 组完整性兜底 | `model_runner.py::_populate_fusion_buffers`(decode CG 路径,真正会遇到 KV retract 拆散组的地方) |
| 输出去重 | `model_runner.py::_finish_fusion_follower`:follower 的解码帧与 leader 重复,不 append/发音频,但仍要在同一步被标记 finished,否则组会"拆分" |

## 核心算法
`fuse_group_logits`(见 `fusion.py` 完整 docstring)对同组 N 行做加权 softmax 归约再转回
log 空间喂给标准 sampler;单行组(非融合请求)原样返回 `logits/temperature`,与 baseline
字节级一致。返回值额外带一个 `is_grouped_B` 掩码,调用方必须据此决定每行喂给 sampler 的
温度——只有真正被分组的行才在归约时把温度折叠进去、随后以 `temperature=1` 采样;单行组必须
保持自己的真实温度,否则会破坏 sampler 的 greedy 短路(`temperature<=阈值` 时不经过
`multinomial`,直接 `argmax`)。这一契约由 `test_voice_fusion.py` 的确定性回归测试守护
(验证 RNG 消耗量而非采样概率巧合)。

`fuse_group_generation_done` 做"组内任一成员 done ⇒ 全部 done"的屏障,让共享 seed 的
sibling 行在同一步终止,不会有的先跑完、有的还在解码的错位。

## 两层 co-batching 保证
单靠 decode 时的断言无法让"同批锁步"这件事在连续批处理引擎下始终成立,因此拆成两层:

**第一层 — 调度器组原子准入(正常路径,让第二层的兜底几乎永不触发):**
`OmniScheduler._enqueue_built_request` 把 N 个 sibling 一起入队,并在
`_fusion_group_members` 记录组成员关系;`get_next_batch_to_run` override 在上游选完
prefill batch 后检查:若某融合组只有部分成员在这批里,就把这部分整体退回
`waiting_queue` 队首,让全组下一轮一起 prefill(退回前会显式释放已分配的 KV 资源,
避免反复被 defer 时泄漏)。decode batch 与非融合 batch 零开销透传。

**第二层 — decode 时组完整性检查(极端 KV 压力下的兜底,按组隔离而非炸全批):**
`OmniScheduler._retract_running_requests` 只作用于 running batch / decode 阶段,所以只有
decode CG 路径(`model_runner.py::_populate_fusion_buffers`)会真的遇到"组被 retract 拆散"。
若某组在本 step 缺员,只隔离该组的在场行——降级为独立单例(不参与融合,避免用不完整分布
产出错误音频),把它们的 `req.finished_reason` 设为 `FINISH_ABORT()`。同批其它未受损的行
(含其它融合组、普通请求)不受影响。

prefill 路径(`model.py::_batch_local_fusion`)保留整体 `RuntimeError`:这条路径只有
`HiggsGenParams`、没有 `Req` 句柄可单独标记 abort;而且 retract 不作用于 prefill 阶段,
理论上第一层生效后这里应该不可达,是一个"不该发生"的断言而非常态防御路径。

## 为什么 ramp(delay/EOC 状态机)不会因参考长度不同而错位
一个容易误判的点:N 个 sibling 的参考音色时长可能不同,直觉上会担心各自的
delay/EOC 状态机(见 `sampler.py::HiggsBatchedSamplerState`)因此错位。实际不会,原因是三点
组合,而非某种"对齐参考长度"的机制:

1. `stages.py` 里 `_MAX_REF_AUDIO_SEC = 100` 把单条参考音频硬顶在 `chunked_prefill_size`
   (8192,见 `engine_builder.py`)以内——"chunked prefill of the multi-codebook prompt is
   unsafe (sampler state machine has no rollback)"。因此任何一个 sibling 的 prefill 都保证
   一步内完成,不会被拆成多个 scheduler tick。
2. `HiggsBatchedSamplerState.reset_row`(`sampler.py:91`)只在 sibling 首次拿到
   sampler-pool 行时(即 prefill 准入那一刻)把 `delay_count`/`step_count` 清零,且之后只由
   `batched_step` 的调用次数(解码步数)驱动——与该 sibling 自己参考音频多长完全无关。
3. 第一层组原子准入保证 N 个 sibling 在同一个 scheduler tick 一起完成 prefill 准入,因而
   在同一个 tick 一起 claim 各自的 sampler-pool 行、一起把 ramp 状态清零、一起产出第一个
   解码帧。之后靠共享 seed + `fuse_group_generation_done` 屏障维持锁步直到组内全部完成。

（历史上这里曾设想过"prefill 时按组内最长参考做 BOC 左填充对齐"的方案,但从未实现,
也不需要——上面三点已经是完整的正确性论证,径直依赖已有的两层 co-batching 保证。）

## 已知限制
- N sibling 行需在同一 decode batch;组原子准入假设 KV 总量 ≥ 一个组的 prefill 占用——若
  `max_running_requests` 或 KV 池太小容不下整组,组会被无限退回、死锁。部署须按
  "1 融合请求 = N 行"给 KV/并发计费。
- sampler pool 容量 = `max_running_requests + 1`;同上,需按 N 计费或抬高上限。
- 仅 logit 融合(非 prompt 拼接);CFG 外推留作 follow-up。

## 仍需真实引擎验证的项(本机 Windows 无 sglang,无法跑真实引擎)
1. **KV 回收**:`get_next_batch_to_run` 摘除 deferred sibling 前会调用
   `_release_request_kv_cache`(与 `abort` 同一条路径)。逻辑上应该正确,但没有在真实引擎上
   跑过"反复触发部分组 defer"的压测,不能确认 pool 计数在高频 defer 下确实归零。
2. **prefill→decode 过渡**:全组同批 prefill 后是否同步进入 running_batch 的首个 decode
   step(而非某行先 decode 一步)。上面"为什么 ramp 不会错位"一节给出的是静态代码论证,
   还需要在真实调度器上用不同长度的参考音频实测确认。
3. **CUDA graph 兼容性**:`_cg_fusion_group`/`_cg_fusion_weight` 每步重填是否与图重放兼容。
4. **非融合热路径零开销**:`FusionRegistry.has_any()` 让零融合流量的服务器跳过每步的
   follower 检查/buffer 填充,已有针对计数器本身的纯 Python 单测
   (`test_voice_fusion.py` 的 `test_registry_*`),但吞吐层面的"确实零开销"还没有真实
   engine 下的压测数据支撑。
