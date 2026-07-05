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
| 组落批可见性(非强制) | `omni_scheduler.py::OmniScheduler.get_next_batch_to_run`(override):只读扫描,某融合组只有部分成员落在同一 prefill batch 时打日志;不改动 batch(见下文"为什么不强制原子准入") |
| 组完整性兜底 | `model_runner.py::_populate_fusion_buffers`(decode CG 路径,真正会遇到"组被拆散"的地方,无论是 KV retract 还是 sibling 还没轮到) |
| 组级联清理 | `omni_scheduler.py::OmniScheduler._cascade_abort_split_fusion_group`(在 `stream_output` 里触发):一个融合成员以 `FINISH_ABORT` 结束时,把组里还注册着的其它成员一起 abort,防止缺席的 sibling 变成永久卡在 `waiting_queue` 里的僵尸请求 |
| 输出去重 | `model_runner.py::_finish_fusion_follower`:follower 的解码帧与 leader 重复,不 append/发音频,但仍要在同一步被标记 finished,否则组会"拆分" |

## 核心算法
`fuse_group_logits`(见 `fusion.py` 完整 docstring)对同组 N 行做加权 softmax 归约再转回
log 空间喂给标准 sampler;单行组(非融合请求)原样返回**未经任何处理的原始 logits**——
不预先除以温度。返回值额外带一个 `is_grouped_B` 掩码,调用方必须据此决定每行喂给 sampler
的温度——只有真正被分组的行才在归约时把温度折叠进去、随后以 `temperature=1` 采样;单行组
必须保持自己的真实温度并且只被除一次,否则会有两个后果:(a) 破坏 sampler 的 greedy 短路
(`temperature<=阈值` 时不经过 `multinomial`,直接 `argmax`);(b) 即便没有触发 greedy
短路,普通(非融合)请求也会被采样两次温度(`T²`)而不是一次——曾经真实出现过的 bug:
`fuse_group_logits` 内部为了在温度缩放后的空间里做融合,会把*所有*行(含单行组)都先除以
温度,如果连返回值也保留这个已经除过的版本,调用方再按 `is_grouped_B` 选择"单行组用真实温度
采样"时就会在已经除过一次的 logits 上再除一次——argmax 具有尺度不变性,所以 (a) 的回归测试
测不出这个问题,只有直接比较采样得到的*概率分布*才能测出来。现在的写法只在内部计算融合概率
时用温度缩放后的 logits,最终返回给单行组的仍是完全原始的 `logits_BNV`,调用方的那一次真实
温度除法就是唯一一次。这一契约由 `test_voice_fusion.py` 的两组回归测试共同守护:greedy 场景
验证 RNG 消耗量而非采样概率巧合(`test_singleton_greedy_sampling_matches_baseline_...`),
非 greedy 场景直接比较采样分布是否与不融合时的 baseline 一致
(`test_singleton_nongreedy_sampling_is_not_scaled_by_temperature_twice`)。

`fuse_group_generation_done` 做"组内任一成员 done ⇒ 全部 done"的屏障,让共享 seed 的
sibling 行在同一步终止,不会有的先跑完、有的还在解码的错位。

## Co-batching:decode 时隔离 + 级联 abort(没有强制的 prefill 原子准入)

"同批锁步"这件事,理想情况下希望调度器保证 N 个 sibling 总是一起进 prefill、一起进
decode。这里**曾经**试图在 `OmniScheduler.get_next_batch_to_run` 里强制这一点:上游选完
prefill batch 后,若某融合组只有部分成员在这批里,就把这些成员从 `batch.reqs` 里摘出
退回 `waiting_queue`。**这个机制已经被移除,因为它是错的**:上游 `get_new_batch_prefill`
返回 batch 之前,已经调用过 `ScheduleBatch.prepare_for_extend()`,把整批请求的
`input_ids`/`seq_lens`/`out_cache_loc` 等张量按*原始*(未摘除前的)`reqs` 顺序拍平好了。
事后再摘 `batch.reqs` 会让这些张量与摘除后的 `reqs` 列表长度对不上,搞坏的不只是被摘除的
sibling,是**这一整批**请求(含无关的普通请求)。上游自己的 `filter_batch` 是只用于 decode
batch 的工具,从不touch extend 张量——没有支持的方式能在 `prepare_for_extend` 之后收缩一个
已经组装好的 prefill batch。

现在的做法只剩一层,且是只读、绝不碰 `ScheduleBatch` 内部张量的:

**Prefill 侧(纯观测,不强制)**:`get_next_batch_to_run` 仍然扫描本批里的融合组成员,
若发现某组只有部分成员在场,只打一条 debug 日志,不做任何 batch 改动。sibling 们是否恰好
一起被 upstream 的 PrefillAdder 选中,完全取决于它们在 `waiting_queue` 里是否相邻、以及
当前 tick 的 token/KV 预算——多数情况下会一起进(`_enqueue_built_request` 把它们相邻插入
队列),但没有硬保证。

**Decode 侧(真正的正确性防线)**:
1. `HiggsTTSModelRunner._populate_fusion_buffers`(decode CG 路径,`OmniScheduler.
   _retract_running_requests` 只作用于 running batch,所以只有这条路径会真的遇到"组被
   retract 拆散",或"sibling 还没轮到进 decode"两种情况)——若某组在本 step 缺员,只隔离该
   组的在场行:降级为独立单例(不参与融合,避免用不完整分布产出错误音频),把它们的
   `req.finished_reason` 设为 `FINISH_ABORT()`。同批其它未受损的行(含其它融合组、普通请求)
   不受影响。
2. `OmniScheduler.stream_output` 里的 `_cascade_abort_split_fusion_group`:上面第 1 步
   abort 的是"在场"的行;组里"缺席"的那个 sibling(比如被 KV 压力 retract、正躺在
   `waiting_queue` 里等下一轮)如果没人管,会在未来某个 tick 自己重新 prefill、自己进
   decode——这时组里只剩它一个,`expected_fusion_group_size` 要么已经因为同伴清场而缩到 1
   (于是它会被当成误打误撞的单例、悄悄产出未融合的音频),要么组注册表还没清干净、它又会
   被判定"缺员"再 abort 一轮——都不是我们想要的。修法:`stream_output` 处理一个 fusion 成员
   的 finish 时,如果它的 `finished_reason` 恰好是 `FINISH_ABORT`(区别于正常同步完成用的
   `FINISH_MATCHED_TOKEN`——组正常结束时全体成员总是在同一 decode step 一起触发屏障、一起
   出现在同一次 `stream_output` 调用里,不会有"其它成员还没完成"的情况),就把组里其它仍
   注册着的成员一起 `abort()` 掉。`abort()` 本身既能从 `waiting_queue` 摘除、也能处理正在
   跑的行,是这个仓库里唯一验证过、安全处理"请求可能在队列也可能在运行"两种状态的通用清理路径。

prefill 路径(`model.py::_batch_local_fusion`)保留整体 `RuntimeError`:这条路径只有
`HiggsGenParams`、没有 `Req` 句柄可单独标记 abort。

**这样做的代价(诚实说明,不是"已解决")**:没有了 prefill 侧的强制原子准入,在调度压力大、
sibling 的 prefill 没能挤进同一 tick 的情况下,融合请求可能会比"总是原子准入"的设计更频繁地
被 decode 侧检测到"缺员"从而直接 abort,而不是透明地多等一两个 tick 再重试。客户端需要能
处理"融合请求返回一个 abort/error 而不是音频"的情况(并可自行重试),这是一个真实的可靠性
权衡,不是缺陷——但也不应该被包装成"两层原子保证都齐了"。

## 为什么 ramp(delay/EOC 状态机)不会因参考长度不同而错位——以及一个尚未补上的口子
一个容易误判的点:N 个 sibling 的参考音色时长可能不同,直觉上会担心各自的
delay/EOC 状态机(见 `sampler.py::HiggsBatchedSamplerState`)因此错位。多数情况下不会,
原因是两点组合,而非某种"对齐参考长度"的机制:

1. `stages.py` 里 `_MAX_REF_AUDIO_SEC = 100` 把单条参考**音频**硬顶在 7500 帧
   (100s×75Hz),留了大约 692 个 token 的余量给文本+特殊 token,不超过 `chunked_prefill_size`
   (8192,见 `engine_builder.py`)——"chunked prefill of the multi-codebook prompt is unsafe
   (sampler state machine has no rollback)"。因此*通常*每个 sibling 的 prefill 都能一步内
   完成,不会被拆成多个 scheduler tick。
2. `HiggsBatchedSamplerState.reset_row`(`sampler.py:91`)只在 sibling 首次拿到
   sampler-pool 行时(即 prefill 准入那一刻)把 `delay_count`/`step_count` 清零,且之后只由
   `batched_step` 的调用次数(解码步数)驱动——与该 sibling 自己参考音频多长完全无关。

**尚未补上的口子**:上面第 1 点只顶住了*音频*本身的长度,没有顶住"音频 + 目标文本 + 特殊
token"的**总**长度。`_MAX_REF_AUDIO_SEC` 留的 692-token 余量对绝大多数正常长度的目标文本
够用,但没有任何代码显式校验"这条 sibling 的完整 prompt 长度 ≤ chunked_prefill_size"——
一段异常长的目标文本 + 一条接近 100s 上限的参考音频组合起来,理论上仍可能压过
`chunked_prefill_size`,触发多 tick 的 chunked prefill,这时"prefill 一步完成"的前提就不
成立了。修这个口子需要在 `build_fusion_sibling_requests`(`request_builders.py`)校验总
prompt 长度,但那里目前拿不到活的 `chunked_prefill_size`(一个 server_args 值,当前这条构
建路径完全不感知调度器配置)——贸然写一个脱离实际配置的硬编码阈值,又会制造新的一份"文档/
常量说一套、真实配置是另一套"的漂移,所以暂时按已知限制记录,而不是塞一个假安全的检查。

（历史上这里曾设想过"prefill 时按组内最长参考做 BOC 左填充对齐"的方案,但从未实现,
也不需要——上面两点已经是完整的正确性论证。）

## 已知限制
- 没有 prefill 侧的强制原子准入(见上文),融合请求在高并发/高 KV 压力下可能比预期更容易被
  decode 侧检测为"缺员"而直接 abort;客户端需要能处理这种情况并重试。
- 单个 sibling 的 prompt 总长度(参考音频 + 目标文本 + 特殊 token)没有显式校验是否超出
  `chunked_prefill_size`,只靠 `_MAX_REF_AUDIO_SEC` 留的固定余量隐式兜底(见上一节)。
- sampler pool 容量 = `max_running_requests + 1`;一个融合请求占 N 行,部署需按
  "1 融合请求 = N 行" 给 KV/并发计费,否则 KV 压力下更容易触发 decode 侧的缺员 abort。
- 仅 logit 融合(非 prompt 拼接);CFG 外推留作 follow-up。
- 融合请求完全绕开现有的参考缓存栈(speaker artifact cache / `reference_code_cache_key` /
  已上传语音),每次都会重新过一遍 codec 编码,已上传的具名语音也无法参与融合。作为 v1 范围
  可以接受,但这是一个尚未文档化过的能力缺口。

## 仍需真实引擎验证的项(本机 Windows 无 sglang,无法跑真实引擎)
1. **decode 侧隔离 + 级联 abort 的端到端行为**:`_populate_fusion_buffers` 的隔离逻辑和
   `_cascade_abort_split_fusion_group` 的级联逻辑都各自过了单测(`test_voice_fusion_pipeline.py`
   里 mock 了 `HiggsTTSModelRunner`/`FusionRegistry` 直接调用真实方法),但两者串起来的真实
   端到端行为(KV 压力下真的触发 retract → 隔离 → 级联 → 客户端收到什么)只验证到"逻辑上
   应该正确",没有在真实引擎上跑过。
2. **prefill→decode 过渡的实际发生率**:没有了强制原子准入,sibling 们在真实负载下实际上有
   多大比例仍然一起进 decode(而不是触发上面的 abort 路径)。这直接决定这个功能在生产环境下
   的可用性,需要用不同长度的参考音频实测。
3. **CUDA graph 兼容性**:`_cg_fusion_group`/`_cg_fusion_weight` 每步重填是否与图重放兼容。
4. **非融合热路径零开销**:`FusionRegistry.has_any()` 让零融合流量的服务器跳过每步的
   follower 检查/buffer 填充,已有针对计数器本身的纯 Python 单测
   (`test_voice_fusion.py` 的 `test_registry_*`),但吞吐层面的"确实零开销"还没有真实
   engine 下的压测数据支撑;此外 `fuse_group_logits` 在 CG 解码路径上是无条件调用的(不像
   eager 路径有 `is_fused` 门槛),对全零融合流量的服务器而言这是一次额外的、可忽略但确实
   存在的 fp32 softmax 开销,不是真正的零成本。
5. **`FINISH_ABORT` 直接赋值 `finished_reason` 而非走 `to_finish` 的时机是否安全**:上游
   `Req.to_finish` 字段的注释明确写着"如果想在事件循环中途 abort 一个请求,应该设置
   `to_finish` 而不是直接设置 `finished_reason`,否则请求会被过滤掉、再也不会响应"。Higgs
   TTS 的整条完成信号链路(包括早已验证工作的正常完成路径 `_mark_sampler_finished`)在这个
   相同的代码层级都是直接设置 `finished_reason`,从未使用 `to_finish`——`_populate_fusion_
   buffers`(BLOCKING-3)、`_cascade_abort_split_fusion_group` 都是照抄这个已经在正常完成路径
   上验证过的既有约定,而不是新造一种写法。逻辑上应该是安全的(时机上早于上游任何会检查
   `finished_reason` 的处理),但这个"为什么 Higgs 的写法能绕开上游注释里的警告"从未在真实
   引擎上专门针对 abort 路径验证过。
