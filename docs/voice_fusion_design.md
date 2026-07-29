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
| 融合注册表 | `fusion.py::FusionRegistry`(线程安全,build 线程写、GPU worker 线程读);`model.py` 的 `set_fusion_group`/`has_any_fusion`/`is_fusion_follower`/`mark_fusion_group_poisoned` 等是对它的薄委托 |
| 请求拆分 | `request_builders.py::build_fusion_sibling_requests`:1 融合 payload → N 条 `HiggsSGLangRequestData`,共享 `fusion_group_id` + 一个具体 seed,leader=第一个 sibling |
| prefill 侧原子准入(尽力,非硬保证) | `omni_scheduler.py::OmniScheduler.get_next_batch_to_run`(override):在调用上游选批之前,先对 `self.waiting_queue`(一个纯 Python list,尚未流入 `ScheduleBatch.prepare_for_extend`)做门控——缺员的组整组暂扣,凑齐且预算内的组整组挪到队首、成员相邻,排在普通请求前面;调用完再把暂扣的请求放回队首。只动 list,不碰张量(见下文"Co-batching") |
| prefill 侧隔离 + 中毒标记 | `model.py::_batch_local_fusion`:prefill 完成后触发首个解码 token 那一步若发现组缺员,隔离在场行(不参与融合)并把该组标记为 `FusionRegistry._poisoned`——因为这里没有 `Req` 句柄,没法自己 abort,而组之后可能"自愈"(所有成员后来都凑齐了),中毒标记是让下面的完整性检查即便凑齐也照样 abort 的唯一办法 |
| 组完整性兜底 | `model_runner.py::_populate_fusion_buffers`(decode CG 路径,真正会遇到"组被拆散"的地方,无论是 KV retract、sibling 还没轮到,还是曾经中毒但现在看起来凑齐的组) |
| 组级联清理 | `omni_scheduler.py::OmniScheduler._cascade_abort_split_fusion_group`(在 `stream_output` 里触发):一个融合成员以 `FINISH_ABORT` 结束时,把组里还注册着的其它成员一起 abort(并向每个被级联的成员发一条 client 可见的 error,leader 也不例外),防止缺席的 sibling 变成永久卡在 `waiting_queue` 里的僵尸请求 |
| 输出去重 | `model_runner.py::_finish_fusion_follower`:follower 的解码帧与 leader 重复,不 append/发音频,但仍要在同一步被标记 finished,否则组会"拆分" |
| 异步 lookahead 规避 | `model_runner.py::HiggsTTSModelRunner.lookahead_eligible`:只要有任何融合请求注册,强制走同步解码路径,避免这个仓库自建的 one-step-lookahead 把 launch 阶段设置的 FINISH_ABORT 误判成"上一步的过期行"而丢弃 |

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

## Co-batching:prefill 侧原子准入门控 + decode 侧隔离/级联 abort 兜底

"同批锁步"这件事,理想情况下希望调度器保证 N 个 sibling 总是一起进 prefill、一起进
decode。这里**曾经**试图在 `OmniScheduler.get_next_batch_to_run` 里、在上游已经选完
batch **之后**强制这一点:若某融合组只有部分成员在这批里,就把这些成员从 `batch.reqs`
里摘出退回 `waiting_queue`。**这个机制已经被移除,因为它是错的**:上游
`get_new_batch_prefill` 返回 batch 之前,已经调用过 `ScheduleBatch.prepare_for_extend()`,
把整批请求的 `input_ids`/`seq_lens`/`out_cache_loc` 等张量按*原始*(未摘除前的)`reqs`
顺序拍平好了。事后再摘 `batch.reqs` 会让这些张量与摘除后的 `reqs` 列表长度对不上,搞坏的
不只是被摘除的 sibling,是**这一整批**请求(含无关的普通请求)。上游自己的 `filter_batch`
是只用于 decode batch 的工具,从不 touch extend 张量——没有支持的方式能在
`prepare_for_extend` 之后收缩一个已经组装好的 prefill batch。

现在的原子准入改为在上游选批**之前**做门控,而不是事后修剪结果——`self.waiting_queue`
在这一步还只是一个纯 Python list,尚未流入 `ScheduleBatch`/`prepare_for_extend`,对它做
过滤和重排不涉及任何张量,不会有上面那种腐化风险:

**Prefill 侧(门控,尽力而为,非硬证明)**:`get_next_batch_to_run` 在调用
`_Upstream.get_next_batch_to_run` 之前,先跑
`_reorder_queue_for_atomic_fusion_admission`(整个方法持有 `self._request_admission_lock`——
和 `abort()` 自己改 `self.waiting_queue` 用的是同一把锁,因为 abort 可能从另一个线程——Stage
自己的事件循环,不是这个调度器的 tick 线程——并发跑进来;这把锁是 `threading.RLock()`,可重入,
所以下面第 5 步里 give-up 路径调用 `self.abort()`——它自己也会 `with self._request_admission_lock`
——不会自锁死):

1. 扫描 `self.waiting_queue`,按 `_fusion_group_members` 把请求分成"某融合组的成员"和
   "普通请求"。
2. 一个融合组若**不是**全部成员当前都在 `waiting_queue` 里(有的还没 build 完、有的在别处
   跑着、有的刚被 retract 还没归队),这一组当前在场的成员**整组暂扣**——不放进这次要交给
   upstream 的队列;这正是"部分成员被送进 prefill"这件事本身,不该发生。这种"缺员"不计入下面
   第 5 步的放弃计数——它总会自己收敛(build 迟早完成,或者 decode 侧兜底迟早把它级联 abort
   掉),不需要一个放弃机制。
3. 若这一刻有一个 chunked prefill 请求正在处理中(`self.chunked_req is not None`),这个 tick
   **所有**融合组一律暂扣——chunked 请求已经吃掉了这个 tick 一部分 chunk/input token 预算,
   而这个数字从外面读不到,与其按一个已经被吃掉一部分、读不出真实剩余值的预算去估算,不如整体
   保守跳过这个 tick。
4. 一个全员在场、且没有 chunked 请求在跑的组,先按请求数(而不是 token 数)上限检查——
   upstream 的准入循环不仅在 token 预算耗尽时停止,一旦
   `len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs)`(通常由
   `max_running_requests` 决定)也会整体停止,这和 token 预算是两个独立的维度,只查 token
   预算不够;超过请求数上限的组直接暂扣。这里的 `running_bs` 和下面 token 预算用的是**同一份**
   `_prefill_in_flight_reqs()`(见下)算出来的在途请求数,而不是单独去读
   `len(self.running_batch.reqs)`——原因和 token 预算完全一样:upstream 把上一个 tick 的
   batch 折进 `running_batch`这件事,发生在这个门控运行之后、真正的准入循环开始之前,门控这一刻
   看到的 `running_batch` 可能还没算上刚结束的那一批,如果只用它算请求数上限,会在 prefill 爆发
   后的那一个 tick 里把这个上限算大,放行一个实际会被请求数上限拦腰截断的组。
   再用 `_estimate_available_prefill_tokens`(镜像 `PrefillAdder.rem_total_tokens`/
   `budget_state` 的主项:可用 KV + 可驱逐 tree cache,减去 `_prefill_in_flight_reqs()`——
   `self.running_batch.reqs` 并上尚未被 upstream 折入的 `self.cur_batch`/`self.last_batch`
   (按 rid 去重)——预留的 `max_new_tokens` 上界,再用 `chunked_prefill_size`/
   `max_prefill_tokens` 分别封顶)和
   `_fusion_group_prefill_cost`(镜像 `add_one_req` 的主项:
   `len(origin_input_ids) + max_new_tokens + page_size`,`origin_input_ids` 是刻意偏保守的
   `extend_input_len` 近似值——如果 sglang 自己的 radix KV cache 真的给某个 sibling 命中了
   前缀、让它的真实 `extend_input_len` 比这个估计小,这只会让这一项的估算偏大,方向仍然安全,
   只会让门控更容易判断"装不下"而不是更容易误判"装得下")判断这一 tick 估计的空闲预算是否装
   得下整组;装得下就把这一组整体挪到 `waiting_queue` 最前面、成员紧邻排列、排在所有普通请求
   之前(多组都装得下时,按扫描到的顺序依次从同一份预算/名额里扣,像多个背包物品顺序装箱一样,
   不会让后一组的估计撞车);装不下就和"缺员"的组一样被整组暂扣,等下一个 tick(可能因为已运行
   请求推进了 decode 释放出更多 KV,或者 chunked 请求跑完了)再试一次——这种"预算/名额不够"的
   暂扣**会**计入第 5 步的放弃计数。
5. 一个组因为预算/名额不够被连续暂扣满 `_MAX_FUSION_WITHHOLD_TICKS`(200)个 tick 仍然没有
   被放行,就不再无限期暂扣下去——`_advance_withhold_ticks_and_give_up` 直接放弃它:对组里每个
   rid 都发一条 client 可见的 error(和 `_cascade_abort_split_fusion_group` 一样,对 follower
   的合成 rid 发也无妨,反正没人订阅那个 routing key),然后 `abort()` 掉组里任意一个成员(会
   级联清掉整组)。给出的 rid 集合会从这次暂扣列表里剔除,不会在下面第 6 步被放回队列。没有这个
   放弃路径,一个门控自己估算"永远装不下"的组会被无限期暂扣,客户端连一个最终的 abort/error 都
   等不到——这本身就是这个门控自己新引入的一种活性倒退,必须堵上。
6. 暂扣的(未被放弃的)请求在 upstream 调用返回后(`finally` 块里、同样持锁)立即放回
   `waiting_queue` 最前面。**放回之前会先剔除掉这段时间内被 `_aborted_request_ids` 标记过的
   rid**:一个正处于暂扣状态的请求本来就不在 `self.waiting_queue` 里,如果这时候另一个线程对
   它(或组里任意成员)调用了 `abort()`——不管是客户端主动 cancel,还是第 5 步自己的放弃路径——
   `abort()` 自己那份"从 `waiting_queue` 里摘除"的清理逻辑根本找不到它(它已经不在队列里了),
   但它确实已经被标记为 aborted、组注册表也被清掉了;如果放回时不做这个检查,这个已经被判了死刑
   的请求会被原样复活成一个不再属于任何融合组的普通请求,之后被正常 admit、解码,把已经废弃的
   请求当成正常任务跑到底。

**为什么"整组挪到队首"是必要的,不只是"全部在场"就够**:upstream 的准入循环
(`Scheduler._get_new_batch_prefill_raw` 内的 `for req in self.waiting_queue: ...
if res != CONTINUE: break`)是按队列顺序逐个尝试、一旦遇到装不下的请求就整体停止——不是
"跳过装不下的、继续找后面能装的"。如果一组的成员虽然都在场,但中间被普通请求隔开,某个插在
中间的无关请求恰好把预算耗尽,组内排在它后面的成员这一 tick 就轮不到,组照样被拆散,即使
"预算总量看起来够整组用"这个判断本身没错。把每个装得下的组整体挪到队首、成员相邻,能保证没有
无关请求能插进同一组的两个成员之间抢预算,这是让"总预算够"这个估计真正转化为"这组一定一起被
尝试"的关键一步,不能省略。这个前提依赖部署用的是默认的 FCFS 调度策略(`calc_priority` 对 FCFS
是纯 no-op,不会在门控之后再打乱队列顺序)——如果哪天切到 `lpm`/`lof`/`random` 之类会重排
`waiting_queue` 的策略,这里的"整组相邻"假设会被 upstream 自己的重排破坏,目前代码和文档都没有
去强制固定/校验 `schedule_policy`,这是一个隐式依赖,没有做防御性检查。

**这个门控是保守估计,不是精确复刻,也不是数学证明**:`_estimate_available_prefill_tokens`
故意只镜像 `PrefillAdder` 预算核算里"可用 KV/tree cache 减去在途请求预留、再按
chunked-prefill/整段 input token 上限封顶"这几项主项,不管 SWA/dllm/优先级抢占这些 Higgs TTS
用不到的分支,也不去精确复刻 `new_token_ratio` 折算——chunked prefill 本身**确实在用**
(`engine_builder.py` 设了 `chunked_prefill_size=8192`),不是"用不到",已经作为一个独立维度
折进了预算封顶里,而不是被忽略。本机没有真实引擎能拿来验证一份"精确复刻"是否真的处处一致,与其
做一个自信但可能在某个分支上算错、从而**放行本不该放行的组**的精确版本,不如做一个"宁可低估、
多等一个 tick"的保守版本——低估的唯一后果是这一 tick 白等,从不会让一个真的装不下的组被误判成
装得下。即便如此,这仍然是概率性的缓解,不是形式化证明"绝不会拆分":一个真实引擎里我们没建模到
的预算维度(比如 KV 压力下 retract 造成的组重新入队后的实际扣费、`page_size > 1` 时的对齐开销)
理论上仍可能让这里估计"装得下"的组被 upstream 实际只收了一部分——这也是下面 decode 侧兜底继续
保留、而不是被这个新门控取代的原因。

**Decode 侧(仍然保留,作为兜底,不是主防线,但仍是唯一的"正确性下限")**:
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

prefill 路径(`model.py::_batch_local_fusion`,prefill 完成后触发首个解码 token 的那一步)
**不再**保留整体 `RuntimeError`。移除组原子准入之前,这里的 raise 被认为"理论上不可达";
但原子准入已经不存在,sibling 的 prefill 落在不同批次是正常情况,再对整批硬 raise 会把这批
里所有无关请求一起炸掉——正是 BLOCKING-3 已经在 decode 侧修掉的那种"殃及无关请求"模式,不该
在 prefill 侧重新引入。现在的做法是隔离而非 raise:把这批里在场的成员降级为独立单例(不
参与融合)、打日志、并把这个 `group_id` 标记为 **poisoned**(`FusionRegistry.mark_poisoned`)。

**为什么需要"中毒"标记,而不只是隔离**:一个组在 prefill 侧被拆散后,可能会"自愈"——比如
sibling A 这一步单独 prefill(被隔离,采了一帧未融合的 codes),sibling B 下一步也单独
prefill(同样被隔离),再下一步两个都进了同一个 decode batch,这时候单看"在场人数是否等于
预期人数"会发现 2/2、完全正常,`_populate_fusion_buffers` 会误以为可以放心恢复融合——但
A 和 B 的 KV 上下文在各自被隔离的那一步已经各自采样了不同的、未融合的第一帧,从那一刻起就
永久错位了,绝不是"看起来凑齐了就可以继续"。中毒标记解决了这个问题:`_populate_fusion_
buffers` 现在即使发现人数凑齐,只要这个组曾经中毒,也会照样隔离+abort,不会被"看起来正常"
骗过去。

**仍然存在、这次没有补上的缺口**:`_batch_local_fusion` 只有 `HiggsGenParams`、没有
`Req` 句柄,没法自己把在场行标记为 abort——它能做的只是隔离+中毒,把"真正 abort"这件事
留给下一个碰到这个组的 decode step。也就是说,从"组被拆散、隔离生效"到"下一个 decode step
检测到中毒并真正 abort"之间,**这一步(以及自愈过程中每一次单独被隔离的那一步)已经采样出
的、未融合的一帧 codes 是真实输出,可能已经产出甚至被流式发出**。中毒标记保证了"组最终一定
会被 abort、不会被自愈骗过去悄悄产出成功结果",但不能追溯撤回已经发生的那一帧输出。这不是
"已解决",是一个已知的、尚未补上的正确性缺口。

**加了 prefill 侧门控之后,decode 侧兜底还会被触发到的场景(诚实说明,不是"已解决")**:
1. **KV 压力下的中途 retract**:门控只在 prefill 准入这一刻起作用——一个组已经全员准入、
   开始 co-batched decode 之后,如果调度器因为 KV 压力对 running batch 做 retract(选中了
   组里的某个成员退回 `waiting_queue`),这件事发生在 prefill 门控完全看不到的地方(运行中
   的请求,不在 `waiting_queue` 里),门控没有能力阻止,也不该去阻止(那是 upstream 自己的
   retract 淘汰逻辑,不是这个仓库拥有的代码路径)。这仍然要靠 decode 侧隔离+级联 abort 兜底。
2. **保守估计仍然可能偏乐观的边界情况**:`_estimate_available_prefill_tokens` /
   `_fusion_group_prefill_cost` 是主项(含 token 预算、chunked-prefill/input-token 上限、
   `max_running_requests` 请求数上限)的保守复刻,不是精确复刻(见上一段),真实引擎里没建模
   到的更细维度(比如 `prefill_max_requests`、context-parallel/LoRA 相关的额外分支、
   `page_size > 1` 时的对齐开销、KV 压力下 retract 拆散后重新入队的组的真实扣费)理论上仍可能
   让门控放行一个实际装不下的组。

以上任何一种情况发生时,融合请求会被 decode 侧检测到"缺员"从而直接 abort,而不是透明地多等
一两个 tick 再重试——客户端仍然需要能处理"融合请求返回一个 abort/error 而不是音频"的情况
(并可自行重试)。门控把这类情况的发生频率从"调度压力下的常态"压到"边界情况",但不是把它
归零,也不应该被包装成"两层原子保证都齐了"。

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
- prefill 侧原子准入门控(见上文)是保守估计 + 尽力而为,不是形式化证明;已运行请求中途被
  KV 压力 retract 拆散一个正在 co-batch decode 的组,门控看不到也管不了,仍然只能靠 decode
  侧隔离+级联 abort 兜底;客户端仍需要能处理"融合请求返回 abort 而不是音频"的情况并自行重试。
- 门控本身可能因为估算持续偏保守而反复暂扣同一个组;`_advance_withhold_ticks_and_give_up`
  在连续暂扣满 `_MAX_FUSION_WITHHOLD_TICKS`(200)个 tick 后放弃并对客户端报错,避免无限期
  暂扣、客户端连最终结果都等不到——但这个阈值本身是拍的一个数字,没有基于真实引擎下"正常情况
  最多需要暂扣几个 tick"的实测数据来标定。
- 单个 sibling 的 prompt 总长度(参考音频 + 目标文本 + 特殊 token)没有显式校验是否超出
  `chunked_prefill_size`,只靠 `_MAX_REF_AUDIO_SEC` 留的固定余量隐式兜底(见上一节)。
- sampler pool 容量 = `max_running_requests + 1`;一个融合请求占 N 行,部署需按
  "1 融合请求 = N 行" 给 KV/并发计费,否则 KV 压力下更容易触发 decode 侧的缺员 abort。
- 仅 logit 融合(非 prompt 拼接);CFG 外推留作 follow-up。
- 融合请求完全绕开现有的参考缓存栈(speaker artifact cache / `reference_code_cache_key` /
  已上传语音),每次都会重新过一遍 codec 编码,已上传的具名语音也无法参与融合。作为 v1 范围
  可以接受,但这是一个尚未文档化过的能力缺口。

## 已修复的两个"隔离/级联"自身的 bug(第二轮对抗审查抓到,均已修正)
第一版的 decode 侧隔离 + 级联 abort(上一节)在实现细节上曾有两处自己的问题,均已修正:

1. **`_populate_fusion_buffers` 的 dirty-flag 越界**:早期实现只把"清空"写到 `[:bs]`(当前
   这一步的 batch size),但 `_cg_fusion_group`/`_cg_fusion_weight` 是固定 `pool_size` 的
   buffer,不随每步 batch size 变化。若一步 bs=4(含一个融合组占 2,3 号槽位)之后,组结束、
   batch 缩到 bs=2,只清了 `[:2]`,槽位 2,3 仍留着旧组的 `group=[2,2]`/`weight=[0.6,0.4]`;
   若 dirty flag 此时被错误地设为 False,后续 batch 再长回 bs=4、两个全新无关请求恰好落在
   槽位 2,3,`has_any_fusion()` 早退检查会让这两个不相关的请求被静默按旧组"融合"在一起。
   修法:dirty→clean 的那次转换改成清空整个 buffer(`[:pool_size]`),而不是只清 `[:bs]`,
   一次性、简单、正确,不需要额外的高水位状态。回归测试:
   `test_populate_buffers_clean_reset_scrubs_stale_slots_beyond_a_shrunk_batch`
   (`test_voice_fusion_pipeline.py`)。
2. **级联 abort 对"同批已完成但还没被下一次 filter_batch 清理"的成员会误伤**:一个组的多个
   在场成员如果*同时*被 `_populate_fusion_buffers` 隔离+标记 FINISH_ABORT,它们会在同一次
   `stream_output` 调用里各自轮到自己的处理——但第一个被处理的成员触发的级联,如果不加区分地
   对"组里其它还注册着的成员"调用 `abort()`,会把同批里*也*刚完成、还没被下一轮 `filter_batch`
   清理出 `running_batch.reqs` 的成员也 `abort()` 掉——而 `abort()` 对"已完成但还在批里"的
   请求走的是立即从 `running_batch.reqs` 摘除这条路径,和 BLOCKING-2 里禁止的"prepare_for_
   extend 之后摘 reqs"是同一类张量/reqs 错位。修法:`stream_output` 先收集本轮所有"本次调用
   即将完成"的 rid 集合,级联只对*不在*这个集合里的成员(即真正缺席、下一轮才可能自己冒出来
   的那个 sibling)调用 `abort()`——同批一起完成的成员各自走自己的正常处理,不需要被级联。

## 仍需真实引擎验证的项(本机 Windows 无 sglang,无法跑真实引擎)
1. **decode 侧隔离 + 级联 abort 的端到端行为**:`_populate_fusion_buffers` 的隔离逻辑和
   `_cascade_abort_split_fusion_group` 的级联逻辑都各自过了单测(`test_voice_fusion_pipeline.py`
   里 mock 了 `HiggsTTSModelRunner`/`FusionRegistry` 直接调用真实方法),但两者串起来的真实
   端到端行为(KV 压力下真的触发 retract → 隔离 → 级联 → 客户端收到什么)只验证到"逻辑上
   应该正确",没有在真实引擎上跑过。
2. **prefill→decode 过渡的实际发生率,以及新增门控的实际效果**:加了
   `_reorder_queue_for_atomic_fusion_admission` 门控之后,sibling 们在真实负载下实际上有
   多大比例一起进 decode(而不是触发 decode 侧的 abort 路径),门控生效前后这个比例的对比,
   `_estimate_available_prefill_tokens` 的保守估计在真实 KV 池/tree cache 状态下是否经常
   过度保守(不必要地多等 tick,浪费吞吐)或——更需要关注——是否存在被低估掉的预算维度导致
   估计过于乐观、仍然放行了实际装不下的组。这些只有在真实引擎 + 真实负载下才能量化,直接决定
   这个功能在生产环境下的可用性,需要用不同长度的参考音频、不同并发度实测。
3. **CUDA graph 兼容性**:`_cg_fusion_group`/`_cg_fusion_weight` 每步重填是否与图重放兼容。
4. **非融合热路径零开销**:`FusionRegistry.has_any()` 让零融合流量的服务器跳过每步的
   follower 检查/buffer 填充,已有针对计数器本身的纯 Python 单测
   (`test_voice_fusion.py` 的 `test_registry_*`),但吞吐层面的"确实零开销"还没有真实
   engine 下的压测数据支撑;此外 `fuse_group_logits` 在 CG 解码路径上是无条件调用的(不像
   eager 路径有 `is_fused` 门槛),对全零融合流量的服务器而言这是一次额外的、可忽略但确实
   存在的 fp32 softmax 开销,不是真正的零成本。
5. **`FINISH_ABORT` 直接赋值 `finished_reason` 而非走 `to_finish` 的时机是否安全**:上游
   `Req.to_finish` 字段的注释明确写着"如果想在事件循环中途 abort 一个请求,应该设置
   `to_finish` 而不是直接设置 `finished_reason`,否则请求会被过滤掉、再也不会响应"。
   **更正**(第二轮审查抓到第一版这里的错误理解):upstream 的 `process_batch_result_decode`
   确实每个 decode step 都会调用 `req.check_finished()`,不是"Higgs 完全不走这条路"。Higgs
   TTS 的整条完成信号链路(包括早已验证工作的正常完成路径 `_mark_sampler_finished`)在这个
   相同的代码层级都是直接设置 `finished_reason`,从未使用 `to_finish`——`_populate_fusion_
   buffers`(BLOCKING-3)、`_cascade_abort_split_fusion_group` 都是照抄这个既有约定,而不是
   新造一种写法。但"Higgs 这套直接赋值的写法为什么能绕开上游注释里的警告、和 `check_finished()`
   的实际交互到底是不是安全的",仍然是一个没有在真实引擎上专门针对 abort 路径验证过的开放问题,
   不是已经证明安全。
6. **异步 lookahead 与 fusion 的交互(已按最保守方式规避,未实测)**:这个仓库自建的
   one-step-lookahead 解码(`enable_async_decode`,`OmniScheduler._resolve_and_process`)
   会在 launch 阶段(`_populate_fusion_buffers` 设置 FINISH_ABORT 发生的地方)之后、resolve
   阶段之前有一个时间窗口;`_resolve_and_process` 用"resolve 前先快照 `req.finished()`"的
   方式区分"上一步就已结束的过期行"和"这一步 resolve 过程中才结束的行",但 launch 阶段设置的
   FINISH_ABORT 发生在这个快照*之前*,会被误判成"上一步的过期行"而被整行摘出批次、永远不会
   走到 `stream_output`,级联 abort 也就永远不会触发——这正是级联机制想避免的"僵尸 sibling"。
   修法:`HiggsTTSModelRunner.lookahead_eligible` 现在在有任何融合流量时返回 `False`
   (`model.has_any_fusion()`),让融合相关的 batch 强制走同步路径,整个 launch/resolve 分离
   带来的时间窗口根本不存在。这个修法本身只在本机做了静态代码走查(确认
   `lookahead_eligible` 返回值确实能让 `_event_loop_async_decode` 完整跳过 launch/resolve
   分离,直接同步跑),没有在真实开启 `enable_async_decode` 的引擎上实测过。
7. **prefill 门控与并发 `abort()` 的真实交互,以及放弃阈值的标定**:
   `_reorder_queue_for_atomic_fusion_admission`/`_restore_queue_after_atomic_fusion_admission`
   持 `self._request_admission_lock` 防止一个正在暂扣窗口期内的请求被另一个线程的 `abort()`
   复活(见"Co-batching"一节),这个交互只在单测里用 mock/手写的 `abort` 验证过锁本身可重入、
   以及 restore 会剔除 `_aborted_request_ids` 里的 rid,没有在真实并发(多个请求同时到达/
   取消、真实 GIL 调度)下跑过。`_MAX_FUSION_WITHHOLD_TICKS=200` 这个放弃阈值也是拍的,没有
   基于真实引擎下"正常场景最多暂扣几个 tick 就该放行"的数据标定过,过小可能在正常波动下就误杀
   本该等等就成的组,过大则放大"客户端要等很久才会等到最终失败"的体感延迟。
