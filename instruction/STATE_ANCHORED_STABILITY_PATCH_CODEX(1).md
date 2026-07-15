# State-Anchored TD-MPC2 数值稳定性修正规格

> 在当前工程中的 State-Anchored 实现上完成本补丁。  
> 只修改 State-Anchored 新增代码和相应测试，不修改现有 TD-MPC2 原文件。  
> 本补丁只完成本文列出的三项修正，不引入其他算法变化或新功能。

---

## 1. 修正目标

当前实现需要修正以下三个问题：

1. state prediction loss 与网络输入共用带 clip 的 normalization，导致预测超出 clip 范围后 loss 对进一步发散失去梯度；
2. dynamics 输出无界 normalized delta，再乘以较大的 \(\sigma\) 恢复 raw state，可能造成 imagined state 数值爆炸；
3. normalizer 当前从 \(\mu=0,\sigma=1\) 开始。应在 `cfg.seed_steps` 随机交互完成后，先使用 seed replay buffer 中所有真实 state 精确初始化 \(\mu,\sigma\)，然后继续按照原来的 Welford 逻辑用后续 replay minibatch 更新，直至达到原有的 freeze update 数。

修正后采用：

\[
x_t=\frac{s_t-\mu}{\sigma},
\]

其中 \(x_t\) 是未截断、可逆的 normalized state carrier。

网络输入使用：

\[
\tilde x_t=\operatorname{clip}(x_t,-c,c).
\]

辅助特征为：

\[
u_t=\phi_\theta(\tilde x_t).
\]

Dynamics 输出有界 normalized delta：

\[
\widehat{\Delta x_t}
=
L\tanh\left(
\frac{
d_\theta([\tilde x_t,u_t],a_t)
}{L}
\right),
\]

并在 normalized state space 内递归：

\[
\hat x_{t+1}
=
x_t+\widehat{\Delta x_t}.
\]

raw state 只在需要解释、记录或未来调用显式状态函数时恢复：

\[
\hat s_{t+1}
=
\mu+\sigma\odot\hat x_{t+1}.
\]

默认：

\[
c=10,\qquad L=5.
\]

---

# 2. 不得改变的内容

本补丁不得修改：

- learned reward；
- policy-based TD target；
- Q ensemble；
- distributional reward/Q；
- MPPI planner；
- policy update；
- replay sampling；
- loss coefficients，包括当前 `sa_state_coef`；
- network widths/depths；
- feature encoder；
- optimizer；
- planning hyperparameters；
- official TD-MPC2 files。

特别注意：

```text
本补丁不调整 sa_state_coef。
```

本次实验只隔离验证 normalization、normalized carrier、bounded delta 和 seed-buffer initialization 的影响。

---

# 3. 需要修改的文件

主要修改：

```text
state_anchored/config.py
state_anchored/layers.py
state_anchored/world_model.py
state_anchored/agent.py
```

相应更新：

```text
tests/test_state_anchored_norm.py
tests/test_state_anchored_model.py
tests/test_state_anchored_smoke.py
```

如需新增只属于 State-Anchored 的辅助文件，可以新增，但不得修改：

```text
common/
trainer/
envs/
tdmpc2.py
train.py
config.yaml
```

---

# 4. 修正一：分离 unclipped normalization 与 clipped network input

## 4.1 `RunningStateNorm` 新接口

在 `state_anchored/layers.py` 中，将当前单一 `normalize()` 拆分为以下接口。

### `normalize_unclipped`

```python
def normalize_unclipped(
    self,
    state: torch.Tensor,
) -> torch.Tensor:
    """
    Convert raw state to normalized state without clipping.

    This function is used for:
    - recursive model carrier initialization;
    - state prediction targets;
    - state consistency loss;
    - numerical diagnostics.
    """
```

公式：

\[
N_{\mathrm{raw}}(s)
=
\frac{s-\mu}{\sigma}.
\]

不得使用 clamp。

### `clip_normalized`

```python
def clip_normalized(
    self,
    normalized_state: torch.Tensor,
) -> torch.Tensor:
    """
    Clip an already-normalized state for neural-network input.
    """
```

公式：

\[
C(x)=\operatorname{clip}(x,-c,c).
\]

若 `clip is None`，直接返回输入。

### `normalize_for_input`

```python
def normalize_for_input(
    self,
    state: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize a raw state, then clip it for neural-network input.
    """
    return self.clip_normalized(
        self.normalize_unclipped(state)
    )
```

### `denormalize`

保留：

\[
s=\mu+\sigma\odot x.
\]

它接收未截断 normalized state。

## 4.2 不再使用含糊的 `normalize()`

删除或停止在 State-Anchored 代码中调用当前含 clip 的：

```python
normalize(...)
```

所有调用位置必须明确选择：

```python
normalize_unclipped(...)
normalize_for_input(...)
clip_normalized(...)
```

如果为了兼容保留 `normalize()`，必须：

- 将其标记为 deprecated；
- 不得在 world model、agent 或测试的核心路径中调用；
- 不得让 Codex自行决定它等价于哪一个接口。

## 4.3 State loss 必须完全 unclipped

修正后，state consistency loss 直接比较未截断 normalized carrier：

\[
\mathcal L_s
=
\ell(\hat x_{t+k},x_{t+k}).
\]

代码应为：

```python
predicted_x = predicted_carrier
target_x = target_carrier

if cfg.sa_state_loss == "mse":
    step_loss = F.mse_loss(
        predicted_x,
        target_x,
    )
else:
    step_loss = F.smooth_l1_loss(
        predicted_x,
        target_x,
        beta=cfg.sa_state_loss_beta,
    )
```

禁止：

```python
clip(predicted_x)
clip(target_x)
normalize_for_input(predicted_raw_state)
normalize_for_input(target_raw_state)
```

一旦预测超过 \(\pm c\)，loss 仍必须继续增长，并保持有效梯度。

## 4.4 网络输入仍然保留 clip

以下模块的 state 输入仍使用：

\[
\tilde x=C(x).
\]

包括：

- auxiliary feature encoder；
- dynamics；
- reward；
- Q；
- policy；
- termination。

clip 只是保护神经网络输入，不得修改递归 carrier 本身。

---

# 5. 修正二：在 normalized state space 内递归，并限制 delta

## 5.1 Carrier 定义改变

当前实现中：

```text
carrier = raw state s
```

修正后：

```text
carrier = unclipped normalized state x
```

这是一个可逆仿射坐标变换，不是自由 latent：

\[
x=N_{\mathrm{raw}}(s),\qquad
s=N_{\mathrm{raw}}^{-1}(x).
\]

状态维度保持：

\[
d_x=d_s.
\]

每一维仍与原始 state 一一对应。

## 5.2 `StateAnchoredWorldModel.encode`

修改为：

```python
def encode(
    self,
    obs: torch.Tensor,
    task=None,
) -> torch.Tensor:
    self._assert_single_task(task)
    return self.state_norm.normalize_unclipped(obs)
```

输入：

```text
raw environment state s
```

输出：

```text
unclipped normalized state x
```

不要返回 raw state，也不要返回 auxiliary feature。

## 5.3 新增 `decode_state`

```python
def decode_state(
    self,
    normalized_state: torch.Tensor,
) -> torch.Tensor:
    return self.state_norm.denormalize(
        normalized_state
    )
```

用于：

- diagnostics；
- evaluation of raw-state magnitudes；
- future explicit reward/constraint interfaces。

当前 reward/Q/policy 路径不需要先 decode。

## 5.4 `features` 接收 normalized carrier

修改：

```python
def features(
    self,
    normalized_state: torch.Tensor,
) -> torch.Tensor:
    network_state = (
        self.state_norm.clip_normalized(
            normalized_state
        )
    )
    feature = self._encoder(network_state)
    return torch.cat(
        [network_state, feature],
        dim=-1,
    )
```

重要：

- `features()` 的输入已经是 normalized carrier；
- 不得再次减 \(\mu\) 或除以 \(\sigma\)；
- 不得缓存 feature；
- 每一个 rollout step 都从当前 \(x_t\) 重新计算 \(u_t\)。

## 5.5 Dynamics 输出限制

在 `state_anchored/config.py` 中新增：

```python
"sa_delta_limit": 5.0,
```

验证：

```python
if cfg.sa_delta_limit <= 0:
    raise ValueError(
        "sa_delta_limit must be positive."
    )
```

在 `world_model.next()` 中：

```python
def next(
    self,
    normalized_state: torch.Tensor,
    action: torch.Tensor,
    task=None,
) -> torch.Tensor:
    self._assert_single_task(task)

    anchored = self.features(normalized_state)

    raw_delta = self._dynamics(
        torch.cat([anchored, action], dim=-1)
    )

    limit = self.cfg.sa_delta_limit

    bounded_delta = limit * torch.tanh(
        raw_delta / limit
    )

    return normalized_state + bounded_delta
```

公式：

\[
\widehat{\Delta x}
=
L\tanh(\tilde{\Delta x}/L).
\]

采用该形式而不是：

\[
L\tanh(\tilde{\Delta x}),
\]

原因是前者在原点附近导数为 1：

\[
\left.
\frac{\partial}{\partial y}
L\tanh(y/L)
\right|_{y=0}
=1.
\]

这样不会在小 delta 区域人为放大梯度。

## 5.6 禁止再次乘以 \(\sigma\)

删除当前 `next()` 中的：

```python
state_norm.scale_delta(...)
raw_state + std * delta
```

递归过程中不得进行：

```text
normalized delta -> raw delta -> raw next state
```

整个 imagined rollout 保持：

```text
x_t -> x_{t+1} -> x_{t+2}
```

只在 diagnostics 等需要时调用 `decode_state()`。

## 5.7 Direct prediction path

当前 `sa_predict_delta=False` 的 direct-state prediction 分支不再属于本补丁设计。

推荐固定只支持 normalized residual dynamics，并在 config validation 中拒绝：

```python
if not cfg.sa_predict_delta:
    raise NotImplementedError(
        "The stabilized State-Anchored model "
        "only supports normalized delta prediction."
    )
```

## 5.8 Zero initialization

保留 dynamics 输出层 zero initialization。

初始时：

\[
\tilde{\Delta x}=0,
\qquad
\widehat{\Delta x}=0,
\qquad
\hat x_{t+1}=x_t.
\]

---

# 6. Reward、Q、policy 和 termination 的输入修改

这些 heads 的算法与结构保持不变。

由于它们现在接收 normalized carrier，内部统一调用：

```python
anchored = self.features(normalized_state)
```

然后保持原来的 head 计算。

禁止：

- 在 head 中调用 `encode()`；
- 将 normalized carrier 当作 raw state 再 normalize；
- 将 carrier decode 后再送回 feature encoder。

---

# 7. 修正三：在 seed collection 后用整个 buffer 精确初始化 \(\mu,\sigma\)，之后继续在线更新

## 7.1 初始化时机

训练流程在：

```python
self._step == cfg.seed_steps
```

时开始 seed-data pretraining。

State-Anchored agent 的第一次 `update(buffer)` 必须在任何以下操作前完成 normalizer 初始化：

- TD target；
- world-model rollout；
- reward/Q/policy forward；
- optimizer step。

逻辑：

```python
if not self.model.state_norm.initialized:
    all_seed_states = extract_all_states(buffer)
    self.model.state_norm.fit(all_seed_states)
```

然后才能进行第一次训练 update。

## 7.2 使用哪些 state

必须使用 seed replay buffer 当前已存储的全部真实 observation：

```text
每个 buffer entry 的 obs，恰好统计一次
```

要求：

- 不使用 sampled subsequences；
- 不重复统计同一个 replay entry；
- 不使用 imagined states；
- 不使用 evaluation states；
- 不使用未来训练 batch 反复更新；
- 过滤非 finite observation；
- state 最后一维必须等于 `state_dim`。

buffer 中 episode 的初始 observation 也是合法 state，可以统计。

如果 seed collection 结束时存在尚未写入 replay 的未完成 episode，只统计 replay 中实际已经存储的 entries，并在日志中报告实际数量。

## 7.3 不修改官方 Buffer

不得修改：

```text
common/buffer.py
```

在 State-Anchored 代码中实现私有 helper，例如：

```python
def _all_replay_states(self, buffer) -> torch.Tensor:
    ...
```

Codex 必须先检查当前 TorchRL `ReplayBuffer` 和 `LazyTensorStorage` API，并使用当前版本可工作的读取方式。

优先使用 ReplayBuffer/Storage 的公开读取或索引接口；如果只能访问当前 wrapper 的内部字段，必须：

- 将访问封装在一个 State-Anchored helper 中；
- 检查字段存在；
- 对 API 不匹配给出清晰错误；
- 不静默退回随机 sampling。

可接受的逻辑示意：

```python
num_entries = len(buffer._buffer)
stored = buffer._buffer[:num_entries]
states = stored.get("obs")
```

具体调用以当前安装版本实际接口为准。

## 7.4 `RunningStateNorm.fit`

新增：

```python
@torch.no_grad()
def fit(
    self,
    states: torch.Tensor,
) -> None:
    """
    Reset statistics and fit exactly once to all
    provided real replay states.
    """
```

要求：

1. reshape 为 `[-1, state_dim]`；
2. 过滤非 finite rows；
3. 至少需要两个有效 state；
4. 重置 `count/mean/m2`；
5. 对全部有效 state 做一次 batch Welford update；
6. 设置 registered buffer：
   ```python
   initialized = True
   ```
7. 输出统计数量、mean abs、std mean/min/max。

新增：

```python
self.register_buffer(
    "initialized",
    torch.tensor(False, dtype=torch.bool),
)
```

`initialized` 必须随 model checkpoint 保存和恢复。

## 7.5 初始化后继续按原逻辑更新统计量

完成 seed-buffer `fit()` 后，完整 seed replay 的统计量作为 Welford 状态的初始值：

```text
count = seed buffer 中参与统计的有效 state 数
mean = seed buffer 的精确均值
m2 = seed buffer 的精确平方离差和
```

之后继续保留当前 normalizer 的在线更新机制。每次训练 update 从 replay buffer 采样到真实 observation sequence 后，继续调用：

```python
self.model.state_norm.update(obs)
```

必须保留：

```python
_sa_update_count
sa_norm_freeze_updates
```

并维持原来的冻结规则：

```python
if self._sa_update_count < cfg.sa_norm_freeze_updates:
    self.model.state_norm.update(obs)
```

要求：

- seed-buffer `fit()` 只执行一次；
- `fit()` 之后的第一次及后续训练 update 都继续使用真实 replay minibatch 更新；
- imagined states 永远不能更新 normalizer；
- evaluation states 永远不能更新 normalizer；
- 达到 `sa_norm_freeze_updates` 后停止更新；
- resume training 时从 checkpoint 恢复 normalizer 统计量和 freeze counter，不能重新 fit seed buffer；
- 如果当前 `_sa_update_count` 不会被 checkpoint 保存，必须修复该问题，但不得改变 freeze 语义。

## 7.6 Resume 与 evaluation

如果 checkpoint 已包含：

```text
count
mean
m2
initialized=True
```

则：

- resume training 不得重新 fit；
- evaluation 不得读取 replay buffer；
- 直接使用 checkpoint 中统计量。

若 evaluation 加载的 checkpoint 中 normalizer 未初始化，应立即报错。

旧版 State-Anchored checkpoint 与新 carrier 语义不兼容。无需兼容旧 checkpoint，但必须给出清晰错误，不得静默加载后运行。

---

# 8. Agent update 修改

## 8.1 `update(buffer)`

修改为：

```python
def update(self, buffer):
    if not bool(
        self.model.state_norm.initialized.item()
    ):
        states = self._all_replay_states(buffer)
        self.model.state_norm.fit(states)

    obs, action, reward, terminated, task = (
        buffer.sample()
    )

    if task is not None:
        raise NotImplementedError(...)

    if (
        self._sa_update_count
        < self.cfg.sa_norm_freeze_updates
    ):
        self.model.state_norm.update(obs)

    self._sa_update_count += 1

    torch.compiler.cudagraph_mark_step_begin()

    return self._update(
        obs,
        action,
        reward,
        terminated,
    )
```

## 8.2 `_update`

真实 target states：

```python
with torch.no_grad():
    next_state = self.model.encode(obs[1:], task)
```

此时 `next_state` 是未截断 normalized target state。

起点：

```python
state = self.model.encode(obs[0], task)
```

递归：

```python
state = self.model.next(
    state,
    action_t,
    task,
)
```

state loss 直接比较：

```python
state
target_next_state
```

不要再次 normalize 或 clip。

---

# 9. Diagnostics 修改

假设 `states` 是 normalized carriers。

## 9.1 Raw imagined state magnitude

```python
raw_states = self.model.decode_state(states)
imagined_state_abs_max = raw_states.abs().max()
```

## 9.2 Unclipped normalized magnitude

```python
normalized_imagined_state_abs_max = (
    states.abs().max()
)
```

这个值不应再被固定卡在 10。

## 9.3 Predicted normalized delta

```python
predicted_delta = states[1:] - states[:-1]
```

记录：

```text
predicted_normalized_delta_abs_mean
predicted_normalized_delta_abs_max
```

并验证：

\[
|\Delta x|\le L+\epsilon.
\]

## 9.4 NRMSE

真实 raw observation 先编码：

```python
real_x = self.model.encode(real_states)
```

然后：

\[
\operatorname{NRMSE}(k)
=
\mathbb E
\sqrt{
\frac1{d_s}
\|\hat x_{t+k}-x_{t+k}\|_2^2
}.
\]

不得 clip。

## 9.5 新增 clip fraction

记录：

```python
imagined_state_clip_fraction = (
    states.abs() > cfg.sa_norm_clip
).float().mean()
```

真实状态：

```python
real_x = self.model.encode(real_states)

real_state_clip_fraction = (
    real_x.abs() > cfg.sa_norm_clip
).float().mean()
```

如果 `sa_norm_clip is None`，两个值记录为 0。

## 9.6 Normalizer initialization diagnostics

第一次 fit 后打印或记录：

```text
state_norm_num_states
state_norm_mean_abs
state_norm_std_mean
state_norm_std_min
state_norm_std_max
```

`state_norm_num_states` 必须等于实际参与统计的 replay entries 数，而不是：

```text
number of gradient updates × sampled batch size
```

---

# 10. Config 修改

在 `state_anchored/config.py` 中：

## 新增

```python
"sa_delta_limit": 5.0,
```

## 保留

```python
"sa_norm_freeze_updates": 100_000,
```

及其现有 validation 和冻结逻辑。

同时保留

```python
"sa_norm_clip": 10.0,
"sa_norm_eps": 1e-5,
"sa_norm_min_std": 1e-3,
"sa_zero_init_dynamics_output": True,
"sa_state_coef": 5.0,
```

不要在本补丁中改 `sa_state_coef`。

---

# 11. 测试要求

## 11.1 Normalizer tests

新增或更新：

1. `normalize_unclipped()` 超过 clip 时不截断；
2. `normalize_for_input()` 正确截断；
3. normalized value 超过 10 时，unclipped loss gradient 非零；
4. `fit(all_states)` 与精确 mean/unbiased variance 一致；
5. `fit()` 会重置旧统计；
6. initialized/count/mean/m2 的 checkpoint roundtrip；
7. finite-row filtering。

## 11.2 World model tests

验证：

1. `encode(raw_state)` 输出未截断 normalized carrier；
2. `decode_state(encode(raw_state))` 恢复 raw state；
3. zero-init 时 `next(x,a)==x`；
4. bounded delta 满足：
   \[
   |\Delta x|\le \texttt{sa_delta_limit}+\epsilon;
   \]
5. feature encoder 实际接收 clipped \(x\)；
6. carrier 自身未被 clip；
7. reward/Q/policy shapes 保持不变；
8. temporal rollout finite。

## 11.3 Agent/buffer initialization tests

使用 fake replay buffer，验证：

1. 第一次 update 在任何 model forward 前调用 `fit()`；
2. `fit()` 完成瞬间，count 等于 seed buffer 中有限 state 数；
3. `fit()` 完成瞬间，mean/std 等于全部 stored states 的精确统计；
4. 完成第一次及后续 minibatch update 后，count 会继续增长；
5. 第二次 `agent.update(buffer)` 不得再次调用 `fit()`；
6. 达到 `sa_norm_freeze_updates` 后 count 不再变化；
7. checkpoint load 后不重新 fit，并正确恢复 freeze counter；
8. helper 读取的是全部 storage，而不是 sampled minibatch。

## 11.4 Loss unclipped test

构造：

```text
target x = 0
predicted x = 20
clip = 10
```

确认 state loss 基于 20，而不是 10，并且 gradient 非零。

## 11.5 Smoke test

```bash
python -m unittest discover \
  -s tests \
  -p 'test_state_anchored_*.py' \
  -v
```

真实环境：

```bash
python train_state_anchored.py \
  task=walker-walk \
  steps=20000 \
  eval_freq=10000 \
  compile=false \
  enable_wandb=false \
  save_video=false \
  exp_name=sa-stability-smoke
```

通过后测试 `compile=true`。

---

# 12. 验收条件

训练日志必须满足：

1. 第一次 pretraining update 前完成 normalizer fit；
2. normalizer 初始化完成瞬间，`state_norm_num_states` 等于 seed replay 实际 entries 数；
3. 后续 minibatch update 期间 normalizer count 继续增长，达到 `sa_norm_freeze_updates` 后停止变化；
4. `normalized_imagined_state_abs_max` 不再永远等于 10；
5. state loss 对超过 clip 的预测仍有梯度；
6. `predicted_normalized_delta_abs_max <= sa_delta_limit + tolerance`；
7. 不出现 NaN/Inf；
8. checkpoint save/load 后统计量和评估行为一致；
9. official TD-MPC2 原文件无 diff。

---

# 13. Codex 最终汇报要求

完成后汇报：

1. 修改的 State-Anchored 文件；
2. normalizer 如何从完整 seed buffer 提取 state；
3. seed buffer 初始化阶段实际统计了多少个 state；
4. 初始化后 normalizer 如何继续更新，以及在哪个 update 数冻结；
5. normalized carrier 的接口变化；
6. bounded delta 的具体实现；
7. 所有测试及结果；
8. `compile=false` smoke test 结果；
9. `compile=true` smoke test 结果；
10. 旧 checkpoint 是否不兼容；
11. 确认没有修改 official TD-MPC2 文件。

---

# 14. 不允许的偏离

不得：

- 修改 `sa_state_coef`；
- 删除或停用 `sa_norm_freeze_updates`；
- 在 seed-buffer `fit()` 后错误地停止 normalizer 更新；
- 修改 dynamics 宽度或层数；
- 删除 feature encoder；
- 改 Q target；
- 改 reward model；
- 改 planner；
- 加 ensemble；
- 加 known reward；
- 加 reanalysis；
- 修改官方 buffer/trainer；
- 用 replay minibatch 反复采样近似全部 seed states；
- 在 state loss 上使用 clip；
- 在 normalized delta 后再次乘 \(\sigma\) 进行递归；
- 静默兼容旧 State-Anchored checkpoint。

---

# 15. 最终公式

必须实现：

\[
x_t
=
\frac{s_t-\mu}{\sigma},
\]

\[
\tilde x_t
=
\operatorname{clip}(x_t,-c,c),
\]

\[
u_t
=
\phi_\theta(\tilde x_t),
\]

\[
\widehat{\Delta x_t}
=
L\tanh\left(
\frac{
d_\theta([\tilde x_t,u_t],a_t)
}{L}
\right),
\]

\[
\hat x_{t+1}
=
x_t+\widehat{\Delta x_t},
\]

\[
\hat s_{t+1}
=
\mu+\sigma\odot\hat x_{t+1}.
\]

其中：

- imagined rollout 递归的是 \(\hat x\)；
- \(\hat x\) 不被 clip；
- clip 只用于神经网络输入；
- state loss 在 \(\hat x\) 上计算；
- \(\mu,\sigma\) 在 seed collection 后先由完整 seed replay 精确初始化；
- 此后继续使用真实 replay minibatch 按原 Welford 逻辑更新，直到 `sa_norm_freeze_updates`。
