# State-Anchored TD-MPC2：Codex 工程实现规格

> 在不修改现有 TD-MPC2 原文件的前提下，新增一个可独立训练和评估的 State-Anchored TD-MPC2 版本。

---

## 0. 最高优先级约束

### 0.1 不允许修改官方实现

不得修改以下官方文件，且不限于这些文件：

- `tdmpc2.py`
- `train.py`
- `evaluate.py`
- `config.yaml`
- `common/world_model.py`
- `common/layers.py`
- `common/buffer.py`
- `common/parser.py`
- `trainer/*`
- `envs/*`

允许：

- import 和复用官方类、函数；
- 继承官方 `TDMPC2`；
- 调用官方 planner、policy update、TD target、buffer、trainer、logger；
- 新建独立模块、训练入口和评估入口。

完成后执行：

```bash
git diff -- tdmpc2.py train.py evaluate.py config.yaml common trainer envs
```

以上官方路径应无任何 diff。

### 0.2 当前实现范围

当前版本只支持：

```text
single-task + online RL + obs=state
```

必须显式拒绝：

- `cfg.multitask == True`
- `cfg.obs != "state"`

### 0.3 保留 TD-MPC2 的其余训练与控制算法

以下逻辑必须保持官方行为：

- replay buffer 与 sequence sampling；
- MPPI planning；
- policy trajectory proposals；
- learned reward distribution；
- distributional Q；
- Q ensemble；
- target Q Polyak update；
- policy entropy objective；
- TD target 中使用 policy prior；
- termination model；
- discount；
- action masking相关接口；
- logger、trainer、checkpoint 流程。

唯一核心变化：

```text
官方：obs -> latent z -> recursively predict z'
新版本：obs/state s -> anchored features -> recursively predict raw s'
```

---

# 1. 方法定义

## 1.1 官方 TD-MPC2

官方世界模型使用：

\[
z_t = h_\theta(s_t),
\]

\[
z_{t+1} = d_\theta(z_t,a_t),
\]

reward、policy 和 Q 均以 latent \(z\) 为输入。

## 1.2 State-Anchored TD-MPC2

定义 raw state：

\[
s_t\in\mathbb R^{d_s}.
\]

运行统计归一化：

\[
x_t=N(s_t)=\operatorname{clip}
\left(
\frac{s_t-\mu}{\max(\sigma,\sigma_{\min})},
-c,c
\right).
\]

辅助高维特征：

\[
u_t=\phi_\theta(x_t)\in\mathbb R^{d_u}.
\]

Anchored feature：

\[
\psi_\theta(s_t)=
\begin{bmatrix}
x_t\\
u_t
\end{bmatrix}.
\]

Dynamics 预测 normalized state delta：

\[
\widehat{\Delta x_t}
=
f_\theta(\psi_\theta(s_t),a_t).
\]

恢复 raw next state：

\[
\hat s_{t+1}
=
s_t+\sigma\odot\widehat{\Delta x_t}.
\]

下一步必须从预测的 raw state 重新计算 feature：

\[
u_{t+1}
=
\phi_\theta(N(\hat s_{t+1})).
\]

禁止直接预测或递归传播 \(u_{t+1}\)。

核心性质：

```text
递归 world-model carrier = raw state s
高维 u = 每一步由当前 state 重新计算的辅助特征
```

---

# 2. 新增文件结构

在官方源码目录（与 `common/`、`trainer/`、`tdmpc2.py` 同级）新增：

```text
state_anchored/
├── __init__.py
├── config.py
├── layers.py
├── world_model.py
└── agent.py

train_state_anchored.py
evaluate_state_anchored.py

tests/
├── test_state_anchored_norm.py
├── test_state_anchored_model.py
└── test_state_anchored_smoke.py
```

不要覆盖任何已有文件。

---

# 3. 配置策略

## 3.1 不修改官方 `config.yaml`

`train_state_anchored.py` 继续加载官方：

```python
@hydra.main(config_name="config", config_path=".")
```

新增参数通过两种方式提供：

1. `state_anchored/config.py` 中应用默认值；
2. 命令行通过 Hydra 的 `+key=value` 覆盖。

例如：

```bash
python train_state_anchored.py \
  task=walker-walk \
  steps=1000000 \
  exp_name=state-anchored \
  compile=false \
  +sa_state_coef=5.0
```

## 3.2 `state_anchored/config.py`

实现：

```python
def apply_state_anchored_defaults(cfg):
    ...
    return cfg
```

只在字段不存在时写入默认值。

推荐默认值：

```python
DEFAULTS = {
    # representation
    "sa_feature_dim": None,          # None -> cfg.latent_dim
    "sa_feature_hidden_dim": None,   # None -> cfg.enc_dim
    "sa_feature_lr_scale": None,     # None -> cfg.enc_lr_scale
    "sa_feature_simnorm": True,

    # state normalization
    "sa_norm_eps": 1e-5,
    "sa_norm_min_std": 1e-3,
    "sa_norm_clip": 10.0,
    "sa_norm_freeze_updates": 100_000,

    # dynamics
    "sa_predict_delta": True,
    "sa_dynamics_hidden_dim": None,  # None -> cfg.mlp_dim
    "sa_dynamics_layers": 2,
    "sa_zero_init_dynamics_output": True,

    # state consistency objective
    "sa_state_loss": "smooth_l1",    # support "mse" and "smooth_l1"
    "sa_state_loss_beta": 1.0,
    "sa_state_coef": 5.0,

    # diagnostics
    "sa_log_diagnostics": True,
}
```

解析完成后设置：

```python
if cfg.sa_feature_dim is None:
    cfg.sa_feature_dim = cfg.latent_dim

if cfg.sa_feature_hidden_dim is None:
    cfg.sa_feature_hidden_dim = cfg.enc_dim

if cfg.sa_feature_lr_scale is None:
    cfg.sa_feature_lr_scale = cfg.enc_lr_scale

if cfg.sa_dynamics_hidden_dim is None:
    cfg.sa_dynamics_hidden_dim = cfg.mlp_dim
```

校验：

```python
assert cfg.obs == "state"
assert not cfg.multitask
assert cfg.sa_dynamics_layers >= 1
assert cfg.sa_state_loss in {"mse", "smooth_l1"}
assert cfg.sa_norm_min_std > 0
```

---

# 4. `state_anchored/layers.py`

## 4.1 `RunningStateNorm`

实现 `torch.nn.Module`：

```python
class RunningStateNorm(nn.Module):
    def __init__(
        self,
        state_dim: int,
        eps: float = 1e-5,
        min_std: float = 1e-3,
        clip: float | None = 10.0,
    ):
        ...
```

注册 buffers：

```python
count: scalar float64
mean:  [state_dim] float64
m2:    [state_dim] float64
```

要求：

- 使用 batch Welford 合并公式；
- `update(states)` 接收任意 leading shape，最后一维为 `state_dim`；
- update 前 reshape 为 `[-1, state_dim]`；
- 只允许传入真实 replay states；
- 必须 `@torch.no_grad()`；
- 不对 predicted states 更新；
- buffers 自动进入 checkpoint；
- `normalize()` 和 `denormalize()` 保持输入 dtype/device；
- 当 `count < 2` 时返回：
  - mean = 当前 mean；
  - std = ones；
- std clamp 到 `min_std`；
- normalize 后可 clamp 至 `[-clip, clip]`。

接口：

```python
@property
def var(self) -> torch.Tensor: ...

@property
def std(self) -> torch.Tensor: ...

def normalize(self, state: torch.Tensor) -> torch.Tensor: ...

def denormalize(self, normalized: torch.Tensor) -> torch.Tensor: ...

def scale_delta(self, normalized_delta: torch.Tensor) -> torch.Tensor:
    # normalized delta -> raw state delta
    return normalized_delta * std
```

重要：

```text
normalize(state) 可以 clip；
scale_delta(delta) 不可以 clip。
```

## 4.2 `StateFeatureEncoder`

使用官方 `common.layers.mlp` 和 `common.layers.SimNorm`。

结构：

```text
normalized state x
-> hidden width cfg.sa_feature_hidden_dim
-> feature width cfg.sa_feature_dim
-> SimNorm(group=cfg.simnorm_dim)
```

建议直接：

```python
self.net = common_layers.mlp(
    state_dim,
    max(cfg.num_enc_layers - 1, 1) * [cfg.sa_feature_hidden_dim],
    cfg.sa_feature_dim,
    act=common_layers.SimNorm(cfg)
        if cfg.sa_feature_simnorm
        else nn.Identity(),
)
```

注意：

- 它是辅助 feature encoder；
- `WorldModel.encode()` 不返回这个 feature；
- 它只由 `features(state)` 内部调用；
- feature 维度默认等于官方 `cfg.latent_dim`，即 model size 5 时为 512。

## 4.3 `StateDeltaDynamics`

第一版优先保持与官方 dynamics 相近的容量和风格。

输入维度：

\[
d_s+d_u+d_a.
\]

输出维度：

\[
d_s.
\]

结构：

```python
common_layers.mlp(
    state_dim + feature_dim + action_dim,
    cfg.sa_dynamics_layers * [cfg.sa_dynamics_hidden_dim],
    state_dim,
)
```

默认即：

```text
[state normalized, feature, action]
-> 512
-> 512
-> normalized state delta
```

不要给输出使用 SimNorm、softmax、tanh 或 LayerNorm。

保存最后一层引用：

```python
@property
def output_layer(self) -> nn.Linear:
    ...
```

初始化顺序：

1. 对整个模型执行官方 `init.weight_init`；
2. 若 `sa_zero_init_dynamics_output=True`：
   - dynamics 最后一层 weight 置零；
   - dynamics 最后一层 bias 置零。

这会使初始模型近似：

\[
\hat s_{t+1}=s_t.
\]

---

# 5. `state_anchored/world_model.py`

实现：

```python
class StateAnchoredWorldModel(nn.Module):
```

接口必须兼容官方 `common.world_model.WorldModel`，从而直接复用官方 planner 和 policy update。

## 5.1 初始化

强制：

```python
assert cfg.obs == "state"
assert not cfg.multitask
```

状态维度：

```python
state_dim = cfg.obs_shape["state"][0]
```

创建：

```python
self.state_norm = RunningStateNorm(...)
self._encoder = StateFeatureEncoder(...)
self._dynamics = StateDeltaDynamics(...)
```

这里保留 `_encoder` 名称是为了：

- 与官方 optimizer 分组保持一致；
- feature encoder 使用 `enc_lr_scale`；
- 方便打印和 checkpoint。

但必须明确：

```text
_encoder 不承担 carrier encoding。
```

Anchored feature dimension：

```python
head_dim = state_dim + cfg.sa_feature_dim
```

创建官方风格 heads：

```python
self._reward = common_layers.mlp(
    head_dim + cfg.action_dim,
    2 * [cfg.mlp_dim],
    max(cfg.num_bins, 1),
)

self._termination = (
    common_layers.mlp(
        head_dim,
        2 * [cfg.mlp_dim],
        1,
    )
    if cfg.episodic
    else None
)

self._pi = common_layers.mlp(
    head_dim,
    2 * [cfg.mlp_dim],
    2 * cfg.action_dim,
)

self._Qs = common_layers.Ensemble([
    common_layers.mlp(
        head_dim + cfg.action_dim,
        2 * [cfg.mlp_dim],
        max(cfg.num_bins, 1),
        dropout=cfg.dropout,
    )
    for _ in range(cfg.num_q)
])
```

复用官方：

- `TensorDictParams`
- detach Q；
- target Q；
- `soft_update_target_Q()`；
- Gaussian policy sampling；
- two-hot inverse；
- 随机取两个 Q；
- `return_type` 行为；
- target Q eval mode；
- `log_std_min` 和 `log_std_dif`。

最安全做法：从官方 `WorldModel` 复制与 Q target 管理相关的少量代码到新文件，但不得修改官方文件。

初始化：

```python
self.apply(init.weight_init)
init.zero_([
    self._reward[-1].weight,
    self._Qs.params["2", "weight"],
])
```

随后再 zero-init dynamics output。

注意：如果 Q 网络层数仍为官方三层序列，其输出层参数 key 应与官方一致；必须写测试确认。

## 5.2 核心接口

### `encode`

```python
def encode(self, obs, task=None):
    assert task is None
    return obs
```

必须返回 raw state，不能返回 normalized state 或 auxiliary feature。

这样：

- planner carrier 是 raw state；
- checkpoint/evaluation 语义清晰；
- residual update 在 raw state 上执行；
- normalizer 可变化而 carrier 不变化。

### `normalize_state`

```python
def normalize_state(self, state):
    return self.state_norm.normalize(state)
```

### `features`

```python
def features(self, state):
    x = self.state_norm.normalize(state)
    u = self._encoder(x)
    return torch.cat([x, u], dim=-1)
```

必须支持：

- `[B, D]`
- `[T, B, D]`
- `[num_samples, D]`

禁止缓存 feature 跨 model step。每次调用都重新计算。

### `next`

```python
def next(self, state, action, task=None):
    assert task is None
    anchored = self.features(state)
    delta_x = self._dynamics(torch.cat([anchored, action], dim=-1))
    raw_delta = self.state_norm.scale_delta(delta_x)
    return state + raw_delta
```

如果 `sa_predict_delta=False`，可选支持直接预测 normalized next state：

```python
next_x = self._dynamics(...)
return self.state_norm.denormalize(next_x)
```

但默认必须为 delta prediction。

### `reward`

```python
def reward(self, state, action, task=None):
    h = self.features(state)
    return self._reward(torch.cat([h, action], dim=-1))
```

### `termination`

```python
def termination(self, state, task=None, unnormalized=False):
    h = self.features(state)
    logits = self._termination(h)
    return logits if unnormalized else torch.sigmoid(logits)
```

### `pi`

完全复用官方 Gaussian policy 逻辑，仅将输入从 latent 改为：

```python
h = self.features(state)
```

### `Q`

完全复用官方 Q 逻辑，仅将输入从 latent 改为：

```python
h = self.features(state)
q_input = torch.cat([h, action], dim=-1)
```

## 5.3 `__repr__`

打印：

```text
State-Anchored TD-MPC2 World Model
State normalizer: ...
Feature encoder: ...
State delta dynamics: ...
Reward: ...
Termination: ...
Policy prior: ...
Q-functions: ...
Carrier dimension: d_s
Auxiliary feature dimension: d_u
Learnable parameters: ...
```

---

# 6. `state_anchored/agent.py`

实现：

```python
class StateAnchoredTDMPC2(TDMPC2):
```

但不要调用 `super().__init__(cfg)`，因为官方构造函数会创建官方 `WorldModel`。

## 6.1 自定义 `__init__`

按照官方 `TDMPC2.__init__` 保持相同行为，只替换：

```python
self.model = StateAnchoredWorldModel(cfg).to(self.device)
```

optimizer 分组：

```python
self.optim = torch.optim.Adam(
    [
        {
            "params": self.model._encoder.parameters(),
            "lr": cfg.lr * cfg.sa_feature_lr_scale,
        },
        {"params": self.model._dynamics.parameters()},
        {"params": self.model._reward.parameters()},
        {
            "params":
                self.model._termination.parameters()
                if cfg.episodic
                else []
        },
        {"params": self.model._Qs.parameters()},
    ],
    lr=cfg.lr,
    capturable=True,
)
```

policy optimizer 保持官方：

```python
self.pi_optim = torch.optim.Adam(
    self.model._pi.parameters(),
    lr=cfg.lr,
    eps=1e-5,
    capturable=True,
)
```

其余必须与官方一致：

- `RunningScale`
- discount；
- `_prev_mean`；
- action-dimension iteration heuristic；
- compile 行为。

新增 buffer：

```python
self.register_buffer(
    "_sa_update_count",
    torch.tensor(0, dtype=torch.long, device=self.device),
)
```

或者使用普通 Python int；如果要 checkpoint 保存，推荐 buffer。

## 6.2 继承而不重写的方法

以下方法应直接继承官方 `TDMPC2`，除非兼容性问题迫使修改：

- `act`
- `_estimate_value`
- `_plan`
- `update_pi`
- `_td_target`
- `_get_discount`
- `save`
- `load`
- `plan` property

原因是新世界模型保持了完全相同的接口：

```text
encode / next / reward / termination / pi / Q
```

这些官方函数变量名仍可能叫 `z`，但实际 tensor 语义是 raw state；不要为了重命名复制整段官方 planner。

## 6.3 重写 `update`

需要在 compiled `_update` 之外更新 Welford statistics：

```python
def update(self, buffer):
    obs, action, reward, terminated, task = buffer.sample()

    if self._sa_update_count < self.cfg.sa_norm_freeze_updates:
        self.model.state_norm.update(obs)

    self._sa_update_count += 1

    kwargs = {}
    if task is not None:
        kwargs["task"] = task

    torch.compiler.cudagraph_mark_step_begin()
    return self._update(
        obs,
        action,
        reward,
        terminated,
        **kwargs,
    )
```

要求：

- `obs` 全部来自 replay，是实际 observation；
- normalizer update 在 `_update` 外；
- update 不参与 autograd；
- `_update` 可以继续 `torch.compile`；
- freeze 后 statistics 不再变化；
- task 第一版必须为 None。

## 6.4 重写 `_update`

尽量逐行保持官方 `_update` 结构。

### 6.4.1 TD target

保持官方算法：

```python
with torch.no_grad():
    next_state = self.model.encode(obs[1:], task)
    td_targets = self._td_target(
        next_state,
        reward,
        terminated,
        task,
    )
```

仍然：

\[
y_t=r_t+\gamma\bar Q(s_{t+1},\pi(s_{t+1})).
\]

禁止在本阶段替换成 MPC action。

### 6.4.2 State rollout

carrier tensor：

```python
state_dim = self.cfg.obs_shape["state"][0]

states = torch.empty(
    self.cfg.horizon + 1,
    self.cfg.batch_size,
    state_dim,
    device=self.device,
)
```

起点：

```python
state = self.model.encode(obs[0], task)
states[0] = state
```

递归：

```python
state_loss = 0.0

for t, (_action, target_next_state) in enumerate(
    zip(action.unbind(0), next_state.unbind(0))
):
    state = self.model.next(state, _action, task)

    pred_x = self.model.normalize_state(state)
    target_x = self.model.normalize_state(target_next_state)

    if cfg.sa_state_loss == "mse":
        step_loss = F.mse_loss(pred_x, target_x)
    else:
        step_loss = F.smooth_l1_loss(
            pred_x,
            target_x,
            beta=cfg.sa_state_loss_beta,
        )

    state_loss = state_loss + step_loss * cfg.rho ** t
    states[t + 1] = state
```

第一版继续使用官方同一个 `rho`，不要新增 state_rho，以减少变量。

### 6.4.3 Reward、Q 和 termination losses

完全保持官方逻辑，只将 `zs` 替换为 `states`：

```python
_model_states = states[:-1]

qs = self.model.Q(
    _model_states,
    action,
    task,
    return_type="all",
)

reward_preds = self.model.reward(
    _model_states,
    action,
    task,
)
```

termination 也保持。

### 6.4.4 总 loss

保持官方 reward/value/termination coefficient。

仅 consistency coefficient 换为独立字段：

```python
total_loss = (
    cfg.sa_state_coef * state_loss
    + cfg.reward_coef * reward_loss
    + cfg.termination_coef * termination_loss
    + cfg.value_coef * value_loss
)
```

不要复用官方 `consistency_coef=20`，因为 SimNorm latent MSE 与 normalized state loss 的尺度不同。

### 6.4.5 Policy update

保持：

```python
pi_info = self.update_pi(states.detach(), task)
```

policy 在 imagined raw state rollout 上训练，但内部会重新构造 anchored features。

### 6.4.6 日志兼容

继续返回官方 key：

```python
"consistency_loss": state_loss
```

同时新增：

```python
"state_loss": state_loss
```

推荐 diagnostics：

```python
"state_norm_mean_abs"
"state_norm_std_mean"
"state_norm_std_min"
"state_norm_std_max"
"imagined_state_abs_max"
```

不要把大 tensor 放进 logger。

---

# 7. `train_state_anchored.py`

尽量复制官方 `train.py` 的入口结构，但使用新增 agent。

流程：

```python
cfg = parse_cfg(cfg)
cfg = apply_state_anchored_defaults(cfg)
```

然后：

```python
trainer = OnlineTrainer(
    cfg=cfg,
    env=make_env(cfg),
    agent=StateAnchoredTDMPC2(cfg),
    buffer=Buffer(cfg),
    logger=Logger(cfg),
)
```

必须拒绝 offline/multitask：

```python
if cfg.multitask:
    raise NotImplementedError(...)
```

运行示例：

```bash
python train_state_anchored.py \
  task=walker-walk \
  steps=1000000 \
  exp_name=state-anchored \
  compile=false
```

Dog：

```bash
python train_state_anchored.py \
  task=dog-run \
  steps=7000000 \
  exp_name=state-anchored-dog \
  compile=true
```

调试阶段先 `compile=false`，通过 smoke test 后再开 `compile=true`。

---

# 8. `evaluate_state_anchored.py`

复制官方 evaluation 流程，但实例化：

```python
StateAnchoredTDMPC2(cfg)
```

并在 parse 后应用 State-Anchored defaults。

示例：

```bash
python evaluate_state_anchored.py \
  task=walker-walk \
  checkpoint=/path/to/checkpoint.pt \
  eval_episodes=10
```

---

# 9. Checkpoint 约束

新 checkpoint 与官方 latent checkpoint 不兼容。

要求：

- 新 agent 的 `save()` 可继承官方方法；
- state normalizer buffers 必须在 `model.state_dict()` 中；
- 加载新 checkpoint 后：
  - count、mean、m2 完全恢复；
  - target Q 正常恢复；
  - evaluation 与保存前输出一致；
- 加载官方 latent checkpoint 时给出清晰 shape mismatch 错误即可，不需要兼容。

建议 checkpoint 文件名或目录明确包含：

```text
state-anchored
```

---

# 10. 单元测试

## 10.1 `test_state_anchored_norm.py`

测试：

1. 已知数据的 mean 与 unbiased variance；
2. 分两个 batch update 与一次性统计结果一致；
3. normalize + denormalize 近似恢复；
4. std 不低于 `min_std`；
5. predicted state 不会被自动加入统计；
6. state_dict save/load 后统计一致；
7. 输入 `[T,B,D]` 可以 update。

容差：

```python
atol=1e-6
rtol=1e-5
```

## 10.2 `test_state_anchored_model.py`

构造最小 cfg，测试：

- `encode(s) == s`；
- `features(s).shape == [..., state_dim + feature_dim]`；
- `next(s,a).shape == s.shape`；
- zero-init 时 `next(s,a)` 初始近似 `s`；
- reward output shape；
- pi output shape；
- Q output shape；
- `[T,B,D]` 输入；
- gradients 能从 reward/Q/state loss 传到：
  - feature encoder；
  - dynamics；
- `next()` 每一步重新调用 encoder；
- 所有输出 finite。

## 10.3 `test_state_anchored_smoke.py`

建议不跑完整环境，仅：

1. 创建 fake Buffer batch；
2. `agent.update()` 一次；
3. 检查所有 metric finite；
4. 检查参数发生更新；
5. 检查 target Q soft update；
6. `compile=false` 必须通过；
7. 若 CI/GPU 允许，再测试 `compile=true`。

---

# 11. 工程验收标准

## 11.1 静态验收

- 官方文件无 diff；
- 新文件通过 `python -m compileall`；
- import 无循环依赖；
- lint 无明显错误；
- type/shape 错误信息清晰。

## 11.2 行为验收

在 `walker-walk` 上：

```bash
python train_state_anchored.py \
  task=walker-walk \
  steps=20000 \
  eval_freq=10000 \
  save_video=false \
  enable_wandb=false \
  compile=false \
  exp_name=sa-smoke
```

要求：

- 启动成功；
- seed pretraining 成功；
- loss 无 NaN/Inf；
- planner 成功执行；
- checkpoint 成功保存；
- evaluation 成功加载；
- imagined state 不爆炸。

随后运行官方 baseline：

```bash
python train.py \
  task=walker-walk \
  steps=20000 \
  eval_freq=10000 \
  save_video=false \
  enable_wandb=false \
  compile=false \
  exp_name=baseline-smoke
```

要求官方版本仍可正常运行。

---

# 12. 推荐初始超参数

保留官方默认：

```yaml
batch_size: 256
rho: 0.5
lr: 3e-4
grad_clip_norm: 20
tau: 0.01

reward_coef: 0.1
value_coef: 0.1
termination_coef: 1

iterations: 6
num_samples: 512
num_elites: 64
num_pi_trajs: 24
horizon: 3
min_std: 0.05
max_std: 2
temperature: 0.5

num_bins: 101
vmin: -10
vmax: 10
num_q: 5
dropout: 0.01

log_std_min: -10
log_std_max: 2
entropy_coef: 1e-4
```

新增默认：

```yaml
sa_feature_dim: 512          # model_size=5
sa_feature_hidden_dim: 256
sa_feature_lr_scale: 0.3
sa_feature_simnorm: true

sa_norm_eps: 1e-5
sa_norm_min_std: 1e-3
sa_norm_clip: 10.0
sa_norm_freeze_updates: 100000

sa_predict_delta: true
sa_dynamics_hidden_dim: 512
sa_dynamics_layers: 2
sa_zero_init_dynamics_output: true

sa_state_loss: smooth_l1
sa_state_loss_beta: 1.0
sa_state_coef: 5.0
```

第一轮只 sweep：

```text
sa_state_coef ∈ {1, 5, 10}
sa_state_loss ∈ {mse, smooth_l1}
```

其余不调，避免失去公平性。

---

# 13. 研究实验所需日志

新增但不影响训练的评估函数，记录：

## 13.1 多步 state prediction

\[
\operatorname{NRMSE}(k)
=
\sqrt{
\frac{1}{d_s}
\left\|
N(\hat s_{t+k})-N(s_{t+k})
\right\|_2^2
}.
\]

至少：

```text
k = 1, 3
```

后续可测 5、10，但训练 buffer 当前 horizon 默认 3，需要独立 evaluation batch。

## 13.2 数值稳定性

记录：

- raw imagined state max abs；
- normalized imagined state max abs；
- predicted normalized delta mean/max；
- state norm std min/max；
- feature entropy 或 SimNorm max（可选）；
- dynamics gradient norm；
- feature encoder gradient norm。

## 13.3 参数量

分别报告：

- feature encoder；
- dynamics；
- reward；
- policy；
- Q ensemble；
- total。

必须与官方 5M 模型进行对比。

---

# 14. 不允许的实现捷径

禁止：

1. 把 `cfg.latent_dim` 改成 `state_dim` 后直接使用官方 WorldModel；
2. 对 raw state 使用 SimNorm；
3. 让 `encode()` 返回 anchored feature；
4. 递归预测 auxiliary feature；
5. 用 predicted states 更新 Welford statistics；
6. 从第一次实验起使用 known reward；
7. 修改 Q target 为 MPC；
8. 修改官方 buffer 来保存额外字段；
9. 修改官方 trainer；
10. 为了 shape 兼容而 pad state 到 512 维并把它称为 state-space；
11. 把 State-Anchored checkpoint 当作官方 checkpoint 加载；
12. 静默支持 multitask 或 rgb。

---

# 15. 实现顺序

Codex 按以下顺序提交：

## Commit 1：基础模块

- `config.py`
- `RunningStateNorm`
- `StateFeatureEncoder`
- `StateDeltaDynamics`
- normalization tests

## Commit 2：World model

- `StateAnchoredWorldModel`
- compatible interfaces
- model shape/gradient tests

## Commit 3：Agent

- custom constructor
- norm update outside compile
- custom `_update`
- inherited planner/TD target/policy update
- fake-batch update test

## Commit 4：Entrypoints

- train script
- evaluate script
- smoke commands
- README usage section（新增独立文档，不修改官方 README）

## Commit 5：Runtime validation

- walker smoke run
- original baseline smoke run
- compile=true check
- checkpoint save/load check

---

# 16. Codex 最终交付内容

Codex 完成后必须输出：

1. 新增文件列表；
2. 每个新增类的职责；
3. 哪些官方函数被直接复用；
4. 哪些方法因 carrier 改变而重写；
5. 所有测试命令及结果；
6. walker smoke run 结果；
7. `git diff --stat`；
8. 官方路径无 diff 的证明；
9. 当前限制；
10. 下一步建议，但不要自行实现 MPC-Q、known reward 或 uncertainty。

---

# 17. 官方接口依据

本设计依赖当前官方实现的以下事实：

- `TDMPC2._plan()` 和 `_estimate_value()` 仅通过 world model 的  
  `encode / next / reward / pi / Q / termination` 接口运行；
- 官方 `_update()` 需要因 carrier 维度和 consistency target 改变而重写；
- 官方 reward/Q 为 distributional heads；
- 官方 Q 使用 ensemble 与 target Q；
- 官方 policy 为 Gaussian prior；
- 官方 trainer 只依赖 agent 的 `act()`、`update()`、`save()`；
- 官方 replay buffer 返回：
  `obs, action, reward, terminated, task`。

Codex 必须先检查本地 checkout 中这些接口是否仍一致；若本地版本略有差异，应保持本文方法不变并适配实际签名。

官方参考：

- World model:  
  https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py
- Agent:  
  https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py
- Layers:  
  https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
- Config:  
  https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/config.yaml
- Buffer:  
  https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/buffer.py
- Online trainer:  
  https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/trainer/online_trainer.py

---

# 18. 一句话实现原则

```text
保留 TD-MPC2 的 planner、critic、policy、reward learning 和训练循环；
把递归 latent carrier 替换为 raw semantic state；
高维 learned representation 只作为每一步从 state 重新计算的辅助特征。
```
