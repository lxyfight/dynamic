# dynamic · 动态需求下的卡车与无人机协同调度

卡车沿高速公路向交接站补货，并把站内取件运回仓库。驻站无人机完成交接站与客户之间的配送和取件。当一笔新取件出现时，系统需要决定是否接单、交给哪架无人机、利用哪一趟卡车回收。在站内，卡车还可以在卸货后装载一个或多个取件批次，决定等候或离站。

本仓库提供事件模拟器、初始路线生成器、在线调度策略和可复算的实验结果。**当前用于比较的算法是冻结参数的六特征线性价值函数近似（VFA）。** 它在四个新布局的 16 个高需求实例中均超过 Myopic 和有界滚动重优化。另行研究的站内等待策略在独立测试中平均降低了收益，尚未纳入默认算法。

## 快速运行

在仓库根目录使用 Python 虚拟环境，安装依赖并运行一个实例。

```shell
python -m pip install -r requirements.txt
python demo.py
```

默认示例读取冻结模型，比较同一组 20 单（其中 8 笔动态取件）在 Myopic 和六特征 VFA 下的全日净收益，并核对终止状态与账本。要重放结果表中的一个 100 单实例，可在根目录运行 `python demo.py --orders 100 --layout 460 --pattern uniform`。下列测试和实验命令从研究代码目录执行。

```shell
cd code/e1_v04
python -m unittest discover -s tests -v
```

默认示例仅需 NumPy。完整测试套件中的 PI 映射和上界测试还需要 Gurobi 与可用许可证。已验证的环境为 Python 3.11、NumPy 1.24 和 Gurobi 11。可以在根目录通过 `python -m pip install -r requirements-pi.txt` 安装包含求解器的依赖；若 Gurobi 装在另一个环境，可将 `E1_PI_PYTHON` 设为该环境的 Python 可执行文件。

下列命令会启动多进程、执行多个完整运营日。按顺序运行可重新筛选特征、冻结参数，然后在新布局上比较四种策略。结果写入仓库下的 `tmp/`。

```shell
python run_high_demand_screen.py
python run_frozen_reopt_comparison.py
python run_frozen_scalar_reference.py --report
```

要只查看既有结果，可直接打开[高需求逐实例对照](results/high_demand_comparison.csv)、[特征筛选明细](results/feature_screen.csv)、[追加训练检查点](results/iteration_checkpoints.csv)和[等待模型逐日验证](results/waiting_policy.csv)。

完成上述全日特征筛选后，可按以下顺序重建等待实验。这条流程对同一个交接站决策分别回放立即离站和预约等待，直到全日结算，运行时间长于单实例示例。

```shell
python run_wait_counterfactual.py
python run_all_wait_counterfactual.py
python run_wait_increment_learning.py protocol
python run_wait_learning_data.py worlds
python run_wait_learning_data.py all
python run_wait_increment_learning.py freeze
python run_wait_increment_learning.py online
python run_wait_increment_learning.py report
python expand_wait_learning_data.py
python run_expanded_wait_learning.py freeze
python run_expanded_wait_learning.py online
python run_expanded_wait_learning.py report
```

## 一个运营日怎样求解

1. **生成需求。** 给定全网订单总数和动态度，先确定配送单与取件单数量，再生成客户位置、重量和可服务交接站。客户位置可服从均匀分布或双簇分布。为控制实验规模，给定取件总数后，按分段非齐次泊松过程的条件分布抽取到达时刻。策略可读取静态配送需求、已披露取件和历史到达强度。当天实际取件总数及未来取件信息只由模拟环境持有。
2. **生成初始方案 I0。** 只使用开工前已知的配送单，选择可行交接站，按容量形成卡车补货批次，再安排无人机服务。初始方案经过有界改进和物理回放，并作为同实例各策略的共同起点。配送需求和卡车补货量取整数，模型允许多辆卡车向同一笔配送需求补货。每笔取件须整单接收或拒绝。
3. **处理事件并选择动作。** 新请求到达时，联合生成接拒、航次调整和回收车趟安排。交接站作业完成或等待复核到期且有实际选择时，比较继续装货、等待和离站。为已披露取件预定装货时间后，若计划仍有效，到点直接执行。卡车已开始的行驶及车辆已开始的服务、装卸操作不可中断；已起飞航次的访问顺序和返站地点固定。
4. **结算。** 先在已知订单上回放候选计划，校验容量、作业顺序及履约承诺。Myopic 选择已知计划净收益最高的动作。六特征 VFA 在该收益上加入线性估计的后续价值。一天结束后，由账本复核履约收入、卡车趟费与时间费、无人机架次费与飞行费、迟到和拒单费用。下列高需求实验的无人机架次费为 25 元／架次。

## 特征和对照算法

全日 VFA 从 12 项候选特征出发，按运营含义分为 5 组，对全部 32 种组合作同样本筛选。先比较留出布局上的拟合质量，再比较入围策略执行一整天的净收益。冻结的六项为**新增回收车趟的预期取件覆盖量、当前时刻、预计剩余取件数、新增车趟数、后续卡车行驶时间、待执行无人机架次数**。其中“当前时刻”和“预计剩余取件数”在同一决策点对所有动作取值相同；它们用于解释不同状态的价值水平。冻结的标准化参数及系数见[六特征模型](results/frozen_six.json)。

| 策略 | 对候选动作的处理 |
| --- | --- |
| Myopic | 对当前已知订单生成可执行续接方案，按已知计划净收益选择。 |
| 滚动重优化（RH） | 在共同候选中选前四个，额外进行最多八次局部邻域回放，择优执行。搜索范围受预算限制。 |
| 单特征 VFA | 给新增回收车趟覆盖的预期取件机会估值，使用固定系数 28。 |
| 六特征 VFA | 在相同的候选机制上，用离线拟合并冻结的线性模型估计后续收益。 |

## 已完成的实验

两城市高需求对照固定取件占比为 40%。四个新布局均覆盖 100 单和 150 单，并分别采用均匀、双簇客户分布。每种规模含 8 个配对实例，各策略使用相同的当天订单和 I0。表中为平均全日净收益，单位为元。

| 订单总数 | Myopic | 滚动重优化 | 单特征 VFA | 冻结六特征 VFA |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 135.69 | 135.70 | 839.42 | **1,001.32** |
| 150 | 322.70 | 323.00 | 1,593.66 | **1,725.80** |

六特征 VFA 在 16 个实例中相对单特征 VFA 为 14 胜、2 负，平均增益 147.02 元／日。六特征 VFA 与 RH 平均分别接下 45.31 和 2.69 笔取件。两者在这批实例中的主动等待次数均为零，收益差与接单及回收运力安排有关。

## 遇到的问题

- **继续训练会改变接单行为。** 每个 episode 是一次完整运营。累计训练 8、12、16、20 个 episode 时，同一批验证实例上的平均净收益依次为 1,461.11、287.05、1,416.98、1,398.87 元。最后一次更新在另外 4 个实例上比原冻结版平均少获 49.35 元，因此继续使用原参数，训练收敛性仍需检验。
- **单次等待与连续续等的收益不同。** 开发阶段对 74 个同状态节点分别执行“立即离站”和“预约 5 分钟后复核”，保持两条轨迹的后续订单相同。由这些配对拟合的扩充模型，在另外 8 个运营实例上选择等待 25 次，较六特征基线平均少获 34.27 元。其中一个 100 单双簇实例，首次单独等待仅损失 0.65 元；整日允许反复决策时，同一辆卡车累计选择等待 22 次，少接 5 单，净收益降低 272.25 元。后续研究需要在训练中覆盖整段驻站期间的多次复核。
- **有利等待样本稀少，外推范围有限。** 留出验证的 7 个节点没有等待获益样本，独立测试的 16 个单节点配对中仅 1 个获益。全日算法的主对照覆盖四个独立布局；更多布局、不同动态度和与完美信息上界的系统比较仍待完成。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| [demo.py](demo.py) | 单实例运行入口 |
| [data/fixtures/](data/fixtures/) | 初始数据、场景规格与回归用例 |
| [code/e1_v04/e1_sim/](code/e1_v04/e1_sim/) | 事件内核、信息边界、状态更新、可行性检查、初始方案、候选动作及账本 |
| [high_demand_features.py](code/e1_v04/high_demand_features.py) | 全日候选特征、冻结策略打分与完整运营日求解 |
| [run_high_demand_screen.py](code/e1_v04/run_high_demand_screen.py) | 按训练、特征验证、策略验证和独立确认四个阶段筛选模型 |
| [run_frozen_reopt_comparison.py](code/e1_v04/run_frozen_reopt_comparison.py) | 新布局上的 VFA、Myopic、RH 配对比较 |
| [run_wait_learning_data.py](code/e1_v04/run_wait_learning_data.py) 和 [run_expanded_wait_learning.py](code/e1_v04/run_expanded_wait_learning.py) | 生成等待反事实训练样本，评估扩充模型 |
| [tests/](code/e1_v04/tests/) | 整数货量、履约承诺、不可中断作业、事件触发、费用及策略接口的回归检查 |
