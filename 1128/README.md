# 1128 代码与第一轮特征筛选归档

本目录保存 2026-11-28 的代码、开发实验及四组 VFA 比较结果。独立测试集未生成、未运行。

- [完整中文报告](results/feature_screen_20261128/REPORT.txt)
- [实测汇总](results/feature_screen_20261128/summary.json)
- [逐日配对结果](results/feature_screen_20261128/paired_validation.json)
- [数据划分与保留测试编号](results/feature_screen_20261128/campaign_manifest.json)
- [最终完整性核验](results/feature_screen_20261128/audit/final_integrity.json)
- [复现脚本](results/feature_screen_20261128/REPRODUCE.ps1)

共140次全天回放。8个验证场景的平均净利润：none −596.180498元，existing_chain_opportunity −596.211540元，加入future_pickup_lateness后 −596.211540元，full −784.504247元。当前暂优none；这不是独立测试结论。

从仓库根目录运行（Python须装有NumPy和可用的Gurobi）：

```powershell
cd 1128/code/e1_v04
python verify_screen_rules.py --output ../../results/feature_screen_rerun/audit/rules.json
python run_feature_screen.py --stage all --jobs 4 --output ../../results/feature_screen_rerun
python analyze_feature_screen.py --output ../../results/feature_screen_rerun
```

使用未存在的新输出目录；各组权重、两轮checkpoint及逐日轨迹均位于 `results/feature_screen_20261128/training/`。原始实测环境和本机命令保存在报告中。

`ARCHIVE_MANIFEST.json`记录复制来源与原始文件校验值；本README在复制后增加了归档导航，.gitattributes设置为保留原始字节以保证实验哈希跨平台一致。以下保留原项目说明。

---

# dynamic · 动态需求下的卡车与无人机协同调度

固定回路卡车沿高速向双侧交接站补给药品、回收样品，驻站无人机服务客户。站内实时库存连接干线运输与末端飞行。系统联合决定动态取件接拒、无人机任务、卡车回收及停站动作，目标为全日期望净利润最大。

**研究设置采用卡车在线选择复核间隔、同一 VFA 评价联合车机方案，以及按分钟生成随机取件批量。** 可在同一批运营日将自适应策略与验证集选出的最佳固定间隔比较。[模型、信息边界与目标函数](model.md)

## 快速运行

Python 3.11 或更高版本，在仓库根目录运行：

```shell
python -m pip install -r requirements.txt
python demo.py --orders 20 --share 0.4 --pattern uniform --temporal peak --arrival-mode p2 --review-mode adaptive --adaptive-intervals 2 5 10
```

未提供权重时，程序按已知义务的净收益运行 Unified-Myopic。每次运行检查完整履约与独立账本。

提供新版训练模型后，在同一运营日比较 Unified-Myopic 与 Unified-VFA：

```shell
python demo.py --orders 20 --pattern dual --temporal flat --model tmp/adaptive_p2/selected_adaptive_model.json
```

不带间隔选项的演示默认使用在线自适应复核和 P2 式需求。训练命令 `train` 默认比较两类策略；此前只给 `--h-values` 的训练命令和只给 `--h` 的演示沿用原固定间隔和条件到单设置。加载模型时按文件恢复其复核规则、需求模式和搜索设置。`python demo.py --legacy` 用于旧六特征算法。

## 本版运营参数

每个 pickup 在释放时获知自己的返仓截止。相对时限从 240、300、360 分钟中逐单独立等概率抽取。迟到从该单的截止开始，按每单每分钟 0.3 元累计至卡车返仓。未来未揭示订单的迟到特征按三种时限的概率求期望。

卡车和无人机在交接站的装货、卸货均按重量计时，每千克 0.025 分钟。客户处每单服务 1 分钟。期初配送车可同时在第 0 分钟发车，运营中按需增开的回收车最迟在第 660 分钟发车。

新版场景须重新拟合 VFA 系数。首轮只使用训练和验证布局筛选特征，保留独立测试布局，直到特征和比较策略确定后再评估。新窗口可使用 [项目交接提示词](NEXT_SESSION.md) 接续实验。

## 随机取件与车辆复核

- `--arrival-mode p2` 借鉴 P2 式（14）的结构，先抽每分钟的到单数，再为各单抽客户位置和货重。这里采用有限支持的二项分布，各时间槽独立，参数经校准使全天期望取件数为 `round(N*d)`。期初配送数为 `N-round(N*d)`，每天实际取件数和动态占比会变。概率参数公开，未来请求的具体实现仅仿真器可见。[P2 正文](https://doi.org/10.1287/trsc.2017.0773)
- `--arrival-mode fixed_count` 固定运营日取件总数，并随机抽取各单到达分钟。
- 空间 `uniform` 为沿线区域均匀分布，`dual` 为两个截断高斯热点。空间对照共享相同释放时间、重量和返仓时限随机流，各自按实际 delivery 位置生成 I0。
- 时间 `peak` 在第 1–150、151–350、351–590 分钟的每分钟期望到达量比为 1∶3∶2；`flat` 的每分钟期望相同。改变 P2 时间分布也可能改变同一个模拟日实现的取件数量。
- 几何服务时滞来自独立的合成历史。固定与自适应复核共用同一组运营日需求和初始配送计划。

主动等待时，每辆符合条件的车选自己的下一次复核。例如 09:03 甲车选 2 分钟、乙车选 10 分钟，两车分别于 09:05 和 09:13 复核。期间有关的新单或作业事件可提前触发决策。若继续尚未到期的等待，保留原到期时间；到期后续等可改选间隔。已知取件照原有预约机制装车。

固定基准每次新设计时器都使用同一间隔。比较模式在相同训练和验证日试验各个固定间隔，并独立训练自适应策略。两边共同采用 5 分钟近端特征窗口；车辆选择的 2、5、10 分钟通过具体后续作业时刻影响联合方案的估值。

## 事件驱动的联合决策

1. 仅用期初 delivery 形成共同初始分配与车机计划。相应货物在本站卸货完成并齐备后才可由无人机发运。
2. 订单到达、装卸完成、无人机卸机就绪、等待到期等异步事件更新状态；存在实际选择时调用策略，同一时刻合并事件后决策。
3. 构造接拒、未起飞航次插入/新航次、已有/新增车趟回收、各车离站/预约等待/主动等待/续等的完整候选。多个卡车可以同时等待。
4. 完整回放检查库存、逐段载重、续航、delivery 期限、执行锁定及客户—站—卡车—仓库完整回收链。所有保留的基础计划先扩展等待组合，再由同一个 `J_known + W_theta` 排名，不先固定一个接单计划后单独评价等待。
5. 执行至下一事件，全日结束后审计履约、收入、车机运营费用、拒单及 pickup 返仓迟到费用。

已排定但未起飞的航次允许可行插单，包括尊重已锁货物和进行中操作的装载阶段。实际起飞后当前模型锁定整个航次的访问序列及返站点。程序采用有限候选预算和邻域搜索，日志记录截断，不宣称穷举全部连续时间方案。

## 统一 VFA 与开发训练

六组、七维候选指标包括按共同预测窗口划分的近期与较远回收机会、新增车趟覆盖增量、无人机拥挤、回收容量、未来取件迟到暴露和既有配送期限余量。特征只使用公开预测和已知计划回放。城市级服务时滞用于近似机会量，实际接单仍须验证车机回收链。

训练标签为 `G - J_known`，两项使用相同全日账本。开发入口采用各运营日等权、无截距的缩放岭回归，迭代仿真并探索候选。模型保存特征、缩放、权重、复核规则和需求模式。以下是以固定间隔运行固定间隔与固定到单数对照的命令（采用本版运营参数）。

```shell
cd code/e1_v04
python run_unified_vfa.py train --h-values 5 10 --train-layouts 710 711 --validation-layouts 810 811 --orders 20 --patterns uniform dual --temporal peak flat --iterations 2 --output ../../tmp/unified_vfa_development
```

输出协议、各 h 的训练/验证记录与 `selected_model.json`。训练和验证布局必须隔离，输出目录必须为空。上述配置用于开发，正式特征消融、多训练种子及独立测试仍须按实验安排完成，不能把小样本验证成绩当成论文最终结论。

### 分规模筛选特征并用 Gurobi 搜索在线动作

先指定订单规模、候选特征组、固定间隔和各数据集的布局。每个 `(特征组合, h)` 从零残差单独训练，共用相同训练日、验证日和搜索预算。验证日的平均全天净利润决定保留哪个组合，收益完全相同则优先保留维数较小的组合。给出 `--test-layouts` 时，程序先写出冻结模型，再在未使用的布局上与相同 h、相同搜索配置下的 Myopic 策略逐日配对。`full` 使用七个特征，`none` 将未来增量设为零；一个组合内的多个组用逗号连接。

以下为验证程序能否运行的小样本配置。在仓库根目录安装在线求解依赖，确保当前 Python 能创建 Gurobi 模型，再运行：

```sh
python -m pip install -r requirements-online-mip.txt
cd code/e1_v04
python run_unified_vfa.py train --review-mode compare --adaptive-intervals 2 5 10 --arrival-mode p2 --action-solver gurobi --mip-pattern-limit 8 \
  --feature-sets none existing_chain_opportunity existing_chain_opportunity,added_trip_reliance full \
  --h-values 2 5 10 --orders 4 6 --validation-orders 6 --test-orders 8 \
  --train-layouts 710 --validation-layouts 810 --test-layouts 910 \
  --days 1 --share 0.4 --patterns uniform --temporal flat --iterations 1 \
  --initial-budget 0 --candidate-budget 4 --enhancement-budget 0 --output ../../tmp/adaptive_p2
```

`results.json` 按组合和轮次记录验证利润，以及对本策略模拟残差的等日权重 RMSE 和 R²。`protocol.json` 保存实例与配置。在 `compare` 实验中，`selected_fixed_model.json` 和 `selected_adaptive_model.json` 分别保存两类策略的验证优胜模型；`selected_model.json` 保存全体验证优胜模型。主要结果是 `heldout_test.json` 中的 `mean_adaptive_minus_fixed`，以及各运营日的 `adaptive_minus_fixed` 和决策开销。正数表示该样本上在线选时长提高净利润。正式实验可分别指定训练、验证和测试的名义订单规模，并按布局和运营日配对分析。

`--action-solver bounded` 使用原有候选搜索。`gurobi` 模式基于当前可见订单、既有计划和已生成的设备操作序列建立整数模型，搜索设备计划重组和多车等待组合。完整物理回放检查库存、约束和已作出的承诺，并计算各可行组合的 `J_known + W` 评分。`--mip-pattern-limit` 分别控制每个决策事件的新增服务组合和新增等待组合，已有可行基础方案参加比较。对微型实例设为 `all` 可查完已表示的组合。接单、航次等基础方案的生成仍受原预算约束。检查运行记录中的搜索截断信息，再解释策略差距。

## 验证与实验安排

```shell
cd code/e1_v04
python -m unittest discover -s tests -v
```

默认邻域搜索只需 NumPy。在线整数搜索在运行策略的 Python 中加载 Gurobi，依赖见 `requirements-online-mip.txt`。历史完美信息映射/上界测试使用 `requirements-pi.txt`，环境变量 `E1_PI_PYTHON` 可指定其独立解释器。

2026-11-26 完整回归 **301 项通过**；训练、验证选 h 和模型加载流程已完成小规模检查，详见 [验证记录](results/unified_vfa_20261126/VALIDATION.md)。这些检查不等同于正式效果实验。

主实验比较在线选择间隔与最佳固定间隔的全天净利润；辅助记录实际取件数与占比、接单率、返仓时效、各复核间隔的选择次数、候选搜索截断及耗时。

历史六特征模型及成绩见 [LEGACY_RESULTS.md](LEGACY_RESULTS.md)。此前自适应 h 的 32 日、313 条回放先导见 [SYSTEM_WAIT_PILOT.md](SYSTEM_WAIT_PILOT.md) 与 [报告](results/system_wait_pilot_20261125/REPORT.md)。本次比较需要在新动作空间和新到单分布下重新训练与评估。

## 代码导航

| 文件 | 用途 |
| --- | --- |
| [model.md](model.md) | 修订模型、SMDP、约束、目标和实验协议 |
| [demo.py](demo.py) | 新版运行入口，`--legacy` 复现旧示例 |
| [unified_vfa_policy.py](code/e1_v04/unified_vfa_policy.py) | 在线逐车选择复核间隔、固定基准、多车联合评分 |
| [unified_vfa_features.py](code/e1_v04/unified_vfa_features.py) | 六组七维候选后决策特征 |
| [run_unified_vfa.py](code/e1_v04/run_unified_vfa.py) | 分别训练和选择自适应与固定策略、独立配对测试 |
| [uniform_wait_worlds.py](code/e1_v04/uniform_wait_worlds.py) | P2 式随机到单、固定总数对照和独立历史样本 |
| [e1_sim/](code/e1_v04/e1_sim/) | 事件、库存、可行性、依赖重排与账本 |
| [tests/](code/e1_v04/tests/) | 履约、锁定、计时、多车组合、特征与信息边界检查 |
