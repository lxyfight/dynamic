# 固定 h 统一 VFA：实现验证记录

日期：2026-11-26。本记录确认实现与开发流程可运行，不构成算法利润优势、最优 h 或特征有效性的实验证据。

## 完整回归

在 `code/e1_v04` 下，使用安装了 Gurobi 的 Python 运行：

```shell
python -m unittest discover -s tests -v
```

结果：**301 项测试，122.366 秒，零失败、零错误**，包含完美信息模型相关检查。完整输出见 [full_tests.txt](full_tests.txt)。仅有 NumPy 的环境可以运行新版示例，但不能据此声称完整测试通过；完整测试的求解器依赖见仓库 `requirements-pi.txt`。

新增检查覆盖：逐车等待计时与提前事件、已有航次承诺、多车联合等待、不同接单基础方案进入同一候选池、未来信息隔离、时空需求配置、特征计算、训练日等权拟合与验证选择边界。

## 开发流程验证

在 `code/e1_v04` 下运行如下小规模配置：

```shell
python run_unified_vfa.py train --h-values 5 10 --train-layouts 710 --validation-layouts 810 --days 1 --orders 8 --share 0.4 --patterns uniform --temporal flat --iterations 1 --initial-budget 0 --candidate-budget 8 --enhancement-budget 0 --wait-budget 4 --output ../../tmp/unified_smoke_verified_20261126
```

- 完成全日仿真、独立账本核对、训练标签生成、拟合、验证选择和模型保存。
- h=5 与 h=10 的验证利润相同，按预定规则选择较小的 h=5。这只验证选择器，不说明 h=5 对研究问题最优。
- 这两个验证轨迹均未接受 pickup，无法用本次小样本检查证明学习策略能提高动态接单或等待收益。
- 模型重新加载、新版零权重示例及旧版 `--legacy` 示例也已完成全日账本检查。
- 正式测试集没有参与开发选择，尚未开展正式规模训练、特征消融、多种子检验与独立测试。

原始开发配置、输入与源码哈希见 [smoke_protocol.json](smoke_protocol.json)，验证与开发拟合记录见 [smoke_results.json](smoke_results.json)。其中权重仅为流程验证产物，不是仓库默认发布策略。后续正式实验应使用新的输出目录和冻结协议。
