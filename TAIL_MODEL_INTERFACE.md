# `tail_model.py`：两段式尾部风险模型的即插即用接口

一个文件，只依赖 numpy（DataFrame 输入输出需要 pandas）。它**直接读取** `artifacts/` 里训练好的 PyTorch 权重，不需要安装 torch，也不需要先导出：

| 文件 | 内容 |
|---|---|
| `artifacts/preprocess.json` | 特征顺序、训练期中位数/均值/标准差、截断值、阈值 u、尺度 s、π_min |
| `artifacts/gpd_model.pt` | 第二段：门控 GPD 混合模型（26,341 个参数） |
| `artifacts/pu_model.pt` | 第一段：跌破概率网络 p_u（23,337 个参数） |

$$
P(L > \ell \mid x) = p_u(x)\cdot S(\ell-u \mid x),\qquad
S(y\mid x) = (1-\pi)\,S_{\text{GPD}}(y;\xi_b,\beta_b)+\pi\,S_{\text{GPD}}(y;\xi_h,\beta_h)
$$

## 用法

```python
from tail_model import TwoStageTailModel

m = TwoStageTailModel.from_artifacts("artifacts")

# 输入：第 t 天的 13 个原始因子（未标准化）。DataFrame、dict 或 (n, 13) 数组都可以；NaN 会填成训练期中位数
out = m.predict(df)                    # DataFrame：xi_b, beta_b, xi_h, beta_h, pi, p_u

m.tail_prob(df, loss=0.10)             # P(第二天跌幅 > 10%)
m.var(df, alpha=0.01)                  # 无条件 VaR99（正数表示损失；p_u ≤ α 时为 NaN，表示 VaR 在 u 以下）
m.es(df, alpha=0.01)                   # 无条件 ES99
m.risk_score(df, d=0.07)               # 风险评分 R = E[(L-d)^+] = p_u · E[(L-d)^+ | L>u]
m.risk_score(df, d=0.07, use_pu=False) # 只看条件部分 E[(L-d)^+ | L>u]

m.cond_survival(df, y)                 # S(y|x)：已跌破 u 的前提下，再多跌 y 以上的概率
m.cond_quantile(df, 0.99)              # 条件 q99（尾部图里的灰色虚线），以损失水平表示
m.gpd_params(df), m.exceedance_prob(df)   # 两段网络各自的原始输出
```

命令行：

```bash
python tail_model.py --selftest
python tail_model.py --csv factors.csv --out predictions.csv --d 0.07 --alpha 0.01
```

`--csv` 输出的列：原始列、5 个分布参数、p_u、risk_score、VaR、ES、cond_q99。

## 13 个因子（列名必须一致）

`vol_20d, downside_vol_20d, kurtosis_60d, momentum_60d, reversal_5d, range_5d, abn_volume, vix, n_8k_30d, filing_change, worst252, e63, echo_q`

计算方法必须与训练时的 `features.py` 完全一致（例如波动率用对数收益，abn_volume = log((5 日均量+1)/(60 日均量+1))）。拿不到的因子可以传 NaN，会被填成训练期中位数。

## 注意

- 必须使用训练时的预处理统计量，不能在新数据上重新计算均值和标准差。本接口已经内置这一点。
- BatchNorm 使用训练得到的滑动统计量，所以每只股票的输出只取决于它自己的因子，和同一批里有哪些股票无关。
- 重新训练后，直接替换 `artifacts/` 里的三个文件即可，不需要改代码。

## 验证

在 `dataset.csv` 的验证期（2023-03-30 之后）上，本接口算出的结果与训练脚本记录在 `preprocess.json` / `pu_meta.json` 里的数值一致：

| 指标 | 本接口 | 训练时记录 |
|---|---|---|
| p_u 验证对数损失 / AUC | 0.26197 / 0.7110 | 0.26197 / 0.7110 |
| GPD 验证 NLL（8,211 个超阈值样本） | −3.1203 | −3.1203 |
| S < 1% / 2.5% / 5% / 10% / 25% / 50% | 0.889 / 2.119 / 4.250 / 8.781 / 23.700 / 48.922 % | 完全相同 |

合成后的无条件 VaR 回测（全部验证行）：超出 VaR99 的比例为 0.829%（期望 1%），超出 VaR99.9 的比例为 0.093%（期望 0.1%）。
