# 共用代码：因子、前向传播、配置规则各只有一份实现

| 文件 | 内容 | 使用者 |
|---|---|---|
| `factor_core.py` | 13 个因子的**唯一**实现（价格因子、目标 Y、历史因子、VIX / SEC 对齐、实盘因子面板） | `features.py`（构建 `dataset.csv`）、`quantconnect/main.py` |
| `tail_model.py` | 两段模型的**唯一**推理实现：前向传播、S(y\|x)、VaR、ES、风险评分 R、回测工具，以及 JSON 导入导出 | 训练脚本、`evaluate_tail.py`、`gpd_predict.py`、`quantconnect/export_model_for_qc.py`、`quantconnect/main.py`、`trading_strategy.ipynb` |
| `strategy_core.py` | 配置规则 `allocate_from_risk`：R → v → B → c → w | `quantconnect/main.py`、`trading_strategy.ipynb` |

三个文件都只依赖 numpy / pandas，不需要 torch，可以原样加入 QuantConnect 项目。每个文件都在 3.2 万字符以内。

PyTorch 网络只在训练脚本里定义一次，因为训练需要自动求导。训练结束后，`check_numpy_forward()` 会检查 `tail_model.py` 能否复现 PyTorch 的输出（相对误差 < 1e-4），通过后自动写出 `quantconnect/gpd_mixture_model.json`。以后修改网络结构时，要同时修改训练脚本和 `tail_model.py`，否则训练时会直接报错。

```
features.py ──▶ factor_core.py ◀── quantconnect/main.py
                      │
             artifacts/dataset.csv
                      │
TQT research machine learning part.py（唯一的 torch 网络）
        │  保存 .pt
        ▼
artifacts/{gpd_model.pt, pu_model.pt, preprocess.json}
        │
        ▼
tail_model.py ◀── evaluate_tail.py, gpd_predict.py, notebook
        │  to_json()（训练后自动导出，或运行 export_model_for_qc.py）
        ▼
quantconnect/gpd_mixture_model.json ──▶ quantconnect/main.py（TwoStageTailModel.from_json）
```

## `tail_model.py` 用法

```python
from tail_model import TwoStageTailModel

m = TwoStageTailModel.from_artifacts("artifacts")     # 读 .pt + preprocess.json（不需要 torch）
m = TwoStageTailModel.from_json(text_or_path)         # 或读导出的 JSON（QuantConnect）

out = m.predict(df)                    # DataFrame：xi_b, beta_b, xi_h, beta_h, pi, p_u
m.risk_score(df, d=0.07)               # R = E[(L-d)^+] = p_u · E[(L-d)^+ | L>u]
m.risk_score(df, d=0.07, use_pu=False) # 只看条件部分
m.tail_prob(df, loss=0.10)             # P(第二天跌幅 > 10%)
m.var(df, alpha=0.01), m.es(df, alpha=0.01)   # 无条件 VaR99 / ES99（p_u ≤ α 时为 NaN）
m.cond_survival(df, y), m.cond_quantile(df, 0.99)
m.to_json("quantconnect/gpd_mixture_model.json")
```

输入：第 t 天的 13 个**原始**因子（未标准化），可以是 DataFrame、dict 或 (n, 13) 数组。NaN 会填成训练期中位数。

函数版本：输入参数数组 `(n, 5)`，供训练脚本和画图使用：`mixture_survival`、`mixture_logpdf`、`mixture_pdf`、`mixture_quantile_excess`、`tail_probability`、`var_es`、`mixture_tail_risk`、`gpd_expected_excess`、`auc_score`、`kupiec_test`、`var_backtest`、`var99_by_decile`。

命令行：

```bash
python tail_model.py --selftest
python tail_model.py --csv factors.csv --out predictions.csv --d 0.07 --alpha 0.01
```

## QuantConnect 项目需要的文件

`main.py`（来自 `quantconnect/`），以及仓库根目录原样复制的 `factor_core.py`、`tail_model.py`、`strategy_core.py`。Object Store 里放 `gpd_mixture_model.json`（键 `MODEL_KEY`），可选放 `qc_sec_factors.csv`（键 `SEC_KEY`）。

`main.py` 新增了配置项 `USE_PU = True`：风险评分乘上 p_u。设为 False 就是原来只用条件部分的做法。

## 测试

```bash
python tests/test_parity.py
```

1. `factor_core` 和构建数据集时的因子公式逐项一致（价格因子、目标、历史因子、QuantConnect 的实盘面板）；
2. JSON 读写前后输出完全一致，`quantconnect/gpd_mixture_model.json` 就是 `artifacts/` 里的模型；
3. 如果有 `artifacts/dataset.csv`，能复现 `preprocess.json` / `pu_meta.json` 里记录的验证结果；
4. 如果装了 torch，和训练脚本的 PyTorch 网络逐项对比。

## 验证结果（本次修改）

- **因子**：`factor_core` 与修改前的 `features.py`、修改前的 QuantConnect `main.py`，在合成数据（含缺失行和 NaN）上差值为 0；在 553,529 行的真实数据上，3 个历史因子差值也为 0。
- **推理**：在验证期上，p_u 的对数损失和 AUC（0.26197 / 0.7110）、GPD 的 NLL（−3.1203），以及 6 个 S<α 比例，都与训练时记录的数值一致；VaR、ES、回测函数与训练脚本原来的实现差值不超过 2e-15。
- `evaluate_tail.py` 完整运行（不需要 torch）。`quantconnect/main.py` 在模拟的 QuantConnect 接口上跑通了 `initialize()` 和 `rebalance()`。
