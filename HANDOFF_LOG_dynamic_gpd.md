# 交接日志：动态 GPD 尾部风险模型（Research 2 / lazy_prices）

> 用途：把这份日志交给新的 Claude（VS Code 环境），让它在**原始机器学习模型代码**的基础上，接入 **Alpaca + FRED + SEC EDGAR** 的数据，在**本地**训练模型，并**把训练好的模型权重保存成本地文件**。
> 附件：`main_quantconnect_version.py` = 在 QuantConnect 上已经跑通的版本，只作为因子计算逻辑的参考，**不要直接在本地运行**（它依赖 QuantConnect 的 `AlgorithmImports`）。

---

## 1. 项目是什么

- 研究目标：建立一个**广义帕累托分布（GPD）**模型，用来描述股票收益率的厚尾（暴跌）风险。GPD 的两个参数——形状参数 **ε (epsilon/xi)** 和尺度参数 **β (beta)**——不再固定，而是由**两个独立的神经网络**根据实时市场因子动态预测。
- 输入：每行 = 某只股票某个交易日的 **10 个因子**（矩阵 X，形状 N×10）。
- 输出：网络输出 (ε, β)，用 **GPD 负对数似然 + 尾部加权惩罚** 做损失函数，梯度下降训练。
- 最终用途：给定当天的因子 → 得到当天的 GPD 分布 → 估计暴跌概率 → 调整持仓。

## 2. 到目前为止的进度

1. 在 QuantConnect 云端完成了数据接入和训练（32 只股票，2016–2024 日频），能跑通：
   - X 形状 (72384, 10)，Y 形状 (72384, 1)
   - 训练集 < 2023-03-13 ≤ 验证集（按日期切分）
   - 尾部阈值 u = 单日亏损 1.7029%，超过阈值的样本：训练 5789 个，验证 1242 个
   - 每次训练大约 18–21 轮后早停，验证集最好的 loss 约为 −2.89 到 −2.91
2. **问题 1：模型在 QuantConnect 上无法下载。** 组织是 Trading Firm 级别，Object Store 的下载需要 Institution 级并签"衍生数据协议"；机器学习模型权重也算衍生数据，所以下载不了。**结论：改为在本地用免费数据源重新训练。**
3. **问题 2：ε 估计不稳定。** 同样的数据跑了 3 次，loss 几乎一样（−2.906 / −2.901 / −2.892），但 ε 变化很大，例如 VZ 在 2024-12-27 的预测：0.52 → 0.23 → 0.52；CAT：0.44 → 0.31 → 0.56。说明 ε 的可辨识性差（见第 9 节的待办）。
4. 已经确认的免费数据源（见第 5 节），并用用户自己的 Alpaca key 在文档页 "Try It" 成功拿到了数据（AAPL 2016-01 日线，HTTP 200）。

## 3. 关键设计决定（请保持一致）

| 项目 | 决定 |
|---|---|
| 股票池 | 32 只跨行业的大盘股，2016–2024 一直在交易（有幸存者偏差，报告中要注明）：`AAPL MSFT AMZN GOOGL NVDA INTC CSCO ORCL IBM JPM BAC WFC GS XOM CVX JNJ PFE MRK UNH PG KO PEP WMT HD MCD NKE DIS CAT BA MMM T VZ` |
| 时间点 | 2016-01-01 到 2024-12-31 的**每个交易日收盘后**取一次（约 32 × 2200 ≈ 7 万行）。原因：Y 是下一个交易日的收益，按日取正好对齐；GPD 只用最差的 10% 样本，样本必须足够多 |
| Y 的定义 | `Y_t = Close(t+1) / Close(t) − 1`（下一个交易日的实际收益率，形状 N×1）。备选 "overnight"：`Open(t+1)/Close(t) − 1`，做成配置项 `TARGET_MODE` |
| 无前视偏差 | X 的第 t 行只能用截至第 t 天收盘的信息；SEC 文件只在**可用日之后**才能使用 |
| GPD 训练样本 | Peaks-over-threshold：亏损 L = −Y；阈值 u = **训练集** L 的 90% 分位数；只用 L > u 的行训练，目标值 = L − u（非负）。原始损失函数里的 `log1p(ε·y/β)` 要求 y ≥ 0，所以不能把原始收益率直接喂进去 |
| 训练 / 验证切分 | **按日期**切 80/20（同一天的所有股票只能在同一侧），不能随机打乱 |
| 预处理 | SEC 因子的缺失值用**训练集中位数**填补；用**训练集**的均值和方差标准化；z 值截断在 ±5 |
| SCALE | `max(median(训练集超阈值样本), 1e-4)`，同时用于 β 的输出缩放和损失函数里的严重度权重 |

## 4. 10 个因子（定义 + 数据来源）

r = 日对数收益率 `log(Close).diff()`

| # | 名称 | 公式 | 来源 |
|---|---|---|---|
| 1 | vol_20d | `r.rolling(20).std() * sqrt(252)` | Alpaca |
| 2 | downside_vol_20d | `sqrt((r.clip(upper=0)**2).rolling(20).mean()) * sqrt(252)` | Alpaca |
| 3 | kurtosis_60d | `r.rolling(60).kurt()` | Alpaca |
| 4 | momentum_60d | `Close / Close.shift(60) − 1` | Alpaca |
| 5 | reversal_5d | `Close / Close.shift(5) − 1` | Alpaca |
| 6 | range_5d | `((High − Low) / Close).rolling(5).mean()` | Alpaca |
| 7 | abn_volume | `log((Vol.rolling(5).mean()+1) / (Vol.rolling(60).mean()+1))` | Alpaca（必须用 feed=sip） |
| 8 | vix | VIX 每日收盘价，按日期前向填充 | FRED，序列 `VIXCLS` |
| 9 | n_8k_30d | 过去 30 个自然日内 8-K 的数量（按文件**可用日**计） | SEC EDGAR |
| 10 | filing_change | 1 − cos 相似度（最新一份 10-K/10-Q 正文 vs 上一份**同类型**报告），"Lazy Prices" 思路；用 HashingVectorizer(n_features=2**18, 只保留 ≥3 个字母的英文词、去停用词、L2 归一化) | SEC EDGAR |

价格因子和 VIX 不能缺失（缺失的行直接删掉）；因子 10 早期没有上一份报告可比时为 NaN，用训练集中位数填补。

## 5. 数据源与调用方法

### 5.1 Alpaca（因子 1–7 和 Y）
- 接口：`GET https://data.alpaca.markets/v2/stocks/bars`
- 参数：
  - `symbols`：逗号分隔，可以一次查多只股票
  - `timeframe=1Day`
  - `start=2016-01-01`，`end=2024-12-31`
  - `limit=10000`（这是所有股票加起来的总数，不是每只股票 1 万）
  - `adjustment=all`（**必须复权**，否则拆股那天会出现假暴跌，污染尾部样本）
  - `feed=sip`（全市场数据；IEX 只占约 2.5% 成交量，会让成交量和振幅失真）
  - 翻页：循环用返回的 `next_page_token` 作为下一次请求的 `page_token`，直到它为 null
- 返回：`bars[symbol]` 是一个列表，每根 K 线的字段：`t` 时间，`o h l c` 开高低收，`v` 成交量，`n` 成交笔数，`vw` 成交量加权均价。
- 请求头：`APCA-API-KEY-ID`、`APCA-API-SECRET-KEY`
- 免费 Basic 套餐：历史数据从 2016-01-04 开始，每分钟 200 次历史请求，只是不能查最近 15 分钟的数据，对本项目没有影响。整个数据集大约 8 页请求就能拉完。
- `asof` 参数可以自动处理股票改名（例如 FB → META）。
- **已知影响**：Alpaca 没有 2016 年以前的数据，60 日窗口需要预热，有效样本从 2016 年 4 月左右开始（每只股票约 2200 行，对 GPD 影响很小）。如果要严格从 2016-01-01 开始，可以用 Tiingo 的日线（免费套餐：每小时 50 次、每天 1000 次）来预热。
- **一定要缓存**：下载一次后存成本地文件（parquet 或 csv），之后训练只读本地文件，不要每次都调 API。

### 5.2 FRED（因子 8）
- 序列 `VIXCLS`（CBOE VIX 每日收盘）。可以用 FRED API（需要免费 key），也可以直接下载 CSV：`https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS`（不需要 key）。
- 节假日的值是 "."，要转成 NaN，再前向填充。

### 5.3 SEC EDGAR（因子 9、10）
- 免费，不需要 key，但**必须带 User-Agent 请求头**（格式 "姓名 邮箱"），频率不超过每秒 10 次。
- 股票代码 → CIK：`https://www.sec.gov/files/company_tickers.json`
- 某家公司的申报历史：`https://data.sec.gov/submissions/CIK##########.json`（CIK 补零到 10 位）。字段 `filings.recent` 里有 `form`、`filingDate`、`acceptanceDateTime`、`accessionNumber`、`primaryDocument`；更早的记录在 `filings.files` 指向的额外 JSON 里，2016 年以前的数据要一起读。
- 正文：`https://www.sec.gov/Archives/edgar/data/{CIK}/{accessionNumber去掉横线}/{primaryDocument}`，先去掉 HTML 标签。
- 可用日：用 `acceptanceDateTime`。如果是在美东时间 16:00 之后提交的，算作**下一个交易日**才可用（偏保守，避免前视偏差）。
- 8-K 计数用 `form == "8-K"`（是否计入 8-K/A 自己决定，但要保持一致）；相似度只比较 10-K 对 10-K、10-Q 对 10-Q。
- 下载的正文要**缓存到本地**，避免重复请求。

## 6. API key 填在哪里（用户手动填）

在项目根目录创建一个 `.env` 文件，由**用户自己**填入（不要把 key 写进代码，也不要提交到 git，把 `.env` 加进 `.gitignore`）：

```
# .env  ← 用户手动填写这三行
ALPACA_API_KEY_ID=在这里填你的 API Key ID
ALPACA_API_SECRET_KEY=在这里填你的 Secret Key
SEC_USER_AGENT=你的名字 你的邮箱
# 如果用 FRED API（不是直接下载 CSV）才需要：
FRED_API_KEY=在这里填你的 FRED key
```

代码里用 `python-dotenv` 读取：`os.environ["ALPACA_API_KEY_ID"]` 等。**新 Claude 需要在代码里留出这几个位置，并明确告诉用户要在 `.env` 里手动填写。**
（Alpaca key 的获取方式：Trading Dashboard → Paper Trading 账户 → 右侧 API Keys → Generate New Keys。Secret Key 只显示一次，丢了只能重新生成。）

## 7. 原始机器学习模型代码（更改前的版本，请在此基础上修改）

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, regularizers


INPUT_DIM = 10
HIDDEN_UNITS = (16, 8)   
L2 = 1e-4

LAMBDA_TAIL = 0.2
LAMBDA_SEVERITY = 1.0

#Build Dense Neural Network


def build_network(name, neurons, use_bn=True):

    model = Sequential(name=name)
    model.add(layers.Input(shape=(INPUT_DIM,)))

    for units in neurons:
        model.add(layers.Dense(
            units,
            use_bias=not use_bn,
            kernel_regularizer=regularizers.l2(L2)
        ))

        if use_bn:
            model.add(layers.BatchNormalization())

        model.add(layers.Activation("relu"))

    model.add(layers.Dense(1, activation="linear"))

    return model

# 3. Two Independent Networks

epsilon_network = build_network("Epsilon_Network", HIDDEN_UNITS)
beta_network = build_network("Beta_Network", HIDDEN_UNITS)


# 4. Prepare Data
# ==========================================

# Demo data: replace X and y with your real data.
# X: (N, 10) market features known before the loss.
# y: (N,) nonnegative excess losses above threshold.

rng = np.random.default_rng(42)
X = rng.normal(size=(600, 10)).astype("float32")

# Simulated GPD exceedances, for demonstration only
true_beta = 0.02 * np.exp(0.2 * X[:, 0])
u = rng.uniform(size=len(X))
y = (true_beta / 0.2) * ((1 - u)**(-0.2) - 1)
y = y.astype("float32")

# Chronological split: 80% train, 20% validation
split = int(0.8 * len(X))

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Standardize features using training data only
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Reference scale for GPD parameters and loss
SCALE = max(float(np.median(y_train)), 1e-4)


# 5. Combine Networks into Dynamic GPD


inputs = layers.Input(shape=(INPUT_DIM,))

epsilon_raw = epsilon_network(inputs)
beta_raw = beta_network(inputs)

# Shape parameter: 0.01 < xi < 0.8
epsilon = layers.Rescaling(0.79, offset=0.01)(
    layers.Activation("sigmoid")(epsilon_raw)
)

# Scale parameter: beta > 0
beta = layers.Rescaling(SCALE, offset=1e-6)(
    layers.Activation("softplus")(beta_raw)
)

outputs = layers.Concatenate()([epsilon, beta])

gpd_model = Model(inputs, outputs, name="Dynamic_GPD")


# ==========================================
# 6. GPD NLL + Severity-Weighted Tail Loss
# ==========================================

def gpd_loss(y_true, y_pred):

    y = tf.reshape(
        tf.cast(y_true, y_pred.dtype), (-1, 1)
    )

    epsilon = y_pred[:, 0:1]
    beta = y_pred[:, 1:2]

    # GPD Negative Log-Likelihood
    log_term = tf.math.log1p(epsilon * y / beta)

    nll = (
        tf.math.log(beta)
        + (1.0 + 1.0 / epsilon) * log_term
    )

    # Tail Probability Penalty
    tail_loss = log_term / epsilon

    # Severity Weight
    weight = 1.0 + LAMBDA_SEVERITY * y / SCALE

    # Combined Loss
    total_loss = nll + LAMBDA_TAIL * weight * tail_loss

    return tf.squeeze(total_loss, axis=-1)


# ==========================================
# 7. Compile and Train
# ==========================================

gpd_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    ),
    loss=gpd_loss
)

history = gpd_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )
    ],
    verbose=1
)


# ==========================================
# 8. Predict Dynamic GPD Parameters
# ==========================================

predictions = gpd_model.predict(X_val[:5], verbose=0)

print("Predicted Shape (epsilon):")
print(predictions[:, 0])

print("\nPredicted Scale (beta):")
print(predictions[:, 1])

gpd_model.summary()
```

**修改原则**：网络结构、ε/β 的输出约束、损失函数的数学形式**保持不变**；只把第 4 节的 Demo 数据换成真实数据（按第 3、4 节的规则构造 X、Y → 超阈值样本）。注意 SCALE 要在模型构建之前算出来（QuantConnect 版本把它改成了 `build_gpd_model(scale)` / `make_gpd_loss(scale)` 两个函数，可以参考）。

## 8. 新 Claude 的任务清单

1. **项目结构**（建议）：`data_fetch.py`（Alpaca / FRED / EDGAR 下载 + 本地缓存）、`features.py`（10 个因子和 Y）、`train.py`（原始模型代码 + 真实数据）、`.env`、`.gitignore`、`requirements.txt`（numpy、pandas、requests、python-dotenv、scikit-learn、tensorflow、pyarrow）。
2. 构建 X (N×10)、Y (N×1)，以及 dates、tickers；把完整数据集存成 `artifacts/dataset.csv`。
3. 按第 3 节做切分、填补缺失、标准化、取超阈值样本，然后训练。
4. **保存训练结果到本地 `artifacts/` 文件夹**（这是用户最关心的）：
   - `gpd_model.keras`（完整模型）
   - `gpd_model.weights.h5`（只有权重）
   - `preprocess.json`：特征名、训练集均值/方差/中位数、截断值 5、阈值 u、SCALE、TAIL_QUANTILE、TARGET_MODE、切分日期、超阈值样本数、超参数、loss 和 val_loss 历史
   - 加载方式：`tf.keras.models.load_model(path, compile=False)`（损失函数是闭包，加载时不用 compile）
5. 打印和 QuantConnect 版本相同的诊断信息（X/Y 形状、u、超阈值样本数、训练轮数、最好的 val_loss、最近几行的 ε/β），方便和第 2 节的结果对照。
6. 告诉用户需要在 `.env` 的哪几行手动填入 key（见第 6 节）。

## 9. 后续待办（用户已经同意方向，尚未实现）

- **ε 不稳定**：可以训练 5–10 个模型取平均（同时报告波动范围）；或者加大正则化、缩小 ε 网络；或者让 ε 只依赖少数几个尾部相关的因子（峰度、VIX、下行波动率）。
- **评估**：在验证集上做 VaR 突破率回测（例如 1% VaR 的实际突破率应该接近 1%），并和**固定参数 GPD**（同一个阈值 u，用训练集做最大似然拟合）比较验证集的负对数似然。这是研究的核心论点。
- **调参**：TAIL_QUANTILE（0.90 / 0.95）、LAMBDA_TAIL、LAMBDA_SEVERITY、学习率、L2；ε 的预测值（0.23–0.67）对个股日收益来说偏高（一般在 0.2–0.4 之间），需要关注。
- **随机性**：设置 `tf.keras.utils.set_random_seed(42)` 和 `tf.config.experimental.enable_op_determinism()`，让结果可以复现。
