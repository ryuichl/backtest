"""
00631L 回測：金字塔買入策略（2 層）
訊號源: 加權指數 (^TWII) 200 日均線
策略邏輯：
  - 加權指數跌破 200MA 連續 3 天 → 賣出全部 00631L，轉現金
  - 加權指數跌到 2 年線 (504日均線) → 買入第一層倉位 00631L
  - 加權指數跌到 3 年線 (756日均線) → 買入剩餘全部倉位 00631L
  - 加權指數漲回 200MA 連續 3 天 → 全部買回 00631L
對照：Buy & Hold、3天→現金、3天→0050 輪換策略
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
import warnings
import sys
import io
warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 下載資料 ──────────────────────────────────────────────
print("正在下載加權指數、0050、00631L 歷史資料...")
tickers = {"TWII": "^TWII", "0050": "0050.TW", "00631L": "00631L.TW"}
raw = {}
for name, ticker in tickers.items():
    data = yf.download(ticker, start="2014-01-01", end=datetime.today().strftime("%Y-%m-%d"), progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    raw[name] = data["Close"]

df = pd.DataFrame({
    "TWII_Close": raw["TWII"],
    "TW50_Close": raw["0050"],
    "L2X_Close": raw["00631L"],
})
df.dropna(inplace=True)

# ── 計算均線 ──────────────────────────────────────────────
df["TWII_MA200"] = df["TWII_Close"].rolling(window=200).mean()
df["TWII_MA504"] = df["TWII_Close"].rolling(window=504).mean()   # 2 年線
df["TWII_MA756"] = df["TWII_Close"].rolling(window=756).mean()   # 3 年線
df.dropna(inplace=True)

# 加權指數是否在 200MA 之上
df["above_ma200"] = (df["TWII_Close"] > df["TWII_MA200"]).astype(int)

# 連續幾天在 200MA 之上/之下
df["above_streak"] = 0
df["below_streak"] = 0
above_streak = 0
below_streak = 0
for i in range(len(df)):
    if df["above_ma200"].iloc[i] == 1:
        above_streak += 1
        below_streak = 0
    else:
        below_streak += 1
        above_streak = 0
    df.iloc[i, df.columns.get_loc("above_streak")] = above_streak
    df.iloc[i, df.columns.get_loc("below_streak")] = below_streak

# 加權指數是否跌破各年線
df["below_ma504"] = (df["TWII_Close"] <= df["TWII_MA504"]).astype(int)
df["below_ma756"] = (df["TWII_Close"] <= df["TWII_MA756"]).astype(int)

# 每日報酬率
df["L2X_Return"] = df["L2X_Close"].pct_change()
df["TW50_Return"] = df["TW50_Close"].pct_change()
df.dropna(inplace=True)

print(f"資料範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"共 {len(df)} 個交易日\n")


# ── 策略邏輯 ──────────────────────────────────────────────
def run_pyramid_strategy(df_period, ratios=(0.5, 0.5)):
    """
    金字塔買入策略（2 層），可自訂各層買入比例。
    ratios: (2年線比例, 3年線比例)，合計應為 1.0
    """
    r_2y, r_3y = ratios
    cash = 0.0
    l2x_value = 10000.0
    equity_curve = []
    trades = 0
    state = "HOLD_ALL"

    hit_2y = False
    hit_3y = False
    sold_total = 0.0

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        l2x_ret = row["L2X_Return"]

        if l2x_value > 0:
            l2x_value *= (1 + l2x_ret)

        total = cash + l2x_value

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        if i == 0:
            equity_curve.append(total)
            continue
        sig = df_period.iloc[i - 1]

        if state == "HOLD_ALL":
            if sig["below_streak"] >= 3:
                cash = total
                l2x_value = 0.0
                sold_total = total
                state = "SOLD"
                hit_2y = False
                hit_3y = False
                trades += 1

        elif state == "SOLD":
            if sig["above_streak"] >= 3:
                l2x_value = total
                cash = 0.0
                state = "HOLD_ALL"
                trades += 1
            elif sig["below_ma504"] == 1 and not hit_2y:
                buy_amount = sold_total * r_2y
                buy_amount = min(buy_amount, cash)
                l2x_value = buy_amount
                cash = total - buy_amount
                state = "PYRAMID_PARTIAL"
                hit_2y = True
                trades += 1

        elif state == "PYRAMID_PARTIAL":
            total = cash + l2x_value

            if sig["above_streak"] >= 3:
                l2x_value = total
                cash = 0.0
                state = "HOLD_ALL"
                trades += 1
            else:
                if sig["below_ma756"] == 1 and not hit_3y:
                    l2x_value += cash
                    cash = 0.0
                    hit_3y = True
                    trades += 1

        equity_curve.append(cash + l2x_value)

    return cash + l2x_value, equity_curve, trades


def run_backtest_switch(df_period, alt_return_col=None, confirm_days=3):
    """
    輪換策略：跌破 200MA 連續 N 天時換持替代標的
    alt_return_col=None 表示換現金
    """
    capital = 10000.0
    holding_main = True
    equity_curve = []
    trades = 0

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        main_return = row["L2X_Return"]
        alt_return = row[alt_return_col] if alt_return_col else 0.0

        # 第一天無前日訊號，按初始持倉計算
        if i == 0:
            if holding_main:
                capital *= (1 + main_return)
            else:
                if alt_return_col:
                    capital *= (1 + alt_return)
            equity_curve.append(capital)
            continue

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        sig = df_period.iloc[i - 1]

        if holding_main:
            should_hold = sig["below_streak"] < confirm_days
        else:
            should_hold = sig["above_streak"] >= confirm_days

        if holding_main and not should_hold:
            capital *= (1 + main_return)
            holding_main = False
            trades += 1
        elif not holding_main and should_hold:
            if alt_return_col:
                capital *= (1 + alt_return)
            holding_main = True
            trades += 1
        elif holding_main:
            capital *= (1 + main_return)
        else:
            if alt_return_col:
                capital *= (1 + alt_return)

        equity_curve.append(capital)

    return capital, equity_curve, trades


def run_buy_and_hold(df_period):
    """單純持有 00631L"""
    capital = 10000.0
    equity_curve = []
    for i in range(len(df_period)):
        capital *= (1 + df_period.iloc[i]["L2X_Return"])
        equity_curve.append(capital)
    return capital, equity_curve


def calc_metrics(equity_curve, dates):
    """計算績效指標"""
    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]

    total_return = (eq[-1] / eq[0] - 1) * 100
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = ((eq[-1] / eq[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd = drawdown.min() * 100

    annual_vol = np.std(returns) * np.sqrt(252) * 100

    excess_return = np.mean(returns) * 252 - 0.02  # 台灣無風險利率約 2%
    sharpe = excess_return / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

    return {
        "總報酬率": f"{total_return:,.1f}%",
        "年化報酬率(CAGR)": f"{cagr:.1f}%",
        "最大回撤": f"{max_dd:.1f}%",
        "年化波動率": f"{annual_vol:.1f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "期末資產": f"${eq[-1]:,.0f}",
    }


# ── 定義時間範圍 ──────────────────────────────────────────
today = df.index[-1]
periods = {}
df_all = df.copy()
if len(df_all) > 0:
    periods["全部資料"] = df_all
df_10y = df[df.index >= (today - timedelta(days=365 * 10))]
if len(df_10y) > 200:
    periods["近 10 年"] = df_10y
df_5y = df[df.index >= (today - timedelta(days=365 * 5))]
if len(df_5y) > 200:
    periods["近 5 年"] = df_5y

# ── 執行回測 ──────────────────────────────────────────────
all_results = {}

for period_name, df_period in periods.items():
    df_period = df_period.copy().reset_index(drop=False)

    if "Date" in df_period.columns:
        date_list = pd.to_datetime(df_period["Date"]).tolist()
    else:
        date_list = df_period.index.tolist()

    # Buy & Hold
    bh_capital, bh_curve = run_buy_and_hold(df_period)
    bh_metrics = calc_metrics(bh_curve, date_list)
    bh_metrics["交易次數"] = 0

    # 3天→現金 輪換策略
    d3c_capital, d3c_curve, d3c_trades = run_backtest_switch(df_period, None)
    d3c_metrics = calc_metrics(d3c_curve, date_list)
    d3c_metrics["交易次數"] = d3c_trades

    # 3天→0050 輪換策略
    d3t_capital, d3t_curve, d3t_trades = run_backtest_switch(df_period, "TW50_Return")
    d3t_metrics = calc_metrics(d3t_curve, date_list)
    d3t_metrics["交易次數"] = d3t_trades

    # 金字塔策略 - 不同比例
    pyramid_configs = {
        "等比 50/50": (0.50, 0.50),
        "前重 70/30": (0.70, 0.30),
        "前重 60/40": (0.60, 0.40),
        "後重 30/70": (0.30, 0.70),
        "後重 40/60": (0.40, 0.60),
        "全押2Y 100/0": (1.00, 0.00),
    }

    results_dict = {
        "Buy & Hold": bh_metrics,
        "3天→現金": d3c_metrics,
        "3天→0050": d3t_metrics,
    }

    for config_name, ratios in pyramid_configs.items():
        pyr_capital, pyr_curve, pyr_trades = run_pyramid_strategy(df_period, ratios=ratios)
        pyr_metrics = calc_metrics(pyr_curve, date_list)
        pyr_metrics["交易次數"] = pyr_trades
        results_dict[config_name] = pyr_metrics

    all_results[period_name] = results_dict

# ── 輸出績效表格 ──────────────────────────────────────────
print("=" * 80)
print("    00631L 回測：金字塔不同比例 vs 輪換策略")
print("=" * 80)
print("\n訊號源: 加權指數 (^TWII) 200 日均線")
print("策略說明：")
print("  輪換：加權指數跌破 200MA 3天 → 換持 現金/0050，漲回 3天 → 換回 00631L")
print("  金字塔 (X/Y)：加權指數跌破 200MA 3天賣出 →")
print("    跌到 2年線 買 X%，跌到 3年線 買剩餘全部")
print("    漲回 200MA 3天 → 全部買回 00631L")

for period_name, strategies in all_results.items():
    print(f"\n{'─' * 80}")
    print(f"  {period_name}")
    print(f"{'─' * 80}")

    headers = ["指標"] + list(strategies.keys())
    metric_keys = ["總報酬率", "年化報酬率(CAGR)", "最大回撤", "年化波動率", "Sharpe Ratio", "期末資產", "交易次數"]

    table_data = []
    for key in metric_keys:
        row = [key]
        for strat in strategies.values():
            row.append(strat[key])
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="right"))

print("\n回測完成！")
