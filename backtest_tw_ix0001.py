"""
00631L 回測：以加權指數 (IX0001) 200 日均線作為進出場訊號
策略比較：
  1. Buy & Hold 00631L
  2. 加權指數跌破 200MA 連續3天 → 賣出換現金，漲回連續3天 → 買回
  3. 加權指數跌破 200MA 連續3天 → 賣出換 0050，漲回連續3天 → 買回 00631L
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
import warnings, sys, io
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 下載資料 ──────────────────────────────────────────────
print("正在下載加權指數 (^TWII)、0050.TW、00631L.TW 歷史資料...")
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

# 計算加權指數 200 日均線
df["TWII_MA200"] = df["TWII_Close"].rolling(window=200).mean()
df.dropna(inplace=True)

# 加權指數是否在 200MA 之上
df["above_ma"] = (df["TWII_Close"] > df["TWII_MA200"]).astype(int)

# 連續幾天在 200MA 之上/之下
df["above_streak"] = 0
df["below_streak"] = 0
above_streak = 0
below_streak = 0
for i in range(len(df)):
    if df["above_ma"].iloc[i] == 1:
        above_streak += 1
        below_streak = 0
    else:
        below_streak += 1
        above_streak = 0
    df.iloc[i, df.columns.get_loc("above_streak")] = above_streak
    df.iloc[i, df.columns.get_loc("below_streak")] = below_streak

# 每日報酬率
df["L2X_Return"] = df["L2X_Close"].pct_change()
df["TW50_Return"] = df["TW50_Close"].pct_change()
df.dropna(inplace=True)

print(f"資料範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"共 {len(df)} 個交易日\n")


# ── 策略邏輯 ──────────────────────────────────────────────
def run_buy_and_hold(df_period):
    """單純持有 00631L"""
    capital = 10000.0
    equity_curve = []
    for i in range(len(df_period)):
        capital *= (1 + df_period.iloc[i]["L2X_Return"])
        equity_curve.append(capital)
    return capital, equity_curve


def run_backtest_cash(df_period, confirm_days=3):
    """策略1: 跌破200MA → 現金"""
    capital = 10000.0
    holding = True
    equity_curve = []
    trades = 0

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        daily_return = row["L2X_Return"]

        # 第一天無前日訊號，按初始持倉計算
        if i == 0:
            if holding:
                capital *= (1 + daily_return)
            equity_curve.append(capital)
            continue

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        sig = df_period.iloc[i - 1]

        if holding:
            should_hold = sig["below_streak"] < confirm_days
        else:
            should_hold = sig["above_streak"] >= confirm_days

        if holding and not should_hold:
            capital *= (1 + daily_return)
            holding = False
            trades += 1
        elif not holding and should_hold:
            holding = True
            trades += 1
        elif holding:
            capital *= (1 + daily_return)

        equity_curve.append(capital)

    return capital, equity_curve, trades


def run_backtest_switch(df_period, confirm_days=3):
    """策略2: 跌破200MA → 換 0050"""
    capital = 10000.0
    holding_l2x = True
    equity_curve = []
    trades = 0

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        l2x_return = row["L2X_Return"]
        tw50_return = row["TW50_Return"]

        # 第一天無前日訊號，按初始持倉計算
        if i == 0:
            if holding_l2x:
                capital *= (1 + l2x_return)
            else:
                capital *= (1 + tw50_return)
            equity_curve.append(capital)
            continue

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        sig = df_period.iloc[i - 1]

        if holding_l2x:
            should_hold_l2x = sig["below_streak"] < confirm_days
        else:
            should_hold_l2x = sig["above_streak"] >= confirm_days

        if holding_l2x and not should_hold_l2x:
            capital *= (1 + l2x_return)
            holding_l2x = False
            trades += 1
        elif not holding_l2x and should_hold_l2x:
            capital *= (1 + tw50_return)
            holding_l2x = True
            trades += 1
        elif holding_l2x:
            capital *= (1 + l2x_return)
        else:
            capital *= (1 + tw50_return)

        equity_curve.append(capital)

    return capital, equity_curve, trades


def calc_metrics(equity_curve, dates):
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
all_curves = {}

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

    # 策略1: 3天確認 → 現金
    cash_capital, cash_curve, cash_trades = run_backtest_cash(df_period, confirm_days=3)
    cash_metrics = calc_metrics(cash_curve, date_list)
    cash_metrics["交易次數"] = cash_trades

    # 策略2: 3天確認 → 0050
    sw_capital, sw_curve, sw_trades = run_backtest_switch(df_period, confirm_days=3)
    sw_metrics = calc_metrics(sw_curve, date_list)
    sw_metrics["交易次數"] = sw_trades

    all_results[period_name] = {
        "Buy & Hold 00631L": bh_metrics,
        "3天→現金": cash_metrics,
        "3天→0050": sw_metrics,
    }
    all_curves[period_name] = {
        "dates": date_list,
        "Buy & Hold 00631L": bh_curve,
        "3天→現金": cash_curve,
        "3天→0050": sw_curve,
    }

# ── 輸出績效表格 ──────────────────────────────────────────
print("=" * 80)
print("    00631L 回測結果：加權指數 (IX0001) 200 日均線策略 (連續3天確認)")
print("=" * 80)

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
