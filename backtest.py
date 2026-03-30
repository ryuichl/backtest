"""
TQQQ 回測：以 QQQ 200 日均線作為進出場訊號
策略比較：
  1. Buy & Hold TQQQ
  2. 即時/3天 → 現金
  3. 即時/3天 → QQQ
  4. 即時/3天 → SPMO
時間範圍：近 10 年、近 5 年
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

# 修正 Windows 終端編碼問題
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 下載資料 ──────────────────────────────────────────────
print("正在下載 QQQ、TQQQ、SPMO 歷史資料...")
tickers = {"QQQ": "QQQ", "TQQQ": "TQQQ", "SPMO": "SPMO"}
raw = {}
for name, ticker in tickers.items():
    data = yf.download(ticker, start="2010-02-11", end=datetime.today().strftime("%Y-%m-%d"), progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    raw[name] = data["Close"]

# 合併資料
df = pd.DataFrame({
    "QQQ_Close": raw["QQQ"],
    "TQQQ_Close": raw["TQQQ"],
    "SPMO_Close": raw["SPMO"],
})
df.dropna(inplace=True)

# 計算 QQQ 200 日均線
df["QQQ_MA200"] = df["QQQ_Close"].rolling(window=200).mean()
df.dropna(inplace=True)

# QQQ 是否在 200MA 之上
df["above_ma"] = (df["QQQ_Close"] > df["QQQ_MA200"]).astype(int)

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
df["TQQQ_Return"] = df["TQQQ_Close"].pct_change()
df["QQQ_Return"] = df["QQQ_Close"].pct_change()
df["SPMO_Return"] = df["SPMO_Close"].pct_change()
df.dropna(inplace=True)

print(f"資料範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"共 {len(df)} 個交易日\n")


# ── 策略邏輯 ──────────────────────────────────────────────
def run_backtest(df_period, strategy_name, confirm_days=0):
    """
    回測策略
    confirm_days=0: 即時策略（當天穿越就操作）
    confirm_days=3: 連續 3 天確認後才操作
    """
    capital = 10000.0
    holding = True  # 開始時持有 TQQQ
    equity_curve = []
    trades = 0

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        daily_return = row["TQQQ_Return"]

        # 第一天無前日訊號，按初始持倉計算
        if i == 0:
            if holding:
                capital *= (1 + daily_return)
            equity_curve.append(capital)
            continue

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        sig = df_period.iloc[i - 1]

        if confirm_days == 0:
            should_hold = sig["above_ma"] == 1
        else:
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


def run_backtest_switch(df_period, alt_return_col, confirm_days=0):
    """
    回測策略：跌破 200MA 時換持替代標的（而非現金）
    alt_return_col: 替代標的的報酬率欄位名（如 "QQQ_Return", "SPMO_Return"）
    confirm_days=0: 即時策略
    confirm_days=3: 連續 3 天確認後才操作
    """
    capital = 10000.0
    holding_tqqq = True  # 開始時持有 TQQQ
    equity_curve = []
    trades = 0

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        tqqq_return = row["TQQQ_Return"]
        alt_return = row[alt_return_col]

        # 第一天無前日訊號，按初始持倉計算
        if i == 0:
            if holding_tqqq:
                capital *= (1 + tqqq_return)
            else:
                capital *= (1 + alt_return)
            equity_curve.append(capital)
            continue

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        sig = df_period.iloc[i - 1]

        if confirm_days == 0:
            should_hold_tqqq = sig["above_ma"] == 1
        else:
            if holding_tqqq:
                should_hold_tqqq = sig["below_streak"] < confirm_days
            else:
                should_hold_tqqq = sig["above_streak"] >= confirm_days

        if holding_tqqq and not should_hold_tqqq:
            capital *= (1 + tqqq_return)
            holding_tqqq = False
            trades += 1
        elif not holding_tqqq and should_hold_tqqq:
            capital *= (1 + alt_return)
            holding_tqqq = True
            trades += 1
        elif holding_tqqq:
            capital *= (1 + tqqq_return)
        else:
            capital *= (1 + alt_return)

        equity_curve.append(capital)

    return capital, equity_curve, trades


def run_buy_and_hold(df_period):
    """單純持有 TQQQ"""
    capital = 10000.0
    equity_curve = []
    for i in range(len(df_period)):
        daily_return = df_period.iloc[i]["TQQQ_Return"]
        capital *= (1 + daily_return)
        equity_curve.append(capital)
    return capital, equity_curve


def calc_metrics(equity_curve, dates):
    """計算績效指標"""
    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]

    total_return = (eq[-1] / eq[0] - 1) * 100
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = ((eq[-1] / eq[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd = drawdown.min() * 100

    # 年化波動率
    annual_vol = np.std(returns) * np.sqrt(252) * 100

    # Sharpe Ratio (假設無風險利率 4%)
    excess_return = np.mean(returns) * 252 - 0.04
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
periods = {
    "近 10 年": df[df.index >= (today - timedelta(days=365 * 10))],
    "近 5 年": df[df.index >= (today - timedelta(days=365 * 5))],
}

# ── 執行回測 ──────────────────────────────────────────────
all_results = {}
all_curves = {}

for period_name, df_period in periods.items():
    df_period = df_period.copy().reset_index(drop=False)
    dates = df_period["Date"].values if "Date" in df_period.columns else df_period.index

    # 使用 df_period 的 index 作為日期（reset_index 後 Date 會變成欄位）
    if "Date" in df_period.columns:
        date_list = pd.to_datetime(df_period["Date"]).tolist()
    else:
        date_list = df_period.index.tolist()

    # Buy & Hold
    bh_capital, bh_curve = run_buy_and_hold(df_period)
    bh_metrics = calc_metrics(bh_curve, date_list)
    bh_metrics["交易次數"] = 0

    # 即時策略
    imm_capital, imm_curve, imm_trades = run_backtest(df_period, "即時", confirm_days=0)
    imm_metrics = calc_metrics(imm_curve, date_list)
    imm_metrics["交易次數"] = imm_trades

    # 3天確認策略
    d3_capital, d3_curve, d3_trades = run_backtest(df_period, "3天確認", confirm_days=3)
    d3_metrics = calc_metrics(d3_curve, date_list)
    d3_metrics["交易次數"] = d3_trades

    # 即時策略(換QQQ)
    imm_q_capital, imm_q_curve, imm_q_trades = run_backtest_switch(df_period, "QQQ_Return", confirm_days=0)
    imm_q_metrics = calc_metrics(imm_q_curve, date_list)
    imm_q_metrics["交易次數"] = imm_q_trades

    # 3天確認策略(換QQQ)
    d3_q_capital, d3_q_curve, d3_q_trades = run_backtest_switch(df_period, "QQQ_Return", confirm_days=3)
    d3_q_metrics = calc_metrics(d3_q_curve, date_list)
    d3_q_metrics["交易次數"] = d3_q_trades

    # 即時策略(換SPMO)
    imm_s_capital, imm_s_curve, imm_s_trades = run_backtest_switch(df_period, "SPMO_Return", confirm_days=0)
    imm_s_metrics = calc_metrics(imm_s_curve, date_list)
    imm_s_metrics["交易次數"] = imm_s_trades

    # 3天確認策略(換SPMO)
    d3_s_capital, d3_s_curve, d3_s_trades = run_backtest_switch(df_period, "SPMO_Return", confirm_days=3)
    d3_s_metrics = calc_metrics(d3_s_curve, date_list)
    d3_s_metrics["交易次數"] = d3_s_trades

    all_results[period_name] = {
        "Buy & Hold": bh_metrics,
        "即時→現金": imm_metrics,
        "3天→現金": d3_metrics,
        "即時→QQQ": imm_q_metrics,
        "3天→QQQ": d3_q_metrics,
        "即時→SPMO": imm_s_metrics,
        "3天→SPMO": d3_s_metrics,
    }
    all_curves[period_name] = {
        "dates": date_list,
        "Buy & Hold": bh_curve,
        "即時→現金": imm_curve,
        "3天→現金": d3_curve,
        "即時→QQQ": imm_q_curve,
        "3天→QQQ": d3_q_curve,
        "即時→SPMO": imm_s_curve,
        "3天→SPMO": d3_s_curve,
    }

# ── 輸出績效表格 ──────────────────────────────────────────
print("=" * 80)
print("               TQQQ 回測結果：QQQ 200 日均線策略")
print("=" * 80)

for period_name, strategies in all_results.items():
    print(f"\n{'─' * 80}")
    print(f"  📊 {period_name}")
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
