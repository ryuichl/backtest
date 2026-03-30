"""
TQQQ 回測：金字塔買入策略（2 層）
策略邏輯：
  - QQQ 跌破 200MA 連續 3 天 → 賣出全部 TQQQ，轉現金
  - QQQ 跌到 2 年線 (504日均線) → 買入第一層倉位 TQQQ
  - QQQ 跌到 3 年線 (756日均線) → 買入剩餘全部倉位 TQQQ
  - QQQ 漲回 200MA 連續 3 天 → 買回全部 TQQQ
對照：Buy & Hold TQQQ、原始 3天→現金 策略
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 下載資料 ──────────────────────────────────────────────
print("正在下載 QQQ、TQQQ、QLD、SPMO 歷史資料...")
tickers = {"QQQ": "QQQ", "TQQQ": "TQQQ", "QLD": "QLD", "SPMO": "SPMO"}
raw = {}
for name, ticker in tickers.items():
    data = yf.download(ticker, start="2010-02-11", end=datetime.today().strftime("%Y-%m-%d"), progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    raw[name] = data["Close"]

df = pd.DataFrame({
    "QQQ_Close": raw["QQQ"],
    "TQQQ_Close": raw["TQQQ"],
    "QLD_Close": raw["QLD"],
    "SPMO_Close": raw["SPMO"],
})
df.dropna(inplace=True)

# ── 計算均線 ──────────────────────────────────────────────
df["QQQ_MA200"] = df["QQQ_Close"].rolling(window=200).mean()
df["QQQ_MA504"] = df["QQQ_Close"].rolling(window=504).mean()   # 2 年線
df["QQQ_MA756"] = df["QQQ_Close"].rolling(window=756).mean()   # 3 年線
df.dropna(inplace=True)

# QQQ 是否在 200MA 之上
df["above_ma200"] = (df["QQQ_Close"] > df["QQQ_MA200"]).astype(int)

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

# QQQ 是否跌破各年線（當天收盤 <= 均線）
df["below_ma504"] = (df["QQQ_Close"] <= df["QQQ_MA504"]).astype(int)
df["below_ma756"] = (df["QQQ_Close"] <= df["QQQ_MA756"]).astype(int)

# 每日報酬率
df["TQQQ_Return"] = df["TQQQ_Close"].pct_change()
df["QQQ_Return"] = df["QQQ_Close"].pct_change()
df["QLD_Return"] = df["QLD_Close"].pct_change()
df["SPMO_Return"] = df["SPMO_Close"].pct_change()
df.dropna(inplace=True)

print(f"資料範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"共 {len(df)} 個交易日\n")


# ── 策略邏輯 ──────────────────────────────────────────────
def run_pyramid_strategy(df_period, ratios=(0.5, 0.5)):
    """
    金字塔買入策略（2 層），可自訂各層買入比例。
    ratios: (2年線比例, 3年線比例)，合計應為 1.0
    比例是以「賣出當下總資產」為基準的絕對比例。
    """
    r_2y, r_3y = ratios
    cash = 0.0
    tqqq_value = 10000.0
    equity_curve = []
    trades = 0
    state = "HOLD_ALL"

    hit_2y = False
    hit_3y = False
    sold_total = 0.0  # 記錄賣出時的總資產，用於計算各層買入金額

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        tqqq_ret = row["TQQQ_Return"]

        if tqqq_value > 0:
            tqqq_value *= (1 + tqqq_ret)

        total = cash + tqqq_value

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        if i == 0:
            equity_curve.append(total)
            continue
        sig = df_period.iloc[i - 1]

        if state == "HOLD_ALL":
            if sig["below_streak"] >= 3:
                cash = total
                tqqq_value = 0.0
                sold_total = total  # 記錄賣出時總資產
                state = "SOLD"
                hit_2y = False
                hit_3y = False
                trades += 1

        elif state == "SOLD":
            if sig["above_streak"] >= 3:
                tqqq_value = total
                cash = 0.0
                state = "HOLD_ALL"
                trades += 1
            elif sig["below_ma504"] == 1 and not hit_2y:
                buy_amount = sold_total * r_2y
                buy_amount = min(buy_amount, cash)
                tqqq_value = buy_amount
                cash = total - buy_amount
                state = "PYRAMID_PARTIAL"
                hit_2y = True
                trades += 1

        elif state == "PYRAMID_PARTIAL":
            total = cash + tqqq_value

            if sig["above_streak"] >= 3:
                tqqq_value = total
                cash = 0.0
                state = "HOLD_ALL"
                trades += 1
            else:
                if sig["below_ma756"] == 1 and not hit_3y:
                    tqqq_value += cash
                    cash = 0.0
                    hit_3y = True
                    trades += 1

        equity_curve.append(cash + tqqq_value)

    return cash + tqqq_value, equity_curve, trades


def run_backtest_switch(df_period, alt_return_col, confirm_days=3):
    """
    輪換策略：跌破 200MA 連續 N 天時換持替代標的
    alt_return_col: 替代標的的報酬率欄位名（如 "QQQ_Return", "SPMO_Return"）
    """
    capital = 10000.0
    holding_tqqq = True
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


def run_buy_and_hold(df_period, return_col="TQQQ_Return"):
    """單純持有指定標的"""
    capital = 10000.0
    equity_curve = []
    for i in range(len(df_period)):
        daily_return = df_period.iloc[i][return_col]
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

    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd = drawdown.min() * 100

    annual_vol = np.std(returns) * np.sqrt(252) * 100

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

    if "Date" in df_period.columns:
        date_list = pd.to_datetime(df_period["Date"]).tolist()
    else:
        date_list = df_period.index.tolist()

    # Buy & Hold TQQQ
    bh_capital, bh_curve = run_buy_and_hold(df_period, "TQQQ_Return")
    bh_metrics = calc_metrics(bh_curve, date_list)
    bh_metrics["交易次數"] = 0

    # Buy & Hold QQQ（原型 ETF 基準）
    bh_qqq_capital, bh_qqq_curve = run_buy_and_hold(df_period, "QQQ_Return")
    bh_qqq_metrics = calc_metrics(bh_qqq_curve, date_list)
    bh_qqq_metrics["交易次數"] = 0

    # Buy & Hold QLD（2x 槓桿 ETF）
    bh_qld_capital, bh_qld_curve = run_buy_and_hold(df_period, "QLD_Return")
    bh_qld_metrics = calc_metrics(bh_qld_curve, date_list)
    bh_qld_metrics["交易次數"] = 0

    # 3天→QQQ 輪換策略
    d3q_capital, d3q_curve, d3q_trades = run_backtest_switch(df_period, "QQQ_Return")
    d3q_metrics = calc_metrics(d3q_curve, date_list)
    d3q_metrics["交易次數"] = d3q_trades

    # 3天→SPMO 輪換策略
    d3s_capital, d3s_curve, d3s_trades = run_backtest_switch(df_period, "SPMO_Return")
    d3s_metrics = calc_metrics(d3s_curve, date_list)
    d3s_metrics["交易次數"] = d3s_trades

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
        "B&H QQQ": bh_qqq_metrics,
        "B&H QLD": bh_qld_metrics,
        "B&H TQQQ": bh_metrics,
        "3天→QQQ": d3q_metrics,
        "3天→SPMO": d3s_metrics,
    }
    curves_dict = {
        "dates": date_list,
        "B&H QQQ": bh_qqq_curve,
        "B&H QLD": bh_qld_curve,
        "B&H TQQQ": bh_curve,
        "3天→QQQ": d3q_curve,
        "3天→SPMO": d3s_curve,
    }

    for config_name, ratios in pyramid_configs.items():
        pyr_capital, pyr_curve, pyr_trades = run_pyramid_strategy(df_period, ratios=ratios)
        pyr_metrics = calc_metrics(pyr_curve, date_list)
        pyr_metrics["交易次數"] = pyr_trades
        results_dict[config_name] = pyr_metrics
        curves_dict[config_name] = pyr_curve

    all_results[period_name] = results_dict
    all_curves[period_name] = curves_dict

# ── 輸出績效表格 ──────────────────────────────────────────
print("=" * 80)
print("     TQQQ 回測：金字塔不同比例 vs 輪換策略")
print("=" * 80)
print("\n策略說明：")
print("  輪換：QQQ 跌破 200MA 3天 → 換持 QQQ/SPMO，漲回 3天 → 換回 TQQQ")
print("  金字塔 (X/Y)：QQQ 跌破 200MA 3天賣出 →")
print("    跌到 2年線 買 X%，跌到 3年線 買剩餘全部")
print("    漲回 200MA 3天 → 全部買回 TQQQ")

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
