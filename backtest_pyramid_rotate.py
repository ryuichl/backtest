"""
TQQQ 回測：金字塔買入策略 — 跌破 200MA 換持替代標的（非現金）
策略邏輯：
  - QQQ 跌破 200MA 連續 3 天 → 賣出全部 TQQQ，換持替代標的（SPMO 或 QQQ）
  - QQQ 跌到 2 年線 (504日均線) → 賣出部分替代標的，買入第一層 TQQQ
  - QQQ 跌到 3 年線 (756日均線) → 賣出全部替代標的，買入剩餘 TQQQ
  - QQQ 漲回 200MA 連續 3 天 → 賣出替代標的，全部買回 TQQQ
對照：原始金字塔（換現金）、Buy & Hold TQQQ
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
print("正在下載 QQQ、TQQQ、SPMO 歷史資料...")
tickers = {"QQQ": "QQQ", "TQQQ": "TQQQ", "SPMO": "SPMO"}
raw = {}
for name, ticker in tickers.items():
    data = yf.download(ticker, start="2010-02-11", end=datetime.today().strftime("%Y-%m-%d"), progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    raw[name] = data["Close"]

df = pd.DataFrame({
    "QQQ_Close": raw["QQQ"],
    "TQQQ_Close": raw["TQQQ"],
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

# QQQ 是否跌破各年線
df["below_ma504"] = (df["QQQ_Close"] <= df["QQQ_MA504"]).astype(int)
df["below_ma756"] = (df["QQQ_Close"] <= df["QQQ_MA756"]).astype(int)

# 每日報酬率
df["TQQQ_Return"] = df["TQQQ_Close"].pct_change()
df["QQQ_Return"] = df["QQQ_Close"].pct_change()
df["SPMO_Return"] = df["SPMO_Close"].pct_change()
df.dropna(inplace=True)

print(f"資料範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"共 {len(df)} 個交易日\n")


# ── 策略邏輯 ──────────────────────────────────────────────
def run_pyramid_cash(df_period, ratios=(0.5, 0.5)):
    """
    原始金字塔策略：跌破 200MA → 換現金，等跌到年線分批買回。
    """
    r_2y, r_3y = ratios
    cash = 0.0
    tqqq_value = 10000.0
    equity_curve = []
    trades = 0
    state = "HOLD_ALL"
    hit_2y = False
    hit_3y = False
    sold_total = 0.0

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
                sold_total = total
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


def run_pyramid_rotate(df_period, alt_return_col, ratios=(0.5, 0.5)):
    """
    金字塔＋輪換策略：跌破 200MA → 換持替代標的（而非現金），
    等跌到年線再分批賣出替代標的、買回 TQQQ。

    alt_return_col: 替代標的報酬率欄位（"QQQ_Return" 或 "SPMO_Return"）
    ratios: (2年線買入比例, 3年線買入比例)
    """
    r_2y, r_3y = ratios
    alt_value = 0.0      # 替代標的持倉價值
    tqqq_value = 10000.0  # TQQQ 持倉價值
    equity_curve = []
    trades = 0
    state = "HOLD_ALL"   # HOLD_ALL / HOLD_ALT / PYRAMID_PARTIAL
    hit_2y = False
    hit_3y = False
    sold_total = 0.0

    for i in range(len(df_period)):
        row = df_period.iloc[i]
        tqqq_ret = row["TQQQ_Return"]
        alt_ret = row[alt_return_col]

        # 更新持倉價值
        if tqqq_value > 0:
            tqqq_value *= (1 + tqqq_ret)
        if alt_value > 0:
            alt_value *= (1 + alt_ret)

        total = alt_value + tqqq_value

        # 用前一天的訊號決定今天的操作（避免 same-bar execution）
        if i == 0:
            equity_curve.append(total)
            continue
        sig = df_period.iloc[i - 1]

        if state == "HOLD_ALL":
            # 跌破 200MA 連續 3 天 → 全部換持替代標的
            if sig["below_streak"] >= 3:
                alt_value = total
                tqqq_value = 0.0
                sold_total = total
                state = "HOLD_ALT"
                hit_2y = False
                hit_3y = False
                trades += 1

        elif state == "HOLD_ALT":
            # 漲回 200MA 連續 3 天 → 全部換回 TQQQ
            if sig["above_streak"] >= 3:
                tqqq_value = total
                alt_value = 0.0
                state = "HOLD_ALL"
                trades += 1
            # 跌到 2 年線 → 賣出部分替代標的，買入第一層 TQQQ
            elif sig["below_ma504"] == 1 and not hit_2y:
                buy_amount = sold_total * r_2y
                buy_amount = min(buy_amount, alt_value)
                tqqq_value = buy_amount
                alt_value = total - buy_amount
                state = "PYRAMID_PARTIAL"
                hit_2y = True
                trades += 1

        elif state == "PYRAMID_PARTIAL":
            total = alt_value + tqqq_value

            # 漲回 200MA 連續 3 天 → 全部換回 TQQQ
            if sig["above_streak"] >= 3:
                tqqq_value = total
                alt_value = 0.0
                state = "HOLD_ALL"
                trades += 1
            else:
                # 跌到 3 年線 → 賣出全部替代標的，買入剩餘 TQQQ
                if sig["below_ma756"] == 1 and not hit_3y:
                    tqqq_value += alt_value
                    alt_value = 0.0
                    hit_3y = True
                    trades += 1

        equity_curve.append(alt_value + tqqq_value)

    return alt_value + tqqq_value, equity_curve, trades


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

pyramid_ratios = (0.50, 0.50)  # 2年線 50% / 3年線 50%

for period_name, df_period in periods.items():
    df_period = df_period.copy().reset_index(drop=False)

    if "Date" in df_period.columns:
        date_list = pd.to_datetime(df_period["Date"]).tolist()
    else:
        date_list = df_period.index.tolist()

    results_dict = {}

    # Buy & Hold TQQQ
    bh_capital, bh_curve = run_buy_and_hold(df_period)
    bh_metrics = calc_metrics(bh_curve, date_list)
    bh_metrics["交易次數"] = 0
    results_dict["Buy&Hold"] = bh_metrics

    # 原始金字塔（換現金）
    cash_capital, cash_curve, cash_trades = run_pyramid_cash(df_period, ratios=pyramid_ratios)
    cash_metrics = calc_metrics(cash_curve, date_list)
    cash_metrics["交易次數"] = cash_trades
    results_dict["金字塔→現金"] = cash_metrics

    # 金字塔＋換 QQQ
    qqq_capital, qqq_curve, qqq_trades = run_pyramid_rotate(df_period, "QQQ_Return", ratios=pyramid_ratios)
    qqq_metrics = calc_metrics(qqq_curve, date_list)
    qqq_metrics["交易次數"] = qqq_trades
    results_dict["金字塔→QQQ"] = qqq_metrics

    # 金字塔＋換 SPMO
    spmo_capital, spmo_curve, spmo_trades = run_pyramid_rotate(df_period, "SPMO_Return", ratios=pyramid_ratios)
    spmo_metrics = calc_metrics(spmo_curve, date_list)
    spmo_metrics["交易次數"] = spmo_trades
    results_dict["金字塔→SPMO"] = spmo_metrics

    all_results[period_name] = results_dict

# ── 輸出績效表格 ──────────────────────────────────────────
print("=" * 80)
print("     TQQQ 回測：金字塔策略 — 跌破200MA換持替代標的 vs 換現金")
print("=" * 80)
print("\n策略說明：")
print("  金字塔→現金：跌破 200MA 3天 → 賣出持現金")
print("  金字塔→QQQ ：跌破 200MA 3天 → 換持 QQQ（非現金）")
print("  金字塔→SPMO：跌破 200MA 3天 → 換持 SPMO（非現金）")
print(f"  金字塔比例：2年線買 {pyramid_ratios[0]:.0%} / 3年線買 {pyramid_ratios[1]:.0%}")
print("  觸及年線時：賣出替代標的（或現金），分批買回 TQQQ")
print("  漲回 200MA 3天 → 全部換回 TQQQ")

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
