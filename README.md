# Leveraged ETF Backtest — 200MA Strategy

槓桿 ETF 200 日均線策略回測工具，涵蓋美股 TQQQ 與台股 00631L。

## 策略概述

以標的指數的 200 日移動平均線 (200MA) 作為進出場訊號，當指數跌破均線時賣出槓桿 ETF 轉持替代標的或現金，漲回時買回，藉此降低槓桿 ETF 的最大回撤。

## 回測腳本

| 腳本 | 說明 |
|------|------|
| `backtest.py` | TQQQ 基本策略 — QQQ 200MA 訊號，替代標的：現金 / QQQ / SPMO |
| `backtest_pyramid.py` | TQQQ 金字塔買入 — 跌至 2 年線、3 年線分批進場 |
| `backtest_pyramid_rotate.py` | TQQQ 金字塔 + 輪換 — 跌破時換持 SPMO/QQQ 而非現金 |
| `backtest_tw.py` | 00631L — 以 0050 200MA 為訊號 |
| `backtest_tw_ix0001.py` | 00631L — 以加權指數 200MA 為訊號 |
| `backtest_tw_pyramid.py` | 00631L 金字塔買入 — 加權指數均線訊號 |

## 使用方式

```bash
pip install yfinance pandas numpy tabulate
python backtest.py
```

## 回測結果

詳見 [reports.md](reports.md)。
