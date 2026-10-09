# TradingView indicator: Turn Ladder + MA Score

`turn_ladder.pine` brings the formations page's tools to your own TradingView charts with live data.

## Install
1. In TradingView open a chart, then **Pine Editor** (bottom panel).
2. Delete the template code, paste the whole contents of `turn_ladder.pine`, click **Save**, then **Add to chart**.
3. Settings (gear icon): swing strength, which timeframes the ladder uses (default 5m, 15m, 1h, 4h, 1D), and which parts to show.

## What you see
- EMA9 / EMA21 / SMA50 / SMA200 and a background tint from the MA score (green ≥ +0.5, red ≤ -0.5).
- **T↑ / T↓**: a turn confirmed on the chart timeframe: after a swing low (high), a close above the swing high (below the swing low) that came before it.
- **Ladder table** (top right): for each timeframe its trend, the last confirmed turn and when, and the close it needs for the next turn the other way.
- **CAP / CLX**: volume ≥ 2× the 20-bar average on a wide bar at a 15-bar low after a slide (selling capitulation) or at a 15-bar high after a rally (buying climax).

## Alerts
Alerts → Create alert → Condition: *Turn Ladder + MA Score* → pick one of:
turn up / down on the chart timeframe, turn up / down on 5m + 15m + 1h together, selling capitulation, buying climax.

Swings use pivots with N bars on each side, so a swing (and a turn) is only known N bars after it happens; that delay is the price of confirmation. Not investment advice.
