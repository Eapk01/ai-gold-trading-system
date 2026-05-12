import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd


mt5.initialize()

from_date = datetime(2026, 4, 21)
to_date = datetime(2026, 5, 1)

deals = mt5.history_deals_get(from_date, to_date)

# Convert to DataFrame
df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())

# Keep only closed trade deals
df = df[df['entry'] == 1]

# Metrics
total_trades = len(df)
wins = len(df[df['profit'] > 0])
losses = len(df[df['profit'] <= 0])

win_rate = (wins / total_trades) * 100 if total_trades else 0
total_profit = df['profit'].sum()

# Summary table
summary = pd.DataFrame({
    "Metric": [
        "Total Trades",
        "Wins",
        "Losses",
        "Win Rate %",
        "Total Profit"
    ],
    "Value": [
        total_trades,
        wins,
        losses,
        round(win_rate, 2),
        round(total_profit, 2)
    ]
})

# Export
with pd.ExcelWriter("mt5_report.xlsx") as writer:
    df.to_excel(writer, sheet_name="Trades", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

print("Report exported.")