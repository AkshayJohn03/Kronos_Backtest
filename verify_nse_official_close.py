import os
import pandas as pd
import numpy as np

base_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
audit_dir = os.path.join(base_dir, "plots", "live_audit_post_market")

print("=" * 80)
print("AUDITING PREDICTIONS AGAINST NSE OFFICIAL CLOSING PRICE (24,383 INR)")
print("=" * 80)

# 2:15 PM Trigger price: 24,251.55 INR
price_at_215pm = 24251.55
official_nse_close = 24383.00
actual_nse_move = official_nse_close - price_at_215pm

predictions = [
    {
        'timeframe': '5 Minute',
        'price_at_215pm': price_at_215pm,
        'predicted_close': 24363.01,
        'official_nse_close': official_nse_close,
        'predicted_move_pts': 24363.01 - price_at_215pm,
        'actual_move_pts': actual_nse_move,
        'mae_pts': abs(official_nse_close - 24363.01),
        'is_win': True,
        'signal': 'BUY CALL (BULLISH)'
    },
    {
        'timeframe': '1 Minute',
        'price_at_215pm': price_at_215pm,
        'predicted_close': 24288.59,
        'official_nse_close': official_nse_close,
        'predicted_move_pts': 24288.59 - price_at_215pm,
        'actual_move_pts': actual_nse_move,
        'mae_pts': abs(official_nse_close - 24288.59),
        'is_win': True,
        'signal': 'BUY CALL (BULLISH)'
    },
    {
        'timeframe': '15 Minute',
        'price_at_215pm': price_at_215pm,
        'predicted_close': 24282.66,
        'official_nse_close': official_nse_close,
        'predicted_move_pts': 24282.66 - price_at_215pm,
        'actual_move_pts': actual_nse_move,
        'mae_pts': abs(official_nse_close - 24282.66),
        'is_win': True,
        'signal': 'BUY CALL (BULLISH)'
    }
]

df_official = pd.DataFrame(predictions)
df_official.to_csv(os.path.join(audit_dir, "nse_official_close_audit.csv"), index=False)

print(df_official.to_string(index=False))
print("\nAUDIT AGAINST OFFICIAL NSE CLOSE (24,383 INR) COMPLETED!")
