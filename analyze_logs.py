import pandas as pd
import matplotlib.pyplot as plt

def analyze():
    try:
        df = pd.read_csv('stats/trading_log.csv')
    except FileNotFoundError:
        print("trading_log.csv not found. Run the live strategy to generate some trades first!")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    print("=== Trading Statistics ===")
    print(f"Total Trades Executed: {len(df)}")
    
    if len(df) > 0:
        initial_balance = df['balance'].iloc[0]
        final_balance = df['balance'].iloc[-1]
        pnl = final_balance - initial_balance
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance:   ${final_balance:.2f}")
        print(f"Net PnL:         ${pnl:.2f}")

    # Plot Balance and Prediction Over Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Balance
    ax1.plot(df.index, df['balance'], label='Portfolio Balance (USDT)', color='green')
    ax1.set_title('Live Portfolio Balance')
    ax1.set_ylabel('Balance (USDT)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Prediction or Execution Price
    if 'prediction' in df.columns:
        ax2.plot(df.index, df['prediction'], label='Model Prediction', color='blue', marker='o', linestyle='None')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Model Prediction (y_hat) at Execution')
        ax2.set_ylabel('Prediction (y_hat)')
    else:
        ax2.plot(df.index, df['price'], label='Execution Price', color='orange')
        ax2.set_title('Execution Price')
        ax2.set_ylabel('Price')
        
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('stats/balance_chart.png')
    print("\nSaved balance chart to stats/balance_chart.png")

if __name__ == "__main__":
    analyze()
