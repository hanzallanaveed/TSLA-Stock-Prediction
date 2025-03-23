# Tesla Stock Trading Simulation

## Project Overview
This project implements a trading strategy for Tesla (TSLA) stock during the simulation period of March 24-28, 2025. The strategy is based on historical analysis of Tesla stock patterns, particularly focusing on March performance patterns.

## Features
- Analyzes historical Tesla stock data to identify optimal trading patterns
- Determines the most efficient buy/sell timing to maximize returns
- Calculates projected performance including transaction costs
- Can check for actual market data once simulation dates arrive

## Trading Strategy
Based on extensive analysis of historical data patterns, the optimal strategy for the March 24-28, 2025 period is:

```
- MONDAY:    BUY: $10,000
- TUESDAY:   HOLD
- WEDNESDAY: HOLD
- THURSDAY:  SELL: all shares
- FRIDAY:    HOLD
```

This strategy was selected after analyzing various approaches and evaluating their historical performance, transaction costs, and risk profiles.

## Projected Performance
The model projects the following performance for the strategy:

```
💰 PROJECTED PERFORMANCE (Based on historical data):
Starting Capital: $10,000.00
Projected Final Value: $9,824.92
Projected Return: -1.75%
Estimated Transaction Costs: $199.24

📈 DAILY PERFORMANCE PROJECTION:
-----------------------------------------------------------------
Day        Action   Cash            Shares     Portfolio Value
-----------------------------------------------------------------
Monday     BUY      $0.00           65.19      $9,811.87
Tuesday    HOLD     $0.00           65.19      $10,761.03
Wednesday  HOLD     $0.00           65.19      $10,939.00
Thursday   SELL     $9,824.92       0.00       $9,824.92
Friday     HOLD     $9,824.92       0.00       $9,824.92
-----------------------------------------------------------------
```

While the projection shows a slight negative return based on broad historical patterns, analysis of specific March periods (like March 21-24, 2022) showed much stronger positive returns. The actual performance will depend on Tesla's market performance during the simulation week.

## How to Use
1. Ensure you have the required dependencies installed:
   ```
   pip install pandas numpy matplotlib yfinance
   ```

2. Run the main script:
   ```
   python teslaStockPrediction.py
   ```

3. The script will output the recommended trading strategy and projected performance.

4. For the actual simulation, submit the daily trading decisions before 9:00 AM EST each day from March 24-28, 2025, following the format:
   - Action: Buy/Hold/Sell
   - Amount: Percentage of funds (for Buy) or shares (for Sell)

## Project Structure
- `teslaStockPrediction.py`: Main script containing the trading strategy and analysis
- `TSLA.csv`: Historical Tesla stock data used for analysis

## Methodology
The strategy was developed through several analytical approaches:
1. Analysis of day-of-week patterns in Tesla stock
2. Specific focus on March trading patterns
3. Testing of various trading strategies with different position sizes
4. Optimization for balance between returns and transaction costs

The final strategy represents the approach that historically showed the most promising balance of potential returns, risk management, and cost efficiency.

## Notes
- All projections are based on historical data and actual results may vary
- The 1% transaction fee is factored into all calculations
- The strategy assumes execution of trades at 10:00 AM EST each day
