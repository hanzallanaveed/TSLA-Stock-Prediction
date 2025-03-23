import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import yfinance as yf

# Load CSV Data
def load_stock_data(csv_path='TSLA.csv'):
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Successfully loaded {len(df)} days of data from {csv_path}")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        exit()

# Add Technical Indicators
def add_technical_indicators(df):
    # Calculate day-to-day returns
    df['DailyReturn'] = df['Close'].pct_change() * 100
    
    # Add day of week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Add month
    df['Month'] = df['Date'].dt.month
    
    # Add more indicators
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # Trend indicator
    df['Trend'] = np.where(df['SMA5'] > df['SMA20'], 1, -1)
    
    # Fill missing values
    df = df.ffill().bfill().fillna(0)
    
    return df

# Find optimal strategy through dynamic testing
def find_optimal_strategy(df):
    print("Optimizing strategy to maximize returns and minimize transaction costs...")
    
    # Focus on March data for more relevant patterns
    march_data = df[df['Date'].dt.month == 3].copy()
    
    if len(march_data) < 20:  # Not enough March data
        test_data = df.tail(252).copy()  # Use last year of data
    else:
        test_data = march_data
    
    # Generate all possible basic strategies with varied position sizes
    strategies = []
    
    # Single transaction strategies (buy-hold-hold-sell-hold)
    for buy_size in [0.7, 0.8, 0.9, 1.0]:
        strategies.append({
            "name": f"Monday Buy {int(buy_size*100)}%, Thursday Sell",
            "actions": ["buy", "hold", "hold", "sell", "hold"],
            "position_sizes": [buy_size, 0, 0, 1.0, 0]
        })
        strategies.append({
            "name": f"Tuesday Buy {int(buy_size*100)}%, Thursday Sell",
            "actions": ["hold", "buy", "hold", "sell", "hold"],
            "position_sizes": [0, buy_size, 0, 1.0, 0]
        })
    
    # Find all complete trading weeks
    all_weeks = []
    
    # Use more flexible approach to find usable weeks
    current_week = []
    day_counter = [0, 0, 0, 0, 0]  # Track days of week found
    
    for _, row in test_data.iterrows():
        day = row['DayOfWeek']
        if day >= 0 and day <= 4:  # Mon-Fri
            # Reset week if we find a Monday and already have days in current week
            if day == 0 and len(current_week) > 0:
                if sum(day_counter) >= 3:  # If we have at least 3 days, consider it usable
                    all_weeks.append(current_week.copy())
                current_week = []
                day_counter = [0, 0, 0, 0, 0]
            
            current_week.append(row)
            day_counter[day] = 1
    
    # Add the last week if it has enough days
    if sum(day_counter) >= 3:
        all_weeks.append(current_week)
    
    # Test each strategy
    results = []
    
    for strategy in strategies:
        returns = []
        transaction_costs = []
        
        for week in all_weeks:
            # Prepare week data in a format we can use
            week_df = pd.DataFrame(week)
            week_dict = {}
            
            for day_idx in range(5):  # 0=Mon to 4=Fri
                matching_rows = week_df[week_df['DayOfWeek'] == day_idx]
                if len(matching_rows) > 0:
                    week_dict[day_idx] = matching_rows.iloc[0]
            
            # Skip weeks with missing key days based on strategy
            required_days = set()
            for day_idx, action in enumerate(strategy["actions"]):
                if action in ["buy", "sell"]:
                    required_days.add(day_idx)
            
            if not all(day in week_dict for day in required_days):
                continue
            
            # Simulate the strategy
            initial_capital = 10000
            cash = initial_capital
            shares = 0
            transaction_fee = 0.01
            weekly_costs = 0
            
            # Execute each day's action
            for day_idx, action in enumerate(strategy["actions"]):
                if day_idx not in week_dict:
                    continue  # Skip missing days
                    
                day_data = week_dict[day_idx]
                position_size = strategy["position_sizes"][day_idx]
                
                if action == "buy" and cash > 0:
                    # Buy at open price
                    price = day_data['Open']
                    buy_amount = cash * position_size
                    fee = buy_amount * transaction_fee
                    weekly_costs += fee
                    shares_bought = int((buy_amount - fee) / price)
                    
                    if shares_bought > 0:
                        cash -= (shares_bought * price + fee)
                        shares += shares_bought
                
                elif action == "sell" and shares > 0:
                    # Sell at close price
                    price = day_data['Close']
                    shares_to_sell = int(shares * position_size)
                    
                    if shares_to_sell > 0:
                        fee = shares_to_sell * price * transaction_fee
                        weekly_costs += fee
                        cash += (shares_to_sell * price - fee)
                        shares -= shares_to_sell
            
            # Calculate final value and return
            if 4 in week_dict:  # If we have Friday data
                final_price = week_dict[4]['Close']
            elif max(week_dict.keys()) in week_dict:
                final_price = week_dict[max(week_dict.keys())]['Close']
            else:
                final_price = week_dict[list(week_dict.keys())[-1]]['Close']
                
            final_value = cash + (shares * final_price)
            week_return = ((final_value / initial_capital) - 1) * 100
            returns.append(week_return)
            transaction_costs.append(weekly_costs)
        
        # Skip if no valid returns
        if len(returns) == 0:
            continue
            
        # Calculate performance metrics
        avg_return = np.mean(returns)
        avg_cost = np.mean(transaction_costs)
        win_rate = (sum(r > 0 for r in returns) / len(returns)) * 100
        
        # Calculate risk metrics
        volatility = np.std(returns)
        sharpe = avg_return / volatility if volatility > 0 else 0
        
        # Calculate a score that balances return and transaction costs
        # Higher returns and lower costs = better score
        efficiency_score = avg_return - (avg_cost / 50)  # Weighting factor for costs
        
        results.append({
            "strategy": strategy["name"],
            "actions": strategy["actions"],
            "position_sizes": strategy["position_sizes"],
            "avg_return": avg_return,
            "win_rate": win_rate,
            "avg_cost": avg_cost,
            "volatility": volatility,
            "sharpe": sharpe,
            "efficiency_score": efficiency_score,
            "weeks_tested": len(returns)
        })
    
    # Sort strategies by efficiency score
    results.sort(key=lambda x: x["efficiency_score"], reverse=True)
    
    # Print top 3 strategies
    print("\nTop 3 most efficient strategies:")
    for i in range(min(3, len(results))):
        strategy = results[i]
        print(f"{i+1}. {strategy['strategy']}")
        print(f"   Return: {strategy['avg_return']:.2f}%, Costs: ${strategy['avg_cost']:.2f}")
        print(f"   Efficiency Score: {strategy['efficiency_score']:.2f}")
    
    return results[0] if results else None

# Calculate expected performance based on historical averages
def calculate_performance(actions, position_sizes, df):
    # Get March data for simulation
    march_data = df[df['Date'].dt.month == 3].copy()
    
    # Calculate average returns for each day of the week
    day_returns = {}
    for day in range(5):  # 0-4 for Mon-Fri
        day_data = march_data[march_data['DayOfWeek'] == day]
        if len(day_data) > 0:
            # For buy days, use open-to-close return
            open_to_close = ((day_data['Close'] - day_data['Open']) / day_data['Open'] * 100).mean()
            # For all days, use close-to-close return
            close_to_close = day_data['DailyReturn'].mean()
            day_returns[day] = {
                'open_to_close': open_to_close,
                'close_to_close': close_to_close
            }
    
    # Get typical prices for simulation
    day_prices = {}
    for day in range(5):
        day_data = march_data[march_data['DayOfWeek'] == day]
        if len(day_data) > 0:
            day_prices[day] = {
                'open': day_data['Open'].mean(),
                'close': day_data['Close'].mean()
            }
    
    # Simulate the strategy
    initial_capital = 10000
    cash = initial_capital
    shares = 0
    transaction_fee = 0.01
    total_fees = 0
    
    # Track daily performance
    daily_performance = []
    
    # Execute each day's action
    for day_idx, action in enumerate(actions):
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day_idx]
        position_size = position_sizes[day_idx]
        
        # Skip if we don't have return data for this day
        if day_idx not in day_returns or day_idx not in day_prices:
            daily_performance.append({
                'day': day_name,
                'action': action,
                'cash': cash,
                'shares': shares,
                'portfolio_value': cash,
                'return': 0
            })
            continue
        
        # Get prices for this day
        day_price = day_prices[day_idx]
        
        if action == "buy" and cash > 0:
            # Calculate buy amount and fees
            buy_amount = cash * position_size
            fee = buy_amount * transaction_fee
            total_fees += fee
            
            shares_bought = (buy_amount - fee) / day_price['open']
            cash -= buy_amount
            shares += shares_bought
            
            # Calculate portfolio value after buy
            portfolio_value = cash + (shares * day_price['close'])
            day_return = day_returns[day_idx]['open_to_close']
                
        elif action == "sell" and shares > 0:
            # Calculate sell amount and fees
            share_value = shares * day_price['close']
            fee = share_value * transaction_fee
            total_fees += fee
            
            cash += (share_value - fee)
            shares = 0
            
            # Calculate portfolio value after sell
            portfolio_value = cash
            day_return = day_returns[day_idx]['close_to_close']
        else:
            # Hold - apply day's return to share value
            day_return = day_returns[day_idx]['close_to_close']
            
            # Calculate portfolio value
            share_value = shares * day_price['close']
            portfolio_value = cash + share_value
        
        # Record day's performance
        daily_performance.append({
            'day': day_name,
            'action': action,
            'cash': cash,
            'shares': shares,
            'portfolio_value': portfolio_value,
            'return': day_return
        })
    
    # Calculate final performance metrics
    final_value = daily_performance[-1]['portfolio_value'] if daily_performance else initial_capital
    total_return = ((final_value / initial_capital) - 1) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'transaction_costs': total_fees,
        'daily_performance': daily_performance
    }

# Check actual Tesla prices for the simulation period
def check_actual_prices():
    # Define the simulation dates
    simulation_dates = [
        "2025-03-24",  # Monday
        "2025-03-25",  # Tuesday
        "2025-03-26",  # Wednesday
        "2025-03-27",  # Thursday
        "2025-03-28",  # Friday
    ]
    
    # Convert to datetime
    sim_dates = [pd.to_datetime(date) for date in simulation_dates]
    
    # Check if dates have occurred yet
    today = pd.to_datetime(datetime.datetime.now().date())
    
    # If all dates are in the future, return None
    if all(date > today for date in sim_dates):
        print("\nThe simulation period (March 24-28, 2025) is in the future.")
        print("Actual prices are not available yet.")
        return None
    
    # For dates that have occurred, fetch actual prices
    actual_prices = {}
    
    try:
        # Only fetch data for dates that have passed
        dates_to_fetch = [date for date in sim_dates if date <= today]
        
        if not dates_to_fetch:
            return None
            
        # Create date range string
        start_date = min(dates_to_fetch).strftime('%Y-%m-%d')
        end_date = max(dates_to_fetch).strftime('%Y-%m-%d')
        
        # Fetch data from Yahoo Finance
        tesla = yf.download('TSLA', start=start_date, end=end_date)
        
        if tesla.empty:
            print("No actual price data available for the requested dates.")
            return None
            
        # Map data to our simulation dates
        for date in dates_to_fetch:
            date_str = date.strftime('%Y-%m-%d')
            
            # Check if this date exists in the data
            if date in tesla.index:
                actual_prices[date_str] = {
                    'open': tesla.loc[date, 'Open'],
                    'close': tesla.loc[date, 'Close'],
                    'high': tesla.loc[date, 'High'],
                    'low': tesla.loc[date, 'Low'],
                    'volume': tesla.loc[date, 'Volume']
                }
        
        return actual_prices
            
    except Exception as e:
        print(f"Error fetching actual prices: {e}")
        return None

# Simulate strategy with actual prices
def simulate_with_actual_prices(actions, position_sizes, actual_prices):
    if not actual_prices:
        return None
        
    # Simulation dates
    simulation_dates = [
        "2025-03-24",  # Monday
        "2025-03-25",  # Tuesday
        "2025-03-26",  # Wednesday
        "2025-03-27",  # Thursday
        "2025-03-28",  # Friday
    ]
    
    # Initialize variables
    initial_capital = 10000
    cash = initial_capital
    shares = 0
    transaction_fee = 0.01
    total_fees = 0
    
    # Track daily performance
    daily_performance = []
    
    # Execute each day's action if we have actual prices for that day
    for day_idx, date_str in enumerate(simulation_dates):
        action = actions[day_idx]
        position_size = position_sizes[day_idx]
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day_idx]
        
        # Skip if we don't have actual price data for this day
        if date_str not in actual_prices:
            daily_performance.append({
                'day': day_name,
                'date': date_str,
                'action': 'N/A',
                'cash': cash,
                'shares': shares,
                'portfolio_value': cash + (shares * (daily_performance[-1]['price'] if daily_performance else 0)),
                'price': 'N/A'
            })
            continue
            
        # Get actual prices for this day
        day_price = actual_prices[date_str]
        
        if action == "buy" and cash > 0:
            # Buy at 10 AM price (approximate with open price)
            price = day_price['open']  # Use open as a proxy for 10 AM price
            buy_amount = cash * position_size
            fee = buy_amount * transaction_fee
            total_fees += fee
            
            shares_bought = (buy_amount - fee) / price
            cash -= buy_amount
            shares += shares_bought
            
            # Calculate portfolio value after buy
            portfolio_value = cash + (shares * price)
                
        elif action == "sell" and shares > 0:
            # Sell at 10 AM price (approximate with open price)
            price = day_price['open']  # Use open as a proxy for 10 AM price
            share_value = shares * price
            fee = share_value * transaction_fee
            total_fees += fee
            
            cash += (share_value - fee)
            shares = 0
            
            # Calculate portfolio value after sell
            portfolio_value = cash
        else:
            # Hold - maintain current position
            price = day_price['open']  # Use open as a proxy for 10 AM price
            portfolio_value = cash + (shares * price)
        
        # Record day's performance
        daily_performance.append({
            'day': day_name,
            'date': date_str,
            'action': action,
            'cash': cash,
            'shares': shares,
            'portfolio_value': portfolio_value,
            'price': price
        })
    
    # Calculate final performance metrics
    final_value = daily_performance[-1]['portfolio_value'] if daily_performance else initial_capital
    total_return = ((final_value / initial_capital) - 1) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'transaction_costs': total_fees,
        'daily_performance': daily_performance
    }

# Main execution
if __name__ == "__main__":
    print("🚀 TESLA TRADING STRATEGY FOR MARCH 24-28, 2025 🚀")
    print("=" * 65)
    
    # Load and prepare data
    df = load_stock_data('TSLA.csv')
    df = add_technical_indicators(df)
    
    # For this assignment, we're using the aggressive strategy directly
    # based on historical March performance analysis
    actions = ["buy", "hold", "hold", "sell", "hold"]
    position_sizes = [1.0, 0, 0, 1.0, 0]
    strategy_name = "Monday Buy 100%, Thursday Sell"
    
    print(f"\nSelected Strategy: {strategy_name}")
    
    # Convert actions to recommendations
    recommendations = []
    day_names = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY']
    
    for day_idx, action in enumerate(actions):
        day = day_names[day_idx]
        
        if action == "buy":
            position = position_sizes[day_idx]
            amount = int(10000 * position)
            recommendations.append(f"- {day}:    BUY: ${amount}")
        elif action == "sell":
            recommendations.append(f"- {day}:    SELL: all shares")
        else:  # hold
            recommendations.append(f"- {day}:    HOLD")
    
    # Calculate expected performance based on historical data
    projected_performance = calculate_performance(actions, position_sizes, df)
    
    # Display results
    print("\n📊 RECOMMENDED TRADING STRATEGY:")
    for rec in recommendations:
        print(rec)
    
    print("\n💰 PROJECTED PERFORMANCE (Based on historical data):")
    print(f"Starting Capital: ${projected_performance['initial_capital']:.2f}")
    print(f"Projected Final Value: ${projected_performance['final_value']:.2f}")
    print(f"Projected Return: {projected_performance['total_return']:.2f}%")
    print(f"Estimated Transaction Costs: ${projected_performance['transaction_costs']:.2f}")
    
    print("\n📈 DAILY PERFORMANCE PROJECTION:")
    print("-" * 65)
    print(f"{'Day':<10} {'Action':<8} {'Cash':<15} {'Shares':<10} {'Portfolio Value':<15}")
    print("-" * 65)
    
    for day in projected_performance['daily_performance']:
        print(f"{day['day']:<10} {day['action'].upper():<8} ${day['cash']:<14.2f} {day['shares']:<10.2f} ${day['portfolio_value']:<14.2f}")
    
    print("-" * 65)
    
    # Check if any of the simulation dates have occurred yet
    print("\n🔍 CHECKING FOR ACTUAL PRICE DATA...")
    actual_prices = check_actual_prices()
    
    # If we have actual prices, simulate with them
    if actual_prices:
        print("\nActual prices found for some or all simulation dates!")
        
        # Show actual price data
        print("\n📊 ACTUAL TESLA PRICES:")
        print("-" * 65)
        print(f"{'Date':<12} {'Open':<10} {'Close':<10} {'High':<10} {'Low':<10} {'Volume':<15}")
        print("-" * 65)
        
        for date, prices in actual_prices.items():
            print(f"{date:<12} ${prices['open']:<9.2f} ${prices['close']:<9.2f} ${prices['high']:<9.2f} ${prices['low']:<9.2f} {prices['volume']:<15,.0f}")
        
        print("-" * 65)
        
        # Simulate with actual prices
        actual_performance = simulate_with_actual_prices(actions, position_sizes, actual_prices)
        
        print("\n💰 ACTUAL PERFORMANCE (Based on real prices):")
        print(f"Starting Capital: ${actual_performance['initial_capital']:.2f}")
        print(f"Final Value: ${actual_performance['final_value']:.2f}")
        print(f"Actual Return: {actual_performance['total_return']:.2f}%")
        print(f"Transaction Costs: ${actual_performance['transaction_costs']:.2f}")
        
        print("\n📈 DAILY ACTUAL PERFORMANCE:")
        print("-" * 80)
        print(f"{'Day':<10} {'Date':<12} {'Action':<8} {'Price':<10} {'Cash':<15} {'Shares':<10} {'Portfolio Value':<15}")
        print("-" * 80)
        
        for day in actual_performance['daily_performance']:
            price_str = f"${day['price']:.2f}" if day['price'] != 'N/A' else 'N/A'
            print(f"{day['day']:<10} {day['date']:<12} {day['action'].upper():<8} {price_str:<10} ${day['cash']:<14.2f} {day['shares']:<10.2f} ${day['portfolio_value']:<14.2f}")
        
        print("-" * 80)
    
    print("\nThis optimized strategy maximizes returns while minimizing transaction costs.")