import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from collections import deque
from scipy.stats import linregress
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class StockAnalysisBot:
    def __init__(self):
        self.LOOKBACK = 30
        self.HISTORICAL_DAYS = 60
        self.MIN_PROFIT = 0.015
        self.STOP_LOSS = -0.02
        self.TREND_WINDOW = 20
        self.REVERSAL_THRESHOLD = 0.01
        self.RISK_REWARD_RATIO = 2  # Minimum 2:1 reward-to-risk
        self.opportunity_history = deque(maxlen=20)

    def get_data(self, symbol, timeframe="5m", limit=None, days=None):
        """Fetch historical price data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            if days:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days + 3)
                df = ticker.history(start=start_date, end=end_date, interval="1d" if timeframe == "1d" else "5m")
            else:
                df = ticker.history(period="7d", interval="5m")
                if limit:
                    df = df.tail(limit)
                    
            if df.empty or "Close" not in df.columns:
                print(f"{datetime.now()}: {symbol} - Empty or invalid data.")
                return None
                
            df = df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open"})
            return df
        except Exception as e:
            print(f"{datetime.now()}: {symbol} - Error fetching data: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate advanced technical indicators."""
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        return df

    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement levels."""
        recent_high = df['high'].tail(60).max()
        recent_low = df['low'].tail(60).min()
        diff = recent_high - recent_low
        
        fib_levels = {
            '0.0%': recent_high,
            '23.6%': recent_high - 0.236 * diff,
            '38.2%': recent_high - 0.382 * diff,
            '50.0%': recent_high - 0.5 * diff,
            '61.8%': recent_high - 0.618 * diff,
            '100.0%': recent_low
        }
        return fib_levels

    def calculate_momentum(self, df, periods=[1, 5, 10, 20]):
        """Calculate momentum scores for different periods."""
        momentum_scores = {}
        for period in periods:
            returns = df['close'].pct_change(periods=period)
            momentum_scores[f'{period}d'] = returns.iloc[-1] * 100
            momentum_scores[f'{period}d_avg'] = returns.mean() * 100
            momentum_scores[f'{period}d_vol'] = returns.std() * 100
        return momentum_scores

    def predict_trend(self, df):
        """Use Random Forest to predict short-term trend."""
        features = df[['close', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'sma20', 'sma50']].dropna()
        if len(features) < 20:
            return None
            
        X = features[:-1]
        y = features['close'].shift(-1)[:-1]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        latest_data = features.tail(1)
        predicted_price = model.predict(latest_data)[0]
        current_price = df['close'].iloc[-1]
        
        return {
            'predicted_price': predicted_price,
            'expected_change': (predicted_price - current_price) / current_price * 100
        }

    def calculate_entry_exit(self, df, fib_levels, current_price, volatility):
        """Calculate suggested entry and exit prices."""
        entry_exit = {
            'entry': {},
            'exit': {},
            'stop_loss': current_price * (1 + self.STOP_LOSS),
            'risk_reward': None
        }
        
        # Entry logic: Look for pullback to 38.2% or 50% Fib level, RSI < 50, near BB lower band
        fib_38_2 = fib_levels['38.2%']
        fib_50_0 = fib_levels['50.0%']
        bb_lower = df['bb_lower'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        if current_price > fib_50_0 and rsi < 50:
            entry_price = min(fib_50_0, bb_lower * 1.005)  # 0.5% above lower BB
        elif current_price > fib_38_2 and rsi < 60:
            entry_price = min(fib_38_2, bb_lower * 1.005)
        else:
            entry_price = current_price * 0.995  # 0.5% below current price as fallback
        
        entry_exit['entry']['price'] = entry_price
        entry_exit['entry']['confidence'] = 0.8 if rsi < 50 and entry_price <= bb_lower * 1.01 else 0.6
        
        # Exit logic: Target 23.6% or 0.0% Fib level, or predicted price
        fib_23_6 = fib_levels['23.6%']
        fib_0_0 = fib_levels['0.0%']
        predicted_price = self.predict_trend(df)['predicted_price'] if self.predict_trend(df) else fib_23_6
        
        primary_target = min(fib_23_6, predicted_price * 1.02)  # 2% above predicted
        secondary_target = fib_0_0 if fib_0_0 > primary_target else primary_target * 1.03
        
        entry_exit['exit']['primary'] = {
            'price': primary_target,
            'probability': 0.75,
            'trailing_stop': primary_target * 0.98  # 2% trailing stop
        }
        entry_exit['exit']['secondary'] = {
            'price': secondary_target,
            'probability': 0.55,
            'trailing_stop': secondary_target * 0.98
        }
        
        # Risk-reward calculation
        risk = entry_price - entry_exit['stop_loss']
        reward = entry_exit['exit']['primary']['price'] - entry_price
        entry_exit['risk_reward'] = reward / risk if risk > 0 else None
        
        return entry_exit

    def generate_trade_pattern(self, df, targets, momentum, trend_prediction, entry_exit):
        """Generate logical trade pattern for the next 5 days."""
        trade_pattern = []
        current_price = df['close'].iloc[-1]
        volatility = df['close'].pct_change().std() * np.sqrt(252)
        trend_strength = trend_prediction['expected_change'] if trend_prediction else 0
        
        for day in range(1, 6):
            day_plan = {
                'day': day,
                'date': (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                'action': 'wait',
                'conditions': [],
                'target_price': None,
                'stop_loss': entry_exit['stop_loss']
            }
            
            # Adjust momentum and volatility for each day
            daily_momentum = momentum['5d'] / 5 * day
            expected_move = current_price * (1 + daily_momentum / 100 + volatility / np.sqrt(252))
            
            # Bullish scenario
            if trend_strength > 0.5 and day <= 3:
                if day == 1:
                    day_plan['action'] = 'buy'
                    day_plan['conditions'] = [
                        f"Price reaches entry: ${entry_exit['entry']['price']:.2f}",
                        f"RSI < 50 (current: {df['rsi'].iloc[-1]:.2f})",
                        "MACD bullish crossover"
                    ]
                    day_plan['target_price'] = entry_exit['exit']['primary']['price']
                else:
                    day_plan['action'] = 'hold'
                    day_plan['conditions'] = [
                        f"Price above ${entry_exit['entry']['price']:.2f}",
                        f"Trailing stop: ${entry_exit['exit']['primary']['trailing_stop']:.2f}"
                    ]
                    day_plan['target_price'] = entry_exit['exit']['secondary']['price'] if day > 2 else entry_exit['exit']['primary']['price']
            
            # Bearish scenario
            elif trend_strength < -0.5:
                day_plan['action'] = 'wait' if day == 1 else 'sell'
                day_plan['conditions'] = [
                    f"Price below ${targets['bearish'].get('38.2%', {}).get('price', current_price):.2f}",
                    f"RSI > 70 (current: {df['rsi'].iloc[-1]:.2f})"
                ]
                day_plan['target_price'] = targets['bearish'].get('50.0%', {}).get('price', current_price * 0.98)
            
            # Range-bound scenario
            else:
                day_plan['action'] = 'buy' if day <= 2 else 'hold'
                day_plan['conditions'] = [
                    f"Price near ${entry_exit['entry']['price']:.2f}",
                    f"Breakout above ${df['bb_upper'].iloc[-1]:.2f} or pullback to ${df['bb_lower'].iloc[-1]:.2f}"
                ]
                day_plan['target_price'] = entry_exit['exit']['primary']['price']
            
            trade_pattern.append(day_plan)
        
        return trade_pattern

    def calculate_targets(self, df, fib_levels, trend_prediction):
        """Calculate precise price targets and expected dates."""
        current_price = df['close'].iloc[-1]
        volatility = df['close'].pct_change().std() * np.sqrt(252)
        
        targets = {
            'bullish': {},
            'bearish': {},
            'stop_loss': current_price * (1 + self.STOP_LOSS)
        }
        
        for level, price in fib_levels.items():
            if price > current_price:
                targets['bullish'][level] = {
                    'price': price,
                    'probability': min(0.9, 1 / (1 + np.exp((price - current_price) / (current_price * volatility)))),
                    'expected_days': max(1, int((price - current_price) / (current_price * volatility / 252)))
                }
                
        for level, price in fib_levels.items():
            if price < current_price:
                targets['bearish'][level] = {
                    'price': price,
                    'probability': min(0.9, 1 / (1 + np.exp((current_price - price) / (current_price * volatility)))),
                    'expected_days': max(1, int((current_price - price) / (current_price * volatility / 252)))
                }
                
        if trend_prediction and trend_prediction['expected_change'] > 0:
            targets['bullish']['predicted'] = {
                'price': trend_prediction['predicted_price'],
                'probability': 0.7,
                'expected_days': 1
            }
        elif trend_prediction:
            targets['bearish']['predicted'] = {
                'price': trend_prediction['predicted_price'],
                'probability': 0.7,
                'expected_days': 1
            }
            
        return targets

    def analyze_stock(self, symbol):
        """Main analysis function."""
        print(f"\n{datetime.now()}: Analyzing {symbol}...")
        
        df_5m = self.get_data(symbol, timeframe="5m", limit=self.LOOKBACK)
        df_1d = self.get_data(symbol, timeframe="1d", days=self.HISTORICAL_DAYS)
        
        if df_5m is None or df_1d is None:
            return {"error": "Unable to fetch data"}
            
        df_5m = self.calculate_technical_indicators(df_5m)
        df_1d = self.calculate_technical_indicators(df_1d)
        
        fib_levels = self.calculate_fibonacci_levels(df_1d)
        momentum = self.calculate_momentum(df_1d)
        trend_prediction = self.predict_trend(df_5m)
        targets = self.calculate_targets(df_1d, fib_levels, trend_prediction)
        
        current_price = df_5m['close'].iloc[-1]
        volatility = df_1d['close'].pct_change().std() * np.sqrt(252)
        entry_exit = self.calculate_entry_exit(df_5m, fib_levels, current_price, volatility)
        trade_pattern = self.generate_trade_pattern(df_1d, targets, momentum, trend_prediction, entry_exit)
        
        report = {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'momentum': momentum,
            'technical_indicators': {
                'rsi': df_5m['rsi'].iloc[-1],
                'macd': df_5m['macd'].iloc[-1],
                'macd_signal': df_5m['macd_signal'].iloc[-1],
                'bb_position': (df_5m['close'].iloc[-1] - df_5m['bb_lower'].iloc[-1]) / 
                              (df_5m['bb_upper'].iloc[-1] - df_5m['bb_lower'].iloc[-1]) if df_5m['bb_upper'].iloc[-1] != df_5m['bb_lower'].iloc[-1] else None
            },
            'fibonacci_levels': fib_levels,
            'targets': targets,
            'trend_prediction': trend_prediction,
            'entry_exit': entry_exit,
            'trade_pattern': trade_pattern
        }
        
        self.opportunity_history.append(report)
        return report

    def print_report(self, report):
        """Print formatted analysis report."""
        if "error" in report:
            print(report["error"])
            return
            
        print(f"\nStock Analysis Report for {report['symbol']}")
        print(f"Generated: {report['timestamp']}")
        print(f"Current Price: ${report['current_price']:.2f}")
        
        print("\nMomentum Analysis:")
        for period, score in report['momentum'].items():
            print(f"{period}: {score:.2f}%")
            
        print("\nTechnical Indicators:")
        print(f"RSI (14): {report['technical_indicators']['rsi']:.2f}")
        print(f"MACD: {report['technical_indicators']['macd']:.4f}")
        print(f"Bollinger Band Position: {report['technical_indicators']['bb_position']:.2f}")
        
        print("\nFibonacci Retracement Levels:")
        for level, price in report['fibonacci_levels'].items():
            print(f"{level}: ${price:.2f}")
            
        print("\nEntry and Exit Strategy:")
        print(f"Suggested Entry Price: ${report['entry_exit']['entry']['price']:.2f} (Confidence: {report['entry_exit']['entry']['confidence']*100:.1f}%)")
        print(f"Primary Exit Price: ${report['entry_exit']['exit']['primary']['price']:.2f} (Probability: {report['entry_exit']['exit']['primary']['probability']*100:.1f}%)")
        print(f"Secondary Exit Price: ${report['entry_exit']['exit']['secondary']['price']:.2f} (Probability: {report['entry_exit']['exit']['secondary']['probability']*100:.1f}%)")
        print(f"Stop Loss: ${report['entry_exit']['stop_loss']:.2f}")
        print(f"Risk-Reward Ratio: {report['entry_exit']['risk_reward']:.2f}" if report['entry_exit']['risk_reward'] else "Risk-Reward: N/A")
        
        print("\nPrice Targets:")
        print("Bullish Targets:")
        for level, data in report['targets']['bullish'].items():
            print(f"{level}: ${data['price']:.2f} (Probability: {data['probability']*100:.1f}%, Expected: {data['expected_days']} days)")
            
        print("\nBearish Targets:")
        for level, data in report['targets']['bearish'].items():
            print(f"{level}: ${data['price']:.2f} (Probability: {data['probability']*100:.1f}%, Expected: {data['expected_days']} days)")
        
        print("\n5-Day Trade Pattern:")
        for day in report['trade_pattern']:
            print(f"\nDay {day['day']} ({day['date']}):")
            print(f"Action: {day['action'].capitalize()}")
            print(f"Target Price: ${day['target_price']:.2f}" if day['target_price'] else "Target Price: N/A")
            print(f"Stop Loss: ${day['stop_loss']:.2f}")
            print("Conditions:")
            for condition in day['conditions']:
                print(f"- {condition}")
        
        if report['trend_prediction']:
            print(f"\nShort-term Prediction:")
            print(f"Predicted Price: ${report['trend_prediction']['predicted_price']:.2f}")
            print(f"Expected Change: {report['trend_prediction']['expected_change']:.2f}%")

def main():
    bot = StockAnalysisBot()
    while True:
        symbol = input("\nEnter stock symbol (or 'quit' to exit): ").upper()
        if symbol.lower() == 'quit':
            break
        report = bot.analyze_stock(symbol)
        bot.print_report(report)

if __name__ == "__main__":
    main()
