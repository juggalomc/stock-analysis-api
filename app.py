from flask import Flask, request, jsonify, render_template
from stock_analysis_bot import StockAnalysisBot
import os

app = Flask(__name__)
bot = StockAnalysisBot()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "AAPL")
    df = bot.get_data(symbol, days=30)
    if df is None:
        return jsonify({"error": "Invalid data"}), 400

    # Calculate technical indicators, Fibonacci levels, and momentum
    df = bot.calculate_technical_indicators(df)
    fib = bot.calculate_fibonacci_levels(df)
    momentum = bot.calculate_momentum(df)
    trend = bot.predict_trend(df)
    
    # Calculate entry/exit strategy
    entry_exit = bot.calculate_entry_exit(df, fib, df['close'].iloc[-1], 0.02)
    
    # Calculate 5-day trade pattern
    trade_pattern = bot.calculate_trade_pattern(df)

    # Calculate price targets
    bullish_target, bearish_target = bot.calculate_price_targets(df)
    
    # Collect technical indicators
    rsi = float(df['RSI'].dropna().iloc[-1]) if 'RSI' in df.columns and not df['RSI'].dropna().empty else 'N/A'
    macd = float(df['MACD'].dropna().iloc[-1]) if 'MACD' in df.columns and not df['MACD'].dropna().empty else 'N/A'
    bollinger_band_position = bot.calculate_bollinger_band_position(df)
    
    # Build the result dictionary
    result = {
        "momentum": momentum,
        "trend": trend,
        "entry_exit": entry_exit,
        "trade_pattern": trade_pattern,  # Add the 5-day trade pattern to the result
        "price_targets": {
            "bullish": bullish_target if bullish_target else 'N/A',
            "bearish": bearish_target if bearish_target else 'N/A'
        },
        "technical_indicators": {
            "RSI": rsi,
            "MACD": macd,
            "Bollinger_Band_Position": bollinger_band_position,
            "Fibonacci_Retracement": fib if fib else 'N/A'
        }
    }
    
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
