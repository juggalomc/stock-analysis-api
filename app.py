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

    df = bot.calculate_technical_indicators(df)
    fib = bot.calculate_fibonacci_levels(df)
    momentum = bot.calculate_momentum(df)
    trend = bot.predict_trend(df)
    entry_exit = bot.calculate_entry_exit(df, fib, df['close'].iloc[-1], 0.02)

    result = {
        "momentum": momentum,
        "trend": trend,
        "entry_exit": entry_exit
    }
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
