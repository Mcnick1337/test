import requests
import pandas as pd
import pandas_ta as ta
import json
import re
import subprocess
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import anthropic

class TradingLearningSystem:
    """
    A learning system that tracks trading signal performance and provides
    feedback to improve future predictions.
    """
    
    def __init__(self, signals_file='signals_claude_log.json', performance_file='signal_claude_performance.json'):
        self.signals_file = signals_file
        self.performance_file = performance_file
        self.performance_data = self.load_performance_data()
    
    def load_performance_data(self) -> List[Dict]:
        """Load existing performance data"""
        try:
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_performance_data(self):
        """Save performance data to file"""
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_data, f, indent=2)
    
    def fetch_historical_price(self, symbol: str, timestamp: str) -> Optional[float]:
        """
        Fetch historical price for a given symbol and timestamp
        """
        try:
            # Convert timestamp to milliseconds
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp_ms = int(dt.timestamp() * 1000)
            
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol,
                'interval': '1h',
                'startTime': timestamp_ms,
                'limit': 1
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if data:
                return float(data[0][4])  # Close price
            return None
        except Exception as e:
            print(f"Error fetching historical price: {e}")
            return None
    
    def evaluate_signal_performance(self, signal_data: Dict) -> Dict:
        """
        Evaluate the performance of a past signal by checking current price
        against the signal's predictions.
        """
        symbol = signal_data.get('symbol')
        signal_type = signal_data.get('Signal')
        entry_price = signal_data.get('Entry Price')
        stop_loss = signal_data.get('Stop Loss')
        take_profit = signal_data.get('Take Profit Targets', [])
        timestamp = signal_data.get('timestamp')
        confidence = signal_data.get('Confidence', 50)
        
        if not all([symbol, signal_type, timestamp]):
            return {'status': 'insufficient_data'}
        
        # Skip Hold signals for performance evaluation
        if signal_type.lower() == 'hold':
            return {'status': 'hold_signal', 'outcome': 'neutral'}
        
        # Get current price
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {'status': 'price_fetch_error'}
        except:
            return {'status': 'price_fetch_error'}
        
        # Convert string prices to float if needed
        try:
            if isinstance(entry_price, str) and entry_price != 'N/A':
                entry_price = float(entry_price.replace('$', '').replace(',', ''))
            if isinstance(stop_loss, str) and stop_loss != 'N/A':
                stop_loss = float(stop_loss.replace('$', '').replace(',', ''))
            if isinstance(take_profit, list) and take_profit:
                take_profit = [float(str(tp).replace('$', '').replace(',', '')) for tp in take_profit if str(tp) != 'N/A']
        except (ValueError, AttributeError):
            return {'status': 'price_conversion_error'}
        
        if entry_price == 'N/A' or not isinstance(entry_price, (int, float)):
            return {'status': 'invalid_entry_price'}
        
        # Calculate performance
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        outcome = 'neutral'
        profit_loss_pct = 0
        hit_target = False
        hit_stop = False
        
        if signal_type.lower() == 'buy':
            if take_profit and current_price >= min(take_profit):
                outcome = 'win'
                hit_target = True
                profit_loss_pct = ((min(take_profit) - entry_price) / entry_price) * 100
            elif stop_loss != 'N/A' and isinstance(stop_loss, (int, float)) and current_price <= stop_loss:
                outcome = 'loss'
                hit_stop = True
                profit_loss_pct = ((stop_loss - entry_price) / entry_price) * 100
            else:
                outcome = 'open' if abs(price_change_pct) < 2 else ('winning' if price_change_pct > 0 else 'losing')
                profit_loss_pct = price_change_pct
        
        elif signal_type.lower() == 'sell':
            if take_profit and current_price <= max(take_profit):
                outcome = 'win'
                hit_target = True
                profit_loss_pct = ((entry_price - max(take_profit)) / entry_price) * 100
            elif stop_loss != 'N/A' and isinstance(stop_loss, (int, float)) and current_price >= stop_loss:
                outcome = 'loss'
                hit_stop = True
                profit_loss_pct = ((entry_price - stop_loss) / entry_price) * 100
            else:
                outcome = 'open' if abs(price_change_pct) < 2 else ('winning' if price_change_pct < 0 else 'losing')
                profit_loss_pct = -price_change_pct
        
        return {
            'status': 'evaluated',
            'outcome': outcome,
            'profit_loss_pct': round(profit_loss_pct, 2),
            'price_change_pct': round(price_change_pct, 2),
            'current_price': current_price,
            'entry_price': entry_price,
            'hit_target': hit_target,
            'hit_stop': hit_stop,
            'confidence': confidence,
            'evaluation_timestamp': datetime.utcnow().isoformat()
        }
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            url = f'https://api.binance.com/api/v3/ticker/price'
            params = {'symbol': symbol}
            response = requests.get(url, params=params)
            data = response.json()
            return float(data['price'])
        except:
            return None
    
    def update_signal_performance(self):
        """
        Update performance data for all signals in the signals log
        """
        try:
            with open(self.signals_file, 'r') as f:
                signals = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("No signals file found or invalid format")
            return
        
        updated_count = 0
        for signal in signals:
            signal_id = f"{signal.get('symbol')}_{signal.get('timestamp')}"
            
            # Check if already evaluated recently (within last hour)
            existing_perf = next((p for p in self.performance_data if p.get('signal_id') == signal_id), None)
            if existing_perf:
                last_eval = existing_perf.get('evaluation_timestamp')
                if last_eval:
                    try:
                        last_eval_dt = datetime.fromisoformat(last_eval)
                        if datetime.utcnow() - last_eval_dt < timedelta(hours=1):
                            continue  # Skip recent evaluations
                    except:
                        pass
            
            performance = self.evaluate_signal_performance(signal)
            if performance.get('status') == 'evaluated':
                performance['signal_id'] = signal_id
                performance['original_signal'] = signal
                
                # Update existing or add new
                if existing_perf:
                    existing_perf.update(performance)
                else:
                    self.performance_data.append(performance)
                
                updated_count += 1
        
        self.save_performance_data()
        print(f"Updated performance for {updated_count} signals")
    
    def get_performance_statistics(self) -> Dict:
        """
        Calculate performance statistics from historical signals
        """
        if not self.performance_data:
            return {}
        
        evaluated_signals = [p for p in self.performance_data if p.get('status') == 'evaluated']
        if not evaluated_signals:
            return {}
        
        wins = [p for p in evaluated_signals if p.get('outcome') in ['win', 'winning']]
        losses = [p for p in evaluated_signals if p.get('outcome') in ['loss', 'losing']]
        
        total_signals = len(evaluated_signals)
        win_rate = (len(wins) / total_signals) * 100 if total_signals > 0 else 0
        
        avg_win = np.mean([p.get('profit_loss_pct', 0) for p in wins]) if wins else 0
        avg_loss = np.mean([p.get('profit_loss_pct', 0) for p in losses]) if losses else 0
        
        # Performance by confidence level - FIX: Convert confidence to int/float
        def safe_confidence(p):
            conf = p.get('confidence', 50)
            try:
                return float(conf) if conf != 'N/A' else 50
            except (ValueError, TypeError):
                return 50
        
        high_conf = [p for p in evaluated_signals if safe_confidence(p) >= 80]
        med_conf = [p for p in evaluated_signals if 60 <= safe_confidence(p) < 80]
        low_conf = [p for p in evaluated_signals if safe_confidence(p) < 60]
        
        return {
            'total_signals': total_signals,
            'win_rate': round(win_rate, 2),
            'average_win': round(avg_win, 2),
            'average_loss': round(avg_loss, 2),
            'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            'high_confidence_signals': len(high_conf),
            'high_conf_win_rate': round((len([p for p in high_conf if p.get('outcome') in ['win', 'winning']]) / len(high_conf)) * 100, 2) if high_conf else 0,
            'medium_confidence_signals': len(med_conf),
            'med_conf_win_rate': round((len([p for p in med_conf if p.get('outcome') in ['win', 'winning']]) / len(med_conf)) * 100, 2) if med_conf else 0,
            'low_confidence_signals': len(low_conf),
            'low_conf_win_rate': round((len([p for p in low_conf if p.get('outcome') in ['win', 'winning']]) / len(low_conf)) * 100, 2) if low_conf else 0
        }
    
    def generate_learning_prompt_addition(self) -> str:
        """
        Generate additional prompt text based on past performance to improve future signals
        """
        stats = self.get_performance_statistics()
        if not stats:
            return ""
        
        recent_signals = sorted(self.performance_data, key=lambda x: x.get('evaluation_timestamp', ''), reverse=True)[:10]
        
        learning_insights = []
        
        # Win rate analysis
        if stats['win_rate'] < 50:
            learning_insights.append("CRITICAL: Recent win rate is below 50%. Increase signal selectivity and require stronger confluences.")
        elif stats['win_rate'] > 70:
            learning_insights.append("POSITIVE: Win rate is strong. Current strategy is working well.")
        
        # Confidence level analysis
        if stats.get('high_conf_win_rate', 0) < stats.get('low_conf_win_rate', 0):
            learning_insights.append("WARNING: High confidence signals performing worse than low confidence. Review confidence calibration.")
        
        # Recent performance patterns
        recent_outcomes = [s.get('outcome') for s in recent_signals]
        recent_losses = recent_outcomes.count('loss') + recent_outcomes.count('losing')
        if recent_losses > 6:  # More than 60% recent losses
            learning_insights.append("ALERT: High recent loss rate detected. Consider more conservative approach.")
        
        prompt_addition = f"""
=== PERFORMANCE LEARNING SYSTEM ===
Historical Performance Statistics:
- Total Signals Analyzed: {stats['total_signals']}
- Overall Win Rate: {stats['win_rate']}%
- Average Win: {stats['average_win']}%
- Average Loss: {stats['average_loss']}%
- Profit Factor: {stats['profit_factor']}

Confidence Level Performance:
- High Confidence (80%+): {stats['high_conf_win_rate']}% win rate ({stats['high_confidence_signals']} signals)
- Medium Confidence (60-79%): {stats['med_conf_win_rate']}% win rate ({stats['medium_confidence_signals']} signals)
- Low Confidence (<60%): {stats['low_conf_win_rate']}% win rate ({stats['low_confidence_signals']} signals)

Learning Insights:
{chr(10).join(['- ' + insight for insight in learning_insights])}

Recent Signal Outcomes (Last 10):
{chr(10).join([f"- {s.get('original_signal', {}).get('symbol', 'N/A')}: {s.get('outcome', 'N/A')} ({s.get('profit_loss_pct', 0):+.1f}%)" for s in recent_signals[:5]])}

ADAPTATION INSTRUCTIONS:
Based on the performance data above, adjust your signal generation to improve future outcomes. 
Pay special attention to the confidence calibration and recent performance patterns.
"""
        
        return prompt_addition

def create_example_signals_file():
    """
    Create an example signals_log.json file with sample data
    """
    example_signals = [
        {
            "Signal": "Buy",
            "Entry Price": "95000.00",
            "Take Profit Targets": ["98000.00", "101000.00"],
            "Stop Loss": "92000.00",
            "Confidence": 85,
            "Scenario": "Breakout",
            "Trade Setup Type": "Momentum",
            "Reasoning": "Strong bullish momentum with multiple confluences. RSI showing strength, MACD bullish crossover, and price breaking above key resistance at $94,500.",
            "Relevant News Headlines": ["Bitcoin ETF inflows reach new highs", "Major institutional adoption announced"],
            "timestamp": "2024-12-15T10:30:00.000000",
            "symbol": "BTCUSDT"
        },
        {
            "Signal": "Sell",
            "Entry Price": "3800.00",
            "Take Profit Targets": ["3600.00", "3400.00"],
            "Stop Loss": "3950.00",
            "Confidence": 72,
            "Scenario": "Reversal",
            "Trade Setup Type": "Mean Reversion",
            "Reasoning": "Overbought conditions with bearish divergence on RSI. Price rejected at key resistance level with declining volume.",
            "Relevant News Headlines": ["Ethereum network congestion concerns", "DeFi protocol exploit reported"],
            "timestamp": "2024-12-15T11:45:00.000000",
            "symbol": "ETHUSDT"
        },
        {
            "Signal": "Hold",
            "Entry Price": "N/A",
            "Take Profit Targets": [],
            "Stop Loss": "N/A",
            "Confidence": 60,
            "Scenario": "Range-Bound",
            "Trade Setup Type": "Other",
            "Reasoning": "Mixed signals with no clear directional bias. Price consolidating in narrow range with low volatility. Wait for clearer setup.",
            "Relevant News Headlines": ["Solana ecosystem updates", "Mixed market sentiment prevails"],
            "timestamp": "2024-12-15T12:15:00.000000",
            "symbol": "SOLUSDT"
        },
        {
            "Signal": "Buy",
            "Entry Price": "96500.00",
            "Take Profit Targets": ["99000.00"],
            "Stop Loss": "94000.00",
            "Confidence": 91,
            "Scenario": "Trend Continuation",
            "Trade Setup Type": "Swing",
            "Reasoning": "Perfect trend continuation setup. Price pulled back to 20 EMA support, bullish flag pattern completion, high volume confirmation.",
            "Relevant News Headlines": ["Bitcoin mining hashrate hits new ATH", "Central bank digital currency developments"],
            "timestamp": "2024-12-15T14:20:00.000000",
            "symbol": "BTCUSDT"
        }
    ]
    
    with open('signals_log.json', 'w') as f:
        json.dump(example_signals, f, indent=2)
    
    print("Created example signals_log.json file with sample data")

# Example usage and integration
if __name__ == "__main__":
    # Create example signals file if it doesn't exist
    try:
        with open('signals_log.json', 'r') as f:
            pass
    except FileNotFoundError:
        create_example_signals_file()
    
    # Initialize learning system
    learning_system = TradingLearningSystem()
    
    # Update performance for all signals
    print("Updating signal performance...")
    learning_system.update_signal_performance()
    
    # Get performance statistics
    stats = learning_system.get_performance_statistics()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate learning prompt addition
    learning_prompt = learning_system.generate_learning_prompt_addition()
    print("\nLearning Prompt Addition:")
    print(learning_prompt)

# -------------------- Market Data --------------------
def fetch_ohlcv(symbol='BTCUSDT', interval='1h', limit=250):
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

# -------------------- Multi-Timeframe OHLCV Fetch --------------------
def fetch_multi_timeframe_ohlcv(symbol):
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    multi_df = {}
    for tf in timeframes:
        try:
            df = fetch_ohlcv(symbol, interval=tf)
            df = compute_indicators(df)
            multi_df[tf] = df
        except Exception as e:
            print(f"⚠️ Error fetching or computing indicators for {symbol} - {tf}: {e}")
    return multi_df

# -------------------- Order Book and Liquidity Zones --------------------
def fetch_order_book(symbol='BTCUSDT', limit=100):
    url = 'https://api.binance.com/api/v3/depth'
    params = {'symbol': symbol, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    return data

def analyze_liquidity_zones(order_book, depth=20):
    bids = [(float(price), float(qty)) for price, qty in order_book['bids'][:depth]]
    asks = [(float(price), float(qty)) for price, qty in order_book['asks'][:depth]]

    # Sum total bid/ask volume and determine concentration zones
    bid_zone = max(bids, key=lambda x: x[1])  # highest volume bid zone
    ask_zone = max(asks, key=lambda x: x[1])  # highest volume ask zone

    bid_volume = sum(p * q for p, q in bids)
    ask_volume = sum(p * q for p, q in asks)

    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) * 100  # percent

    return {
        "bid_zone_price": bid_zone[0],
        "bid_zone_volume": bid_zone[1],
        "ask_zone_price": ask_zone[0],
        "ask_zone_volume": ask_zone[1],
        "bid_ask_imbalance_pct": imbalance
    }

#-------------------- Diagnostic --------------------
# def print_status(name, series):
    if series is None:
        print(f"❌ {name} NOT CALCULATED (series is None)")
        return
    if series.isnull().all():
        print(f"❌ {name} NOT CALCULATED (all NaNs)")
        return
    if (series == 0).all():
        print(f"❌ {name} NOT CALCULATED (all zeros)")
        return
    print(f"✅ {name} CALCULATED")

# -------------------- Indicators --------------------
def compute_indicators(df):
    df['rsi'] = ta.rsi(df['close'], length=14)  # RSI
    # print_status("RSI", df['rsi'])

    macd = ta.macd(df['close'])  # MACD
    df = pd.concat([df, macd], axis=1)
    # MACD can be multiple columns (macd, macdsignal, macdhist), check if any column is valid
    # if macd is None or macd.isnull().all().all() or (macd == 0).all().all():
    #    print("❌ MACD NOT CALCULATED")
    # else:
    #    print("✅ MACD CALCULATED")

    df['ema_20'] = ta.ema(df['close'], length=20)  # EMA20
    # print_status("EMA20", df['ema_20'])

    df['ema_50'] = ta.ema(df['close'], length=50)  # EMA50
    # print_status("EMA50", df['ema_50'])

    df['ema_200'] = ta.ema(df['close'], length=200)  # EMA200
    # print_status("EMA200", df['ema_200'])

    bbands = ta.bbands(df['close'], length=20, std=2)  # Bollinger Bands
    df = pd.concat([df, bbands], axis=1)
    # if bbands is None or bbands.isnull().all().all() or (bbands == 0).all().all():
    #     print("❌ BOLLINGER BANDS NOT CALCULATED")
    # else:
    #     print("✅ BOLLINGER BANDS CALCULATED")

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)  # ATR
    # print_status("ATR", df['atr'])

    ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
    if isinstance(ichimoku, tuple) and all(isinstance(i, pd.DataFrame) for i in ichimoku):
        ichimoku_df = pd.concat(ichimoku, axis=1)
        df = df.join(ichimoku_df)
        # For simplicity check one ichimoku component for validity:
        first_comp = ichimoku_df.iloc[:, 0]
    #     print_status("ICHIMOKU", first_comp)
    # else:
    #     print("❌ ICHIMOKU NOT CALCULATED")

    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])  # VWAP
    # print_status("VWAP", df['vwap'])

    adx = ta.adx(df['high'], df['low'], df['close'])  # ADX
    df = pd.concat([df, adx], axis=1)
    # Check if ADX column exists and valid
    # if 'ADX_14' in df.columns:
    #     print_status("ADX", df['ADX_14'])
    # else:
    #     print("❌ ADX NOT CALCULATED")

    return df


# --------------- Custom EMA and ATR ------------------
def calculate_ema(prices, period):
    emas = []
    k = 2 / (period + 1)
    for i, price in enumerate(prices):
        if i == 0:
            emas.append(price)
        else:
            ema = price * k + emas[-1] * (1 - k)
            emas.append(ema)

    # Check if any valid EMA calculated (non-zero)
    # if all(e == 0 or e is None for e in emas):
    #     print(f"❌ EMA{period} NOT CALCULATED (custom)")
    # else:
    #     print(f"✅ EMA{period} CALCULATED (custom)")

    return emas


def calculate_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(len(highs)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        trs.append(tr)
    atrs = []
    for i in range(len(trs)):
        if i < period - 1:
            atrs.append(None)
        elif i == period - 1:
            atrs.append(sum(trs[:period]) / period)
        else:
            atr = (atrs[-1] * (period - 1) + trs[i]) / period
            atrs.append(atr)

    # if all(a == 0 or a is None for a in atrs):
    #    print(f"❌ ATR NOT CALCULATED (custom, period={period})")
    # else:
    #    print(f"✅ ATR CALCULATED (custom, period={period})")

    return atrs

# ----------------- Volume Indicator -----------------
def calculate_volume_profile(order_book, bucket_size=50):
    """
    Calculates volume profile by grouping prices into buckets and summing volumes.
    bucket_size is price range width (e.g., 50 USD)
    """
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])
    
    def aggregate_side(side):
        buckets = {}
        for price_str, vol_str in side:
            price = float(price_str)
            volume = float(vol_str)
            bucket = int(price // bucket_size) * bucket_size
            buckets[bucket] = buckets.get(bucket, 0) + volume
        return buckets

    bid_buckets = aggregate_side(bids)
    ask_buckets = aggregate_side(asks)

    # Find top volume bucket overall (bids + asks combined)
    combined_buckets = {}
    for bkt, vol in bid_buckets.items():
        combined_buckets[bkt] = combined_buckets.get(bkt, 0) + vol
    for bkt, vol in ask_buckets.items():
        combined_buckets[bkt] = combined_buckets.get(bkt, 0) + vol

    if combined_buckets:
        top_volume_zone = max(combined_buckets, key=combined_buckets.get)
        volume_at_top_zone = combined_buckets[top_volume_zone]
    else:
        top_volume_zone = None
        volume_at_top_zone = None

    return {
        'bids': bid_buckets,
        'asks': ask_buckets,
        'top_volume_zone': top_volume_zone,
        'volume_at_top_zone': volume_at_top_zone
    }

# ----------------- Trend detection ------------------
def detect_trend(df):
    """
    Enhanced trend detection logic using EMA20, EMA200, ATR and ADX.

    Args:
        df: DataFrame with computed indicators (must include 'ema_20', 'ema_200', 'close', 'atr', 'ADX_14')

    Returns:
        dict with trend_direction ('bullish', 'bearish', 'neutral'), trend_strength, adx
    """
    try:
        latest = df.iloc[-1]
        ema20 = latest['ema_20']
        ema200 = latest['ema_200']
        close = latest['close']
        atr = latest['atr']
        adx = latest.get('ADX_14', 0)

        # Basic EMAs crossover logic
        if ema20 > ema200 and close > ema20:
            trend_direction = 'bullish'
        elif ema20 < ema200 and close < ema20:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'

        # Trend strength based on ADX and ATR volatility
        if adx > 25:
            strength = 'strong'
        elif adx > 15:
            strength = 'moderate'
        else:
            strength = 'weak'

        # Incorporate ATR for volatility check (higher ATR means higher volatility)
        volatility = 'high' if atr and atr > (0.01 * close) else 'low'  # simple relative ATR

        return {
            'trend_direction': trend_direction,
            'trend_strength': strength,
            'adx': round(adx, 2),
            'volatility': volatility
        }
    except Exception as e:
        print(f"⚠️ Error in trend detection: {e}")
        return {
            'trend_direction': 'unknown',
            'trend_strength': 'unknown',
            'adx': None,
            'volatility': 'unknown'
        }

# -------------------- Market Metrics --------------------
def fetch_market_cap_and_volume():
    url = 'https://api.coingecko.com/api/v3/global'
    response = requests.get(url)
    data = response.json().get('data', {})
    return {
        'market_cap_usd': data.get('total_market_cap', {}).get('usd'),
        'volume_usd': data.get('total_volume', {}).get('usd')
    }

def fetch_hash_rate():
    url = 'https://api.blockchain.info/charts/hash-rate?timespan=1days&format=json'
    response = requests.get(url)
    data = response.json().get('values', [])
    return data[-1]['y'] if data else None

def fetch_network_security():
    etherscan_api = 'BAZQ1AXBP5K54NEVWQ7TFY6JUZVA42D31P'
    url = f'https://api.etherscan.io/api?module=stats&action=nodecount&apikey={etherscan_api}'
    response = requests.get(url)
    result = response.json()
    return {
        'nodes': result.get('result')
    }

# -------------------- News Fetch --------------------
def fetch_news(api_key, query='cryptocurrency', language='en', page_size=5):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': language,
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    articles = response.json().get('articles', [])
    return articles

# -------------------- LLM Invocation (FIXED) --------------------
def run_ollama(prompt, model):
    command = ['ollama', 'run', model]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=prompt.encode())
    if process.returncode != 0:
        raise RuntimeError(f"Ollama error: {stderr.decode()}")
    
    response = stdout.decode().strip()
    
    # Save raw response for debugging
    with open('ollama_raw_response.txt', 'w', encoding='utf-8') as f:
        f.write(response)

def run_claude(prompt, api_key):
    """
    Run Claude Sonnet 4 via Anthropic API instead of Ollama
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.content[0].text
        
        # Save raw response for debugging
        with open('claude_raw_response.txt', 'w', encoding='utf-8') as f:
            f.write(response_text)
        
        # Try to find JSON content
        try:
            json_start = response_text.index('{')
            json_part = response_text[json_start:]
            
            # Find the last closing brace
            brace_count = 0
            json_end = -1
            for i, char in enumerate(json_part):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > 0:
                json_part = json_part[:json_end]
            else:
                # If we can't find proper closing, try to add one
                if not json_part.strip().endswith('}'):
                    json_part += '}'
            
            return json.loads(json_part)
            
        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print(f"Raw response saved to claude_raw_response.txt")
            
            # Try to parse the text response and extract key information
            try:
                signal = "Hold"
                entry_price = "N/A"
                stop_loss = "N/A"
                take_profit = []
                confidence = 50
                scenario = "Uncertain"
                setup_type = "Other"
                reasoning = "Model returned text instead of JSON"
                
                # Extract signal type
                response_lower = response_text.lower()
                if "sell" in response_lower or "short" in response_lower:
                    signal = "Sell"
                elif "buy" in response_lower or "long" in response_lower:
                    signal = "Buy"
                
                # Extract prices using regex
                import re
                price_pattern = r'\$(\d+(?:\.\d{2})?)'
                prices = re.findall(price_pattern, response_text)
                
                if len(prices) >= 3:
                    entry_price = prices[0]
                    stop_loss = prices[1] 
                    take_profit = [prices[2]]
                
                # Extract confidence if mentioned
                conf_match = re.search(r'(\d+)%', response_text)
                if conf_match:
                    confidence = int(conf_match.group(1))
                
                # Extract scenario
                if "trend continuation" in response_lower:
                    scenario = "Trend Continuation"
                elif "reversal" in response_lower:
                    scenario = "Reversal"
                elif "breakout" in response_lower:
                    scenario = "Breakout"
                elif "range" in response_lower:
                    scenario = "Range-Bound"
                    
                # Extract setup type
                if "momentum" in response_lower:
                    setup_type = "Momentum"
                elif "mean reversion" in response_lower:
                    setup_type = "Mean Reversion"
                elif "swing" in response_lower:
                    setup_type = "Swing"
                elif "scalping" in response_lower:
                    setup_type = "Scalping"
                
                return {
                    "Signal": signal,
                    "Entry Price": entry_price,
                    "Take Profit Targets": take_profit,
                    "Stop Loss": stop_loss,
                    "Confidence": confidence,
                    "Scenario": scenario,
                    "Trade Setup Type": setup_type,
                    "Reasoning": f"Parsed from text response: {reasoning}",
                    "Relevant News Headlines": ["Unable to parse structured response"]
                }
                
            except Exception as parse_error:
                print(f"⚠️ Text parsing also failed: {parse_error}")
                # Final fallback
                return {
                    "Signal": "Hold",
                    "Entry Price": "N/A",
                    "Take Profit Targets": [],
                    "Stop Loss": "N/A",
                    "Confidence": 50,
                    "Scenario": "Uncertain",
                    "Trade Setup Type": "Other",
                    "Reasoning": f"Both JSON and text parsing failed. Raw response: {response_text[:200]}...",
                    "Relevant News Headlines": ["Unable to parse model response"]
                }
                
    except Exception as e:
        print(f"⚠️ Claude API error: {e}")
        return {
            "Signal": "Hold",
            "Entry Price": "N/A",
            "Take Profit Targets": [],
            "Stop Loss": "N/A",
            "Confidence": 50,
            "Scenario": "API Error",
            "Trade Setup Type": "Other",
            "Reasoning": f"Claude API error: {str(e)}",
            "Relevant News Headlines": ["API connection failed"]
        }

# -------------------- News Analysis (FIXED) --------------------
def analyze_news_with_llama(news_articles):
    summaries = []
    unsafe_keywords = [
        "torture", "kidnap", "rape", "murder", "abuse", "assault",
        "executed", "suicide", "terror", "blood", "kill", "dead", "corpse"
    ]

    for article in news_articles:
        try:
            # Safely get article content
            title = article.get('title', '') if article else ''
            description = article.get('description', '') if article else ''
            
            # Skip if no meaningful content
            if not title and not description:
                print(f"⚠️ Skipping article due to empty content")
                continue
                
            content = title + ' ' + description
            
            # Check for unsafe content before processing
            content_lower = content.lower()
            if any(word in content_lower for word in unsafe_keywords):
                print(f"⚠️ Skipping article due to flagged content: {title[:50]}...")
                continue

            prompt = f"""
You are a seasoned financial analyst. Read the following crypto news article and reply ONLY in JSON format.

Article:
{content}

Reply in this format only:
{{
  "summary": "<concise summary>",
  "sentiment": "<Positive | Neutral | Negative>",
  "reason": "<short reason for sentiment>"
}}
"""
            
            # Get response from Ollama
            response = run_ollama(prompt, 'llama3.2:3b')
            
            # Check if response is None or empty
            if response is None:
                print(f"⚠️ Skipping article due to None response from Ollama: {title[:50]}...")
                continue
                
            # Check if response is a dictionary (expected)
            if not isinstance(response, dict):
                print(f"⚠️ Skipping article due to unexpected response type: {type(response)}")
                continue
            
            # Additional safety check for response content
            response_text = json.dumps(response).lower()
            if any(word in response_text for word in unsafe_keywords):
                print(f"⚠️ Skipping article due to flagged content in response: {title[:50]}...")
                continue

            # Safely extract fields with defaults
            summary = response.get('summary', 'No summary available')
            sentiment = response.get('sentiment', 'Neutral')
            reason = response.get('reason', 'No reason provided')
            
            # Validate sentiment value
            valid_sentiments = ['positive', 'neutral', 'negative']
            if sentiment.lower() not in valid_sentiments:
                sentiment = 'Neutral'
            
            summaries.append({
                'title': title,
                'summary': summary,
                'sentiment': sentiment,
                'reason': reason
            })
            
        except Exception as e:
            # More detailed error logging
            article_title = article.get('title', 'Unknown title') if article else 'No article data'
            print(f"⚠️ Skipping article '{article_title[:50]}...' due to error: {str(e)}")
            continue

    # If no summaries were generated, provide a default
    if not summaries:
        summaries.append({
            'title': 'No news processed',
            'summary': 'No cryptocurrency news articles were successfully processed',
            'sentiment': 'Neutral',
            'reason': 'No valid articles available for analysis'
        })

    return summaries

# -------------------- Enhanced Signal Analysis --------------------
def calculate_signal_confluence(indicators, mtf_data, trend_info, liquidity):
    confluence_score = 0
    signal_factors = []

    rsi = indicators.get('RSI')
    if rsi:
        if rsi < 30:
            confluence_score += 15
            signal_factors.append("RSI_OVERSOLD")
        elif rsi > 70:
            confluence_score -= 15
            signal_factors.append("RSI_OVERBOUGHT")

    macd = indicators.get('MACD')
    macd_signal = indicators.get('MACD_signal')
    if macd and macd_signal:
        if macd > macd_signal:
            confluence_score += 10
            signal_factors.append("MACD_BULLISH_CROSS")
        elif macd < macd_signal:
            confluence_score -= 10
            signal_factors.append("MACD_BEARISH_CROSS")

    close = indicators.get('Close')
    ema20 = indicators.get('EMA_20')
    ema50 = indicators.get('EMA_50')
    ema200 = indicators.get('EMA_200')
    if all([close, ema20, ema50, ema200]):
        if close > ema20 > ema50 > ema200:
            confluence_score += 20
            signal_factors.append("STRONG_BULLISH_TREND")
        elif close < ema20 < ema50 < ema200:
            confluence_score -= 20
            signal_factors.append("STRONG_BEARISH_TREND")

    adx = indicators.get('ADX')
    if adx:
        if adx > 30:
            confluence_score += 5
            signal_factors.append("STRONG_TREND_ADX")
        elif adx < 15:
            confluence_score -= 5
            signal_factors.append("WEAK_TREND_ADX")

    bb_upper = indicators.get('BB_Upper')
    bb_lower = indicators.get('BB_Lower')
    if close and bb_upper and bb_lower:
        if close > bb_upper:
            confluence_score += 5
            signal_factors.append("BB_BREAKOUT_UP")
        elif close < bb_lower:
            confluence_score -= 5
            signal_factors.append("BB_BREAKOUT_DOWN")

    bullish_tfs = sum(1 for tf in mtf_data.values() if tf.get('EMA_20', 0) > tf.get('EMA_200', 0))
    if bullish_tfs >= 4:
        confluence_score += 10
        signal_factors.append("MTF_BULLISH_ALIGNMENT")
    elif bullish_tfs <= 2:
        confluence_score -= 10
        signal_factors.append("MTF_BEARISH_ALIGNMENT")

    imbalance = liquidity.get('bid_ask_imbalance_pct', 0)
    if imbalance > 5:
        confluence_score += 5
        signal_factors.append("BID_DOMINANCE")
    elif imbalance < -5:
        confluence_score -= 5
        signal_factors.append("ASK_DOMINANCE")

    return {
        'confluence_score': confluence_score,
        'signal_factors': signal_factors,
        'confluence_strength': 'HIGH' if abs(confluence_score) > 30 else 'MEDIUM' if abs(confluence_score) > 15 else 'LOW'
    }

def calculate_risk_reward_metrics(indicators, liquidity, atr_multiplier=2):
    close = indicators.get('Close', 0)
    atr = indicators.get('ATR', 0)
    if not close or not atr:
        return {}

    stop_loss_distance = atr * atr_multiplier

    support = liquidity.get('bid_zone_price', close * 0.98)
    resistance = liquidity.get('ask_zone_price', close * 1.02)

    if close < support:
        support = close - atr * 1.5
    if close > resistance:
        resistance = close + atr * 1.5

    rr = abs(resistance - close) / stop_loss_distance if stop_loss_distance > 0 else 0
    rr = min(rr, 10)

    return {
        'atr_stop_distance': stop_loss_distance,
        'support_level': support,
        'resistance_level': resistance,
        'risk_reward_ratio': rr
    }

# -------------------- Signal Generation --------------------
def generate_trade_signal(df, df_1h, news_summaries, metrics, liquidity, volume_profile, multi_df, trend_info, api_key):
    latest_data = df.iloc[-1].to_dict()
    indicators = {
        'RSI': latest_data.get('rsi'),
        'MACD': latest_data.get('MACD_12_26_9'),
        'MACD_signal': latest_data.get('MACDs_12_26_9'),
        'MACD_hist': latest_data.get('MACDh_12_26_9'),
        'EMA_20': latest_data.get('ema_20'),
        'EMA_50': latest_data.get('ema_50'),
        'EMA_200': latest_data.get('ema_200'),
        'VWAP': latest_data.get('vwap'),
        'ATR': latest_data.get('atr'),
        'ADX': latest_data.get('ADX_14'),
        'Close': latest_data.get('close'),
        'Volume': latest_data.get('volume'),
        'BB_Upper': latest_data.get('BBU_20_2.0'),
        'BB_Lower': latest_data.get('BBL_20_2.0'),
        'BB_Middle': latest_data.get('BBM_20_2.0')
    }

    ichimoku_signal = latest_data.get('tenkan_sen', 'N/A')  # Using actual Ichimoku field if available

     # --- Multi-timeframe indicators ---
    def safe_get(mtf, tf, column):
        try:
            return round(mtf[tf][column].iloc[-1], 4)
        except Exception:
            return 'N/A'

    tf_list = ['1m', '5m', '15m', '1h', '4h', '1d']
    mtf_data = {}
    for tf in tf_list:
        mtf_data[tf] = {
            'RSI': safe_get(multi_df, tf, 'rsi'),
            'EMA_20': safe_get(multi_df, tf, 'ema_20'),
            'EMA_200': safe_get(multi_df, tf, 'ema_200'),
            'ATR': safe_get(multi_df, tf, 'atr'),
            'VWAP': safe_get(multi_df, tf, 'vwap'),
            'MACD': safe_get(multi_df, tf, 'MACD_12_26_9'),
            'Volume': safe_get(multi_df, tf, 'volume')
        }

    # Enhanced analysis
    confluence_analysis = calculate_signal_confluence(indicators, mtf_data, trend_info, liquidity)
    risk_metrics = calculate_risk_reward_metrics(indicators, liquidity)

    # Self learning system integration
    learning_system = TradingLearningSystem()
    learning_system.update_signal_performance()
    learning_addition = learning_system.generate_learning_prompt_addition()

    # News sentiment analysis
    bullish_news = sum(1 for news in news_summaries if news.get('sentiment', '').lower() == 'positive')
    bearish_news = sum(1 for news in news_summaries if news.get('sentiment', '').lower() == 'negative')
    news_bias = 'BULLISH' if bullish_news > bearish_news else 'BEARISH' if bearish_news > bullish_news else 'NEUTRAL'
    news_input = '\n'.join([f"{item['title']}: {item['summary']}" for item in news_summaries])
    
    prompt = f"""
YOU are an elite institutional crypto trader with 10+ years experience. Your track record includes 78% win rate with average 3.2:1 R/R ratio.

CRITICAL INSTRUCTIONS:
1. Analyze ALL data points systematically
2. Provide SPECIFIC entry, SL, and TP levels based on technical confluence
3. Evaluate the confluence of signals.
4. Identify the scenario type: Breakout, Reversal, Trend Continuation, or Range-Bound.
5. Assess market regime and adapt strategy accordingly
6. Generate a clear trade signal institutional-grade precision, only in structured JSON format.

=== CURRENT MARKET DATA ===
Symbol: {latest_data.get('symbol', 'CRYPTO')}
Current Price: ${indicators.get('Close', 0):,.2f}
24h Volume: {indicators.get('Volume', 0):,.0f}

=== TECHNICAL INDICATORS ===
RSI(14): {indicators.get('RSI', 0):.2f}
MACD: {indicators.get('MACD', 0):.4f} | Signal: {indicators.get('MACD_signal', 0):.4f} | Histogram: {indicators.get('MACD_hist', 0):.4f}
EMA20: ${indicators.get('EMA_20', 0):,.2f} | EMA50: ${indicators.get('EMA_50', 0):,.2f} | EMA200: ${indicators.get('EMA_200', 0):,.2f}
VWAP: ${indicators.get('VWAP', 0):,.2f}
ATR(14): {indicators.get('ATR', 0):.2f}
ADX: {indicators.get('ADX', 0):.2f}
Bollinger Bands: Upper ${indicators.get('BB_Upper', 0):,.2f} | Middle ${indicators.get('BB_Middle', 0):,.2f} | Lower ${indicators.get('BB_Lower', 0):,.2f}

=== MULTI-TIMEFRAME ANALYSIS ===
1min  - RSI: {mtf_data['1m']['RSI']} | EMA20: {mtf_data['1m']['EMA_20']} | EMA200: {mtf_data['1m']['EMA_200']} | VWAP: {mtf_data['1m']['VWAP']}
5min  - RSI: {mtf_data['5m']['RSI']} | EMA20: {mtf_data['5m']['EMA_20']} | EMA200: {mtf_data['5m']['EMA_200']} | VWAP: {mtf_data['5m']['VWAP']}
15min - RSI: {mtf_data['15m']['RSI']} | EMA20: {mtf_data['15m']['EMA_20']} | EMA200: {mtf_data['15m']['EMA_200']} | VWAP: {mtf_data['15m']['VWAP']}
1hour - RSI: {mtf_data['1h']['RSI']} | EMA20: {mtf_data['1h']['EMA_20']} | EMA200: {mtf_data['1h']['EMA_200']} | VWAP: {mtf_data['1h']['VWAP']}
4hour - RSI: {mtf_data['4h']['RSI']} | EMA20: {mtf_data['4h']['EMA_20']} | EMA200: {mtf_data['4h']['EMA_200']} | VWAP: {mtf_data['4h']['VWAP']}
1day  - RSI: {mtf_data['1d']['RSI']} | EMA20: {mtf_data['1d']['EMA_20']} | EMA200: {mtf_data['1d']['EMA_200']} | VWAP: {mtf_data['1d']['VWAP']}

=== LIQUIDITY & ORDER FLOW ===
Strongest Bid Zone: ${liquidity.get('bid_zone_price', 0):,.2f} (Volume: {liquidity.get('bid_zone_volume', 0):,.4f})
Strongest Ask Zone: ${liquidity.get('ask_zone_price', 0):,.2f} (Volume: {liquidity.get('ask_zone_volume', 0):,.4f})
Order Book Imbalance: {liquidity.get('bid_ask_imbalance_pct', 0):+.2f}%
High Volume Node: ${volume_profile.get('top_volume_zone', 0):,.0f} (Volume: {volume_profile.get('volume_at_top_zone', 0):,.2f})

=== TREND & MOMENTUM ANALYSIS ===
Primary Trend: {trend_info.get('trend_direction', 'N/A').upper()}
Trend Strength: {trend_info.get('trend_strength', 'N/A').upper()}
ADX Reading: {trend_info.get('adx', 'N/A')}
Volatility Regime: {trend_info.get('volatility', 'N/A').upper()}

=== CONFLUENCE ANALYSIS ===
Confluence Score: {confluence_analysis.get('confluence_score', 0)}/100
Signal Strength: {confluence_analysis.get('confluence_strength', 'LOW')}
Active Factors: {', '.join(confluence_analysis.get('signal_factors', []))}

=== RISK MANAGEMENT METRICS ===
ATR Stop Distance: ${risk_metrics.get('atr_stop_distance', 0):,.2f}
Key Support: ${risk_metrics.get('support_level', 0):,.2f}
Key Resistance: ${risk_metrics.get('resistance_level', 0):,.2f}
Estimated R/R Ratio: {risk_metrics.get('risk_reward_ratio', 0):.2f}:1

=== FUNDAMENTAL BACKDROP ===
Total Crypto Market Cap: ${metrics.get('market_cap_usd', 0):,.0f}
24h Global Volume: ${metrics.get('volume_usd', 0):,.0f}
Bitcoin Hash Rate: {metrics.get('hash_rate', 'N/A')} EH/s
Ethereum Network Nodes: {metrics.get('nodes', 'N/A')}

ADDITIONAL CONTEXT FOR DECISION MAKING:
- Current market regime appears {trend_info.get('trend_direction', 'unclear').upper()}
- Volatility is {trend_info.get('volatility', 'unknown').upper()}
- Order flow shows {'BID' if liquidity.get('bid_ask_imbalance_pct', 0) > 0 else 'ASK'} dominance
- Multi-timeframe alignment is {'STRONG' if abs(confluence_analysis.get('confluence_score', 0)) > 30 else 'WEAK'}

SIGNAL QUALITY STANDARDS:
- confidence (80-95%): Multiple confluences, clear trend, low risk
- confidence (60-79%): Some confluences, moderate risk
- confidence (40-59%): Few confluences, high risk, consider passing

=== NEWS SENTIMENT ANALYSIS ===
News Bias: {news_bias}
Bullish Articles: {bullish_news} | Bearish Articles: {bearish_news} | Total: {len(news_summaries)}
Key Headlines: {' | '.join([f"{news.get('title', '')[:50]}..." for news in news_summaries[:3]])}

{learning_addition}
Based on the performance learning data above, generate your signal with improved accuracy.

VALIDATION CHECK: Before responding, verify your signal follows these rules.
CRITICAL TRADING RULES - NEVER VIOLATE:
🔴 BUY Signals: Take Profit > Entry Price > Stop Loss
🔴 SELL Signals: Stop Loss > Entry Price > Take Profit  
🔴 Entry prices must be realistic 


=== TRADING DIRECTIVE ===
Based on the comprehensive analysis above, provide your institutional-grade trading recommendation ONLY in the following JSON format:

{{
- Signal (Buy, Sell, Hold)
- Entry Price
- Take Profit Targets (list of 1-2)
- Stop Loss
- Confidence (0-100%)
- Scenario (Breakout | Reversal | Trend Continuation | Range-Bound)
- Trade Setup Type (Swing | Scalping | Momentum | Mean Reversion | Other)
- Reasoning
- Relevant News Headlines (summarize 2-3 most impactful)
}}

IMPORTANT:
Do NOT output any text, explanations, reasoning steps, or comments outside the JSON.  
"""
    return run_claude(prompt, api_key)

# -------------------- File Save --------------------
def save_signal_to_file(signal, symbol, filename='signals_claude_log.json'):
    signal['timestamp'] = datetime.utcnow().isoformat()
    signal['symbol'] = symbol
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(signal)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# -------------------- Discord Notification --------------------
def send_discord_notification(webhook_url, signal_data):
    signal_type = signal_data['Signal'].lower()

    image_url = {
        'buy': "https://i.imgur.com/UPEAkWm.png",
        'hold': "https://i.imgur.com/20UDHex.png"
    }.get(signal_type, "https://i.imgur.com/k5jtXNh.png")
    color = {
        'buy': 0x00FF00,
        'hold': 0x7289DA
    }.get(signal_type, 0xFF0000)

        # Fix the news headlines formatting
    news_headlines = signal_data.get('Relevant News Headlines', [])
    
    # Ensure it's a list, not a string
    if isinstance(news_headlines, str):
        # If it's a string, split by common delimiters or wrap in a list
        if news_headlines.strip():
            news_headlines = [news_headlines]
        else:
            news_headlines = ["No significant news headlines available for analysis"]
    elif not news_headlines or (isinstance(news_headlines, list) and len(news_headlines) == 0):
        news_headlines = ["No significant news headlines available for analysis"]
    
    # Format the headlines properly
    formatted_headlines = []
    for headline in news_headlines:
        if isinstance(headline, str) and headline.strip():
            # Limit headline length to prevent Discord embed limits
            truncated_headline = headline[:100] + "..." if len(headline) > 100 else headline
            formatted_headlines.append(f"• {truncated_headline}")
    
    # If no valid headlines after processing
    if not formatted_headlines:
        formatted_headlines = ["• No significant news headlines available for analysis"]
    
    # Join the headlines with newlines
    headlines_text = '\n'.join(formatted_headlines)

    fields = [
        {"name": "Symbol", "value": signal_data['symbol'], "inline": True},
        # Include Entry Price only if not hold
        *([] if signal_type == 'hold' else [{"name": "Entry Price", "value": f"${signal_data['Entry Price']}", "inline": True}]),
        {"name": "Confidence", "value": f"{signal_data['Confidence']}%", "inline": True},
        # Include Take Profit Targets only if not hold
        *([] if signal_type == 'hold' else [{"name": "Take Profit Targets", "value": ', '.join([f"${tp}" for tp in signal_data['Take Profit Targets']]), "inline": False}]),
        # Include Stop Loss only if not hold
        *([] if signal_type == 'hold' else [{"name": "Stop Loss", "value": f"${signal_data['Stop Loss']}", "inline": True}]),
         # Add Scenario
        *([] if signal_type == 'hold' else [{"name": "Scenario", "value": signal_data.get('Scenario', 'N/A'), "inline": True}]),
        # Add Trade Setup Type
        *([] if signal_type == 'hold' else [{"name": "Trade Setup Type", "value": signal_data.get('Trade Setup Type', 'N/A'), "inline": True}]),
        {"name": "Reasoning", "value": signal_data['Reasoning'], "inline": False},
        {"name": "🚨 Relevant News Headlines", "value": headlines_text, "inline": False}
    ]

    embed = {
        "title": f"📈 Trade Signal: {signal_data['Signal']}",
        "color": color,
        "fields": fields,
        "image": {"url": image_url},
        "footer": { "text": "Always Do Your Own Research!" }
    }

    data = {"username": "Crypto AI MAX", "embeds": [embed]}
    headers = {"Content-Type": "application/json"}
    requests.post(webhook_url, data=json.dumps(data), headers=headers)

# -------------------- Main Flow --------------------
if __name__ == "__main__":
    start_time = datetime.utcnow()
    NEWS_API_KEY = '33aea78f5a054aea8827002fa0ecdbca'
    ANTHROPIC_API_KEY = 'test'
    DISCORD_WEBHOOK_URL = 'https://ptb.discord.com/api/webhooks/1375475186626859040/6kIQy-YoU-ZcydVgi9USedoPucu8cfnpT0cHwktMfAvl25X-TSvzCuENQsnb9KL5Hm2X'

    market_metrics = fetch_market_cap_and_volume()
    market_metrics['hash_rate'] = fetch_hash_rate()
    market_metrics['nodes'] = fetch_network_security().get('nodes')
    news = fetch_news(NEWS_API_KEY)
    news_summary = analyze_news_with_llama(news)

    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        multi_df = fetch_multi_timeframe_ohlcv(symbol)
        df = fetch_ohlcv(symbol)
        df = compute_indicators(df)

        df_1h = multi_df.get('1h')

        order_book = fetch_order_book(symbol)
        liquidity_zones = analyze_liquidity_zones(order_book)
        volume_profile = calculate_volume_profile(order_book)
        trend_info = detect_trend(df)

        signal = generate_trade_signal(
        df=df,
        df_1h=df_1h,
        news_summaries=news_summary,
        metrics=market_metrics,
        liquidity=liquidity_zones,
        volume_profile=volume_profile,
        multi_df=multi_df,
        trend_info=trend_info,
        api_key=ANTHROPIC_API_KEY  
    )

    save_signal_to_file(signal, symbol)
    send_discord_notification(DISCORD_WEBHOOK_URL, signal)

print("Start time (UTC minute):", start_time.strftime("%Y-%m-%d %H:%M"))
