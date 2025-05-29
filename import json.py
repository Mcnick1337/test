import json
import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np
from typing import Dict, List, Optional

class TradingLearningSystem:
    """
    A learning system that tracks trading signal performance and provides
    feedback to improve future predictions.
    """
    
    def __init__(self, signals_file='signals_log.json', performance_file='signal_performance.json'):
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
        
        # Performance by confidence level
        high_conf = [p for p in evaluated_signals if p.get('confidence', 50) >= 80]
        med_conf = [p for p in evaluated_signals if 60 <= p.get('confidence', 50) < 80]
        low_conf = [p for p in evaluated_signals if p.get('confidence', 50) < 60]
        
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

# Integration function to add to the main trading script
def integrate_learning_system_to_main_script():
    """
    This function shows how to integrate the learning system into the main trading script.
    Add this to your generate_trade_signal function.
    """
    
    # Add this at the beginning of generate_trade_signal function:
    learning_system = TradingLearningSystem()
    learning_system.update_signal_performance()
    learning_addition = learning_system.generate_learning_prompt_addition()
    
    # Then modify your prompt to include the learning addition:
    # prompt = f"""
    # {your_existing_prompt}
    # 
    # {learning_addition}
    # 
    # Based on the performance learning data above, generate your signal with improved accuracy.
    # """
    
    return learning_addition