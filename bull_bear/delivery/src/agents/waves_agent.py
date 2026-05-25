#!/usr/bin/env python3
"""
艾略特波浪理论自动识别系统 - StockAgent (修正版)
支持从 Yahoo Finance / akshare 自动拉取股票数据，提供波浪分析服务
"""

import sys
import json
import argparse
import warnings
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 尝试导入网络数据获取库
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# ======================== 波浪分析核心代码 ========================

class WaveType(Enum):
    IMPULSE = "推动浪"
    CORRECTIVE = "调整浪"
    UNKNOWN = "未知"

class WaveDegree(Enum):
    GRAND_SUPERCYCLE = "超级循环浪"
    SUPERCYCLE = "循环浪"
    CYCLE = "大浪"
    PRIMARY = "基本浪"
    INTERMEDIATE = "中浪"
    MINOR = "小浪"
    MINUTE = "细浪"

@dataclass
class PivotPoint:
    index: int
    price: float
    date: str   # 始终存储字符串格式的日期
    is_high: bool

    def __repr__(self):
        point_type = "高点" if self.is_high else "低点"
        return f"{point_type}[{self.date}]: {self.price:.2f}"

@dataclass
class Wave:
    wave_num: Union[int, str]   # 推动浪用 int (1,2,3,4,5)，调整浪用 str ('A','B','C')
    start: PivotPoint
    end: PivotPoint
    degree: WaveDegree
    wave_type: WaveType
    sub_waves: List['Wave']

    @property
    def price_change(self) -> float:
        return self.end.price - self.start.price

    @property
    def price_change_pct(self) -> float:
        return (self.end.price - self.start.price) / self.start.price * 100

    @property
    def duration(self) -> int:
        return self.end.index - self.start.index

    def __repr__(self):
        direction = "↑" if self.price_change > 0 else "↓"
        return f"浪{self.wave_num}({self.degree.value}){direction}: {self.start.price:.2f}→{self.end.price:.2f} ({self.price_change_pct:+.2f}%)"

class ZigZagIndicator:
    def __init__(self, deviation: float = 0.05, depth: int = 10):
        self.deviation = deviation
        self.depth = depth

    def calculate(self, highs: np.ndarray, lows: np.ndarray, dates: List[str]) -> List[PivotPoint]:
        pivots = []
        n = len(highs)
        if n < 3:
            return pivots

        trend = 0
        last_pivot_idx = 0
        last_pivot_price = (highs[0] + lows[0]) / 2
        last_pivot_is_high = None

        for i in range(1, n):
            if trend == 0:
                if highs[i] > last_pivot_price * (1 + self.deviation):
                    trend = 1
                    pivots.append(PivotPoint(0, last_pivot_price, dates[0], False))
                    last_pivot_idx = i
                    last_pivot_price = highs[i]
                    last_pivot_is_high = True
                elif lows[i] < last_pivot_price * (1 - self.deviation):
                    trend = -1
                    pivots.append(PivotPoint(0, last_pivot_price, dates[0], True))
                    last_pivot_idx = i
                    last_pivot_price = lows[i]
                    last_pivot_is_high = False
            elif trend == 1:
                if highs[i] > last_pivot_price:
                    last_pivot_idx = i
                    last_pivot_price = highs[i]
                elif lows[i] < last_pivot_price * (1 - self.deviation):
                    pivots.append(PivotPoint(last_pivot_idx, last_pivot_price, dates[last_pivot_idx], True))
                    trend = -1
                    last_pivot_idx = i
                    last_pivot_price = lows[i]
                    last_pivot_is_high = False
            elif trend == -1:
                if lows[i] < last_pivot_price:
                    last_pivot_idx = i
                    last_pivot_price = lows[i]
                elif highs[i] > last_pivot_price * (1 + self.deviation):
                    pivots.append(PivotPoint(last_pivot_idx, last_pivot_price, dates[last_pivot_idx], False))
                    trend = 1
                    last_pivot_idx = i
                    last_pivot_price = highs[i]
                    last_pivot_is_high = True

        if last_pivot_is_high is not None:
            pivots.append(PivotPoint(last_pivot_idx, last_pivot_price, dates[last_pivot_idx], last_pivot_is_high))
        return pivots

class FibonacciCalculator:
    RATIOS = {'0.236':0.236,'0.382':0.382,'0.5':0.5,'0.618':0.618,'0.786':0.786,'1.0':1.0,'1.272':1.272,'1.618':1.618,'2.618':2.618}
    @staticmethod
    def retracement(start: float, end: float, ratio: float) -> float:
        return end - (end - start) * ratio
    @staticmethod
    def extension(start: float, end: float, ratio: float) -> float:
        return end + (end - start) * ratio
    @staticmethod
    def check_ratio(actual: float, expected: float, tolerance: float = 0.1) -> bool:
        return abs(actual - expected) / expected < tolerance

class WaveValidator:
    def __init__(self):
        self.fib = FibonacciCalculator()

    def validate_impulse(self, waves: List[Wave]) -> Tuple[bool, float, str]:
        if len(waves) != 5:
            return False, 0, "推动浪必须有5个子浪"
        w1,w2,w3,w4,w5 = waves
        score = 1.0
        messages = []
        if w2.end.price < w1.start.price:
            return False, 0, "浪2跌破浪1起点"
        w1_len = abs(w1.price_change)
        w3_len = abs(w3.price_change)
        w5_len = abs(w5.price_change)
        if w3_len < min(w1_len, w5_len):
            return False, 0, "浪3是最短推动浪"
        if w4.end.price < w1.end.price and w4.start.price > w1.start.price:
            if w4.end.price < w1.end.price * 0.98:
                score *= 0.8
                messages.append("浪4与浪1轻微重叠")
        if w3.wave_type == WaveType.CORRECTIVE:
            return False, 0, "浪3不能是调整浪"
        w2_retrace = abs(w2.price_change) / w1_len if w1_len>0 else 0
        if not (0.382 <= w2_retrace <= 0.786):
            score *= 0.9
            messages.append(f"浪2回调比例异常({w2_retrace:.3f})")
        w3_ratio = w3_len / w1_len if w1_len>0 else 0
        if not (1.0 <= w3_ratio <= 2.618):
            score *= 0.9
            messages.append(f"浪3/浪1比例异常({w3_ratio:.3f})")
        w5_ratio = w5_len / w1_len if w1_len>0 else 0
        if not (0.5 <= w5_ratio <= 2.0):
            score *= 0.9
        return True, score, "; ".join(messages) if messages else "符合推动浪结构"

    def validate_corrective(self, waves: List[Wave]) -> Tuple[bool, float, str]:
        if len(waves) != 3:
            return False, 0, "ABC调整必须有3个子浪"
        a,b,c = waves
        score = 1.0
        messages = []
        if b.end.price > a.start.price:
            score *= 0.9
            messages.append("B浪反弹过深，可能是平台形")
        a_len = abs(a.price_change)
        c_len = abs(c.price_change)
        if a_len > 0:
            ratio = c_len / a_len
            if not (0.618 <= ratio <= 2.618):
                score *= 0.9
                messages.append(f"C/A比例异常({ratio:.3f})")
        return True, score, "; ".join(messages) if messages else "符合调整浪结构"

class ElliottWaveAnalyzer:
    def __init__(self, deviation: float = 0.05):
        self.zigzag = ZigZagIndicator(deviation=deviation)
        self.validator = WaveValidator()
        self.fib = FibonacciCalculator()

    def analyze(self, df: pd.DataFrame) -> Dict:
        # 确保日期列为字符串
        dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
        pivots = self.zigzag.calculate(df['high'].values, df['low'].values, dates)
        impulse_result = self._match_impulse(pivots)
        corrective_result = self._match_corrective(pivots)
        current_position = self._determine_position(pivots, impulse_result, corrective_result, df)
        return {
            'pivots': [{'index':p.index,'price':p.price,'date':p.date,'is_high':p.is_high} for p in pivots],
            'impulse': impulse_result,
            'corrective': corrective_result,
            'current_position': current_position,
            'prediction': self._generate_prediction(current_position, df)
        }

    def _match_impulse(self, pivots: List[PivotPoint], start_idx: int = 0) -> Optional[Dict]:
        if len(pivots) - start_idx < 5:
            return None
        best_score = 0
        best_waves = None
        for i in range(start_idx, len(pivots)-4):
            candidate_waves = []
            current_idx = i
            for wave_num in [1,2,3,4,5]:
                if current_idx >= len(pivots)-1:
                    break
                start = pivots[current_idx]
                end_idx = self._find_wave_end(pivots, current_idx, wave_num)
                if end_idx is None:
                    break
                end = pivots[end_idx]
                wave = Wave(wave_num, start, end, WaveDegree.INTERMEDIATE,
                            WaveType.IMPULSE if wave_num in [1,3,5] else WaveType.CORRECTIVE, [])
                candidate_waves.append(wave)
                current_idx = end_idx
            if len(candidate_waves) == 5:
                valid, score, msg = self.validator.validate_impulse(candidate_waves)
                if valid and score > best_score:
                    best_score = score
                    best_waves = candidate_waves
        if best_waves:
            return {
                'waves': [{'wave_num':w.wave_num,'start':asdict(w.start),'end':asdict(w.end),
                           'degree':w.degree.value,'wave_type':w.wave_type.value,'price_change':w.price_change,
                           'price_change_pct':w.price_change_pct,'duration':w.duration} for w in best_waves],
                'score': best_score,
                'start_date': best_waves[0].start.date,
                'end_date': best_waves[-1].end.date,
                'total_change': sum(w.price_change for w in best_waves)
            }
        return None

    def _match_corrective(self, pivots: List[PivotPoint], start_idx: int = 0) -> Optional[Dict]:
        if len(pivots) - start_idx < 3:
            return None
        best_score = 0
        best_waves = None
        for i in range(start_idx, len(pivots)-2):
            candidate_waves = []
            current_idx = i
            for wave_num, label in enumerate(['A','B','C'], 1):
                if current_idx >= len(pivots)-1:
                    break
                start = pivots[current_idx]
                end_idx = self._find_wave_end(pivots, current_idx, wave_num, is_corrective=True)
                if end_idx is None:
                    break
                end = pivots[end_idx]
                wave = Wave(label, start, end, WaveDegree.INTERMEDIATE, WaveType.CORRECTIVE, [])
                candidate_waves.append(wave)
                current_idx = end_idx
            if len(candidate_waves) == 3:
                valid, score, msg = self.validator.validate_corrective(candidate_waves)
                if valid and score > best_score:
                    best_score = score
                    best_waves = candidate_waves
        if best_waves:
            return {
                'waves': [{'wave_num':w.wave_num,'start':asdict(w.start),'end':asdict(w.end),
                           'degree':w.degree.value,'wave_type':w.wave_type.value,'price_change':w.price_change,
                           'price_change_pct':w.price_change_pct,'duration':w.duration} for w in best_waves],
                'score': best_score,
                'start_date': best_waves[0].start.date,
                'end_date': best_waves[-1].end.date,
                'total_change': sum(w.price_change for w in best_waves)
            }
        return None

    def _find_wave_end(self, pivots: List[PivotPoint], start_idx: int, wave_num: Union[int, str], is_corrective: bool = False) -> Optional[int]:
        start = pivots[start_idx]
        min_move = 0.03
        for i in range(start_idx+1, len(pivots)):
            end = pivots[i]
            change_pct = (end.price - start.price) / start.price
            if not is_corrective and isinstance(wave_num, int) and wave_num in [1,3,5]:
                # 推动浪 1,3,5 需要同向
                if start.is_high and not end.is_high:
                    continue
                if not start.is_high and end.is_high:
                    continue
                if abs(change_pct) > min_move:
                    return i
            else:
                # 调整浪或推动浪中的2,4浪，需要反向
                if start.is_high == end.is_high:
                    continue
                if abs(change_pct) > min_move * 0.5:
                    return i
        return None

    def _determine_position(self, pivots: List[PivotPoint], impulse: Optional[Dict], corrective: Optional[Dict], df: pd.DataFrame) -> Dict:
        current_price = df['close'].iloc[-1]
        current_date = df['date'].iloc[-1]
        recent_pivots = pivots[-5:] if len(pivots)>=5 else pivots
        position = {'current_price': current_price, 'current_date': str(current_date),
                    'primary_degree': '未知', 'sub_degree': '未知', 'wave_location': '未知', 'confidence': 0}
        if impulse and impulse['score'] > 0.7:
            waves = impulse['waves']
            last_wave = waves[-1]
            if last_wave['wave_num'] == 5:
                if current_price > last_wave['end']['price'] * 0.98:
                    position['primary_degree'] = 'III浪或V浪'
                    position['sub_degree'] = '5浪末端或已结束'
                    position['wave_location'] = '推动浪末期，警惕调整'
                    position['confidence'] = impulse['score']
        if corrective and corrective['score'] > 0.7:
            waves = corrective['waves']
            last_wave = waves[-1]
            if last_wave['wave_num'] == 'C':
                position['primary_degree'] = 'IV浪或II浪'
                position['sub_degree'] = 'C浪末端'
                position['wave_location'] = '调整浪末期，准备买入'
                position['confidence'] = corrective['score']
        if len(recent_pivots) >= 2:
            last_pivot = recent_pivots[-1]
            if last_pivot.is_high:
                if current_price < last_pivot.price * 0.95:
                    position['wave_location'] = '可能处于A浪或C浪下跌中'
                else:
                    position['wave_location'] = '可能在B浪反弹中'
            else:
                if current_price > last_pivot.price * 1.05:
                    position['wave_location'] = '可能处于1浪或3浪上涨中'
                else:
                    position['wave_location'] = '可能在2浪或4浪调整中'
        return position

    def _generate_prediction(self, position: Dict, df: pd.DataFrame) -> Dict:
        current_price = position['current_price']
        prediction = {'short_term': '', 'medium_term': '', 'targets': [], 'stop_loss': None, 'action': '观望'}
        location = position.get('wave_location', '')
        if '调整浪末期' in location or 'C浪' in location:
            prediction['short_term'] = '调整接近尾声，关注止跌信号'
            prediction['medium_term'] = '有望开启新一轮推动浪'
            prediction['targets'] = [current_price*1.05, current_price*1.12, current_price*1.20]
            prediction['stop_loss'] = current_price*0.95
            prediction['action'] = '逢低买入'
        elif '推动浪末期' in location:
            prediction['short_term'] = '推动浪可能结束，注意止盈'
            prediction['medium_term'] = '调整后仍有上涨空间'
            prediction['targets'] = [current_price*0.95, current_price*0.90, current_price*0.85]
            prediction['stop_loss'] = current_price*1.03
            prediction['action'] = '减仓或观望'
        else:
            prediction['short_term'] = '方向不明，等待信号'
            prediction['medium_term'] = '需要更多数据确认浪型'
            prediction['action'] = '观望'
        return prediction

# ======================== 数据拉取模块 ========================

def fetch_data_yfinance(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    if not YFINANCE_AVAILABLE:
        return None
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df = df.reset_index()
        df.rename(columns={'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date','open','high','low','close']]
        df = df.dropna()
        return df
    except Exception as e:
        print(f"yfinance 获取数据失败: {e}")
        return None

def fetch_data_akshare(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    if not AKSHARE_AVAILABLE:
        return None
    try:
        symbol = ticker.split('.')[0]
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date.replace('-',''), end_date=end_date.replace('-',''), adjust="qfq")
        if df.empty:
            return None
        df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date','open','high','low','close']]
        df = df.dropna()
        return df
    except Exception as e:
        print(f"akshare 获取数据失败: {e}")
        return None

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = fetch_data_yfinance(ticker, start_date, end_date)
    if df is not None:
        print(f"使用 yfinance 获取 {ticker} 数据成功")
        return df
    df = fetch_data_akshare(ticker, start_date, end_date)
    if df is not None:
        print(f"使用 akshare 获取 {ticker} 数据成功")
        return df
    raise RuntimeError(f"无法获取股票数据，请检查网络或安装 yfinance/akshare，ticker: {ticker}")

# ======================== 命令行接口 ========================

def run_analysis(ticker: str, start_date: str, end_date: str, deviation: float = 0.05) -> Dict:
    print(f"正在获取 {ticker} 数据 {start_date} -> {end_date} ...")
    df = fetch_stock_data(ticker, start_date, end_date)
    print(f"获取到 {len(df)} 条日线数据")
    analyzer = ElliottWaveAnalyzer(deviation=deviation)
    result = analyzer.analyze(df)
    result['ticker'] = ticker
    result['data_range'] = {'start': start_date, 'end': end_date}
    return result

def print_result(result: Dict):
    pos = result['current_position']
    pred = result['prediction']
    print("\n" + "="*60)
    print(f"股票: {result.get('ticker','未知')}")
    print(f"当前价格: {pos['current_price']:.2f} ({pos['current_date']})")
    print(f"波浪位置: {pos['wave_location']}")
    print(f"主要浪级: {pos['primary_degree']}")
    print(f"置信度: {pos['confidence']:.2%}")
    print("\n预测建议:")
    print(f"  短期: {pred['short_term']}")
    print(f"  中期: {pred['medium_term']}")
    print(f"  操作: {pred['action']}")
    if pred['targets']:
        print("  目标价位:", ", ".join(f"{t:.2f}" for t in pred['targets']))
    if pred['stop_loss']:
        print(f"  止损位: {pred['stop_loss']:.2f}")
    if result.get('impulse'):
        print("\n识别到推动浪结构 (评分: {:.2%})".format(result['impulse']['score']))
        for w in result['impulse']['waves']:
            print(f"  浪{w['wave_num']}: {w['start']['price']:.2f} -> {w['end']['price']:.2f} ({w['price_change_pct']:+.2f}%)")
    if result.get('corrective'):
        print("\n识别到调整浪结构 (评分: {:.2%})".format(result['corrective']['score']))
        for w in result['corrective']['waves']:
            print(f"  浪{w['wave_num']}: {w['start']['price']:.2f} -> {w['end']['price']:.2f} ({w['price_change_pct']:+.2f}%)")
    print("="*60)

# ======================== HTTP 服务 ========================

def start_http_server(host: str = '0.0.0.0', port: int = 5000):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask 未安装，无法启动 HTTP 服务。请安装: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    @app.route('/analyze', methods=['POST'])
    def analyze_endpoint():
        data = request.get_json()
        if not data:
            return jsonify({'error': '请提供 JSON 请求体'}), 400
        ticker = data.get('ticker')
        start = data.get('start') or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        end = data.get('end') or datetime.now().strftime('%Y-%m-%d')
        deviation = data.get('deviation', 0.05)

        if not ticker:
            return jsonify({'error': '缺少 ticker 参数'}), 400

        try:
            result = run_analysis(ticker, start, end, deviation)
            # 转换所有日期字段为字符串
            result['current_position']['current_date'] = str(result['current_position']['current_date'])
            if result.get('impulse'):
                result['impulse']['start_date'] = str(result['impulse']['start_date'])
                result['impulse']['end_date'] = str(result['impulse']['end_date'])
            if result.get('corrective'):
                result['corrective']['start_date'] = str(result['corrective']['start_date'])
                result['corrective']['end_date'] = str(result['corrective']['end_date'])
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})

    print(f"启动艾略特波浪分析服务: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

# ======================== 主入口 ========================

def main():
    parser = argparse.ArgumentParser(description='艾略特波浪理论股票分析 Agent')
    parser.add_argument('--ticker', type=str, help='股票代码，例如 AAPL, 000001.SS')
    parser.add_argument('--start', type=str, default=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'), help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='结束日期 YYYY-MM-DD')
    parser.add_argument('--deviation', type=float, default=0.05, help='ZigZag 波动阈值，默认0.05')
    parser.add_argument('--serve', action='store_true', help='启动 HTTP 服务模式')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='HTTP 服务监听地址')
    parser.add_argument('--port', type=int, default=5000, help='HTTP 服务端口')
    parser.add_argument('--json', action='store_true', help='以 JSON 格式输出结果')

    args = parser.parse_args()

    if args.serve:
        start_http_server(args.host, args.port)
    elif args.ticker:
        result = run_analysis(args.ticker, args.start, args.end, args.deviation)
        if args.json:
            # 转换日期对象
            result['current_position']['current_date'] = str(result['current_position']['current_date'])
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python stock_agent.py --ticker AAPL --start 2024-01-01")
        print("  python stock_agent.py --ticker 000001.SS --serve --port 8080")
        print("  curl -X POST http://localhost:5000/analyze -H 'Content-Type: application/json' -d '{\"ticker\":\"AAPL\"}'")

if __name__ == '__main__':
    main()