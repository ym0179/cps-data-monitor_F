"""
MAS Finance Dashboard - Flask Application
PythonAnywhere 배포용
"""
from flask import Flask, render_template, jsonify, request
import sys
import os

# 기존 로직 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# ============================================
# 메인 라우트
# ============================================

@app.route('/')
def index():
    """메인 홈페이지"""
    # 나중에 실제 데이터로 교체
    market_data = {
        'dow_futures': {'value': '49,112.00', 'change': '-55.00', 'percent': '-0.11%', 'up': False},
        'nasdaq_futures': {'value': '26,223.75', 'change': '+67.50', 'percent': '+0.26%', 'up': True},
        'sp500_futures': {'value': '7,013.50', 'change': '+6.25', 'percent': '+0.09%', 'up': True},
        'dow': {'value': '49,015.60', 'change': '+12.19', 'percent': '+0.02%', 'up': True},
        'nasdaq': {'value': '23,857', 'change': '+40.35', 'percent': '+0.17%', 'up': True},
    }
    return render_template('index.html', market_data=market_data)


@app.route('/search')
def search():
    """검색 페이지"""
    query = request.args.get('q', '')
    return render_template('search.html', query=query)


@app.route('/earnings')
def earnings():
    """Earnings Event Trading 페이지 (기존 Idio Score 기능)"""
    return render_template('earnings.html')


@app.route('/etf')
def etf():
    """Active ETF Analysis 페이지"""
    return render_template('etf.html')


@app.route('/market')
def market():
    """MS Monitoring 페이지"""
    return render_template('market.html')


@app.route('/favorites')
def favorites():
    """관심 종목 페이지"""
    return render_template('favorites.html')


# ============================================
# API 엔드포인트 (AJAX용)
# ============================================

@app.route('/api/market-data')
def api_market_data():
    """시장 데이터 API"""
    # 나중에 실제 데이터 fetching 로직 추가
    data = {
        'dow_futures': {'value': 49112.00, 'change': -55.00},
        'nasdaq_futures': {'value': 26223.75, 'change': 67.50},
    }
    return jsonify(data)


@app.route('/api/search')
def api_search():
    """종목 검색 API"""
    query = request.args.get('q', '')
    # 나중에 실제 검색 로직 추가
    results = []
    return jsonify(results)


# ============================================
# 실행
# ============================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
