from flask import Flask, render_template, jsonify, request
import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cps-strategy-team-2026'

# =============================================================================
# 데이터 수집 함수들
# =============================================================================

def fetch_statcounter_data(metric="browser", device="desktop", from_year="2019", from_month="01"):
    """StatCounter 데이터 수집"""
    now = datetime.now()
    to_year = now.year
    to_month = now.month

    base_url = "https://gs.statcounter.com/chart.php"

    stat_type_map = {
        "browser": "browser",
        "search_engine": "search_engine",
        "os": "os_combined"
    }

    params = {
        "device_hidden": device,
        "multi-device": "true",
        "statType_hidden": stat_type_map.get(metric, "browser"),
        "region_hidden": "ww",
        "granularity": "monthly",
        "fromInt": f"{from_year}{from_month}",
        "toInt": f"{to_year}{to_month:02d}",
        "csv": "1"
    }

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(base_url, params=params, headers=headers, verify=False, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
            df.set_index('Date', inplace=True)
            return df
    except Exception as e:
        print(f"StatCounter Error: {e}")

    return pd.DataFrame()

def fetch_etf_data(etf_idx="22"):
    """TIMEFOLIO ETF 데이터 수집"""
    try:
        from logic.etf import ActiveETFMonitor
        monitor = ActiveETFMonitor(
            url=f"https://timefolioetf.co.kr/m11_view.php?idx={etf_idx}",
            etf_name="ETF"
        )
        today = datetime.now().strftime("%Y-%m-%d")
        df = monitor.get_portfolio_data(today)
        return df
    except Exception as e:
        print(f"ETF Error: {e}")
        return pd.DataFrame()

# =============================================================================
# 라우트
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/earnings')
def earnings():
    return render_template('earnings.html')

@app.route('/etf')
def etf():
    return render_template('etf.html')

# =============================================================================
# API 엔드포인트
# =============================================================================

@app.route('/api/statcounter/<metric>')
def api_statcounter(metric):
    """StatCounter 데이터 API"""
    device = request.args.get('device', 'desktop')

    df = fetch_statcounter_data(metric=metric, device=device)

    if df.empty:
        return jsonify({'error': 'No data available'}), 404

    # 최근 24개월만
    df = df.tail(24)

    # JSON 형태로 변환
    data = {
        'dates': df.index.tolist(),
        'series': {}
    }

    for col in df.columns:
        data['series'][col] = df[col].tolist()

    return jsonify(data)

@app.route('/api/os-rivalry')
def api_os_rivalry():
    """OS 시장 점유율 API (Android vs iOS)"""
    device = request.args.get('device', 'mobile')

    df = fetch_statcounter_data(metric="os", device=device, from_year="2015")

    if df.empty:
        return jsonify({'error': 'No data available'}), 404

    # Android, iOS만 필터
    targets = ['Android', 'iOS']
    valid_cols = [c for c in df.columns if c in targets]

    if not valid_cols:
        return jsonify({'error': 'No Android/iOS data'}), 404

    df = df[valid_cols].tail(60)  # 최근 5년

    data = {
        'dates': df.index.tolist(),
        'series': {}
    }

    for col in df.columns:
        data['series'][col] = df[col].tolist()

    return jsonify(data)

@app.route('/api/etf/list')
def api_etf_list():
    """ETF 목록 API"""
    etf_categories = {
        "해외주식형": {
            "글로벌탑픽": "22", "글로벌바이오": "9", "우주테크&방산": "20",
            "S&P500": "5", "나스닥100": "2", "글로벌AI": "6",
            "차이나AI": "19", "미국배당다우존스": "18"
        },
        "국내주식형": {
            "K신재생에너지": "16", "K바이오": "13", "Korea플러스배당": "12",
            "코스피": "11", "코리아밸류업": "15"
        }
    }
    return jsonify(etf_categories)

@app.route('/api/etf/portfolio/<idx>')
def api_etf_portfolio(idx):
    """ETF 포트폴리오 데이터 API"""
    df = fetch_etf_data(idx)

    if df.empty:
        return jsonify({'error': 'No data available'}), 404

    # DataFrame을 JSON으로
    records = df.to_dict('records')
    return jsonify({
        'data': records,
        'total': len(records)
    })

if __name__ == '__main__':
    app.run(debug=True)
