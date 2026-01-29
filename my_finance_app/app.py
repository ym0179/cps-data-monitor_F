"""
=============================================================================
고객상품전략팀 Flask Application
=============================================================================
3가지 주요 기능:
1. MS Monitoring: StatCounter API로 브라우저/OS 시장점유율 분석
2. Active ETF: TIMEFOLIO 웹사이트에서 ETF 포트폴리오 크롤링
3. Earnings Trading: Goldman Sachs 방법론 기반 Idio Score 계산
=============================================================================
"""

from flask import Flask, render_template, jsonify, request, Response
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import json
from bs4 import BeautifulSoup
import urllib3
from etf_monitor import TimeETFMonitor, KiwoomETFMonitor
from logic.earnings import (
    fetch_yahoo_earnings_calendar,
    fetch_analyst_consensus,
    calculate_earnings_metrics,
    calculate_trade_alert
)

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cps-strategy-team-2026'

# =============================================================================
# 파일 기반 캐시 (Scheduled Task로 매일 갱신)
# =============================================================================
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

def load_cache_file(filename):
    """캐시 파일에서 데이터 로드"""
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"캐시 파일 로드 오류: {e}")
    return None


# =============================================================================
# 1. MS Monitoring (StatCounter)
# =============================================================================
#
# [API 소스]
# - URL: https://gs.statcounter.com/chart.php
# - 형식: CSV 다운로드
#
# [파라미터]
# - device_hidden: "desktop" | "mobile" | "tablet" | "desktop+mobile"
# - statType_hidden: "browser" | "os_combined" | "search_engine"
# - region_hidden: "ww" (Worldwide)
# - granularity: "monthly"
# - fromInt: "201901" (시작년월 YYYYMM)
# - toInt: "202601" (종료년월 YYYYMM)
# - csv: "1" (CSV 형식)
#
# [반환 데이터]
# CSV 형식으로 각 브라우저/OS별 월별 점유율(%) 반환
# 예: Date, Chrome, Safari, Edge, Firefox, Other...
#     2024-01, 65.12, 18.45, 5.23, 2.89, 8.31
# =============================================================================

def fetch_statcounter_data(metric="browser", device="desktop", from_year="2019", from_month="01"):
    """
    StatCounter에서 시장점유율 데이터 수집
    (Search Engine은 캐시 파일 사용, 다른 메트릭은 직접 호출)

    Parameters:
    -----------
    metric : str
        - "browser": 브라우저 시장점유율 (Chrome, Safari, Edge 등)
        - "os": 운영체제 시장점유율 (Windows, macOS, Android, iOS 등)
        - "search_engine": 검색엔진 시장점유율 (Google, Bing, Yahoo 등)

    device : str
        - "desktop": 데스크톱만
        - "mobile": 모바일만
        - "desktop+mobile": 데스크톱 + 모바일 통합

    Returns:
    --------
    DataFrame with Date index and market share % for each category
    """
    now = datetime.now()
    to_year = now.year
    to_month = now.month

    base_url = "https://gs.statcounter.com/chart.php"

    # StatCounter API는 statType_hidden 값으로 메트릭 구분
    stat_type_map = {
        "browser": "browser",           # 브라우저
        "search_engine": "search_engine", # 검색엔진
        "os": "os"                       # 운영체제 (os_combined가 아님)
    }

    # from_month를 2자리로 보장
    from_month = str(from_month).zfill(2)

    params = {
        "device_hidden": device,         # 디바이스 종류
        "multi-device": "true",          # 멀티 디바이스 지원
        "statType_hidden": stat_type_map.get(metric, "browser"),
        "region_hidden": "ww",           # Worldwide (전세계)
        "granularity": "monthly",        # 월별 데이터
        "fromInt": f"{from_year}{from_month}",  # 시작 년월 (예: 201901)
        "toInt": f"{to_year}{to_month:02d}",    # 종료 년월
        "csv": "1"                       # CSV 형식으로 반환
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, verify=False, timeout=30)
        if response.status_code == 200:
            # CSV 텍스트를 DataFrame으로 변환
            df = pd.read_csv(io.StringIO(response.text))

            # Date 컬럼이 있는지 확인
            if 'Date' not in df.columns:
                print(f"StatCounter API Error: No 'Date' column in response")
                return pd.DataFrame()

            # 날짜 포맷 변환: YYYY-MM
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m')

            # NaT (잘못된 날짜) 제거
            df = df[df['Date'] != 'NaT']

            # 2019년 이전 데이터 필터링 (혹시 모를 경우 대비)
            if metric == "os" and from_year == "2019":
                df = df[df['Date'] >= '2019-01']

            df.set_index('Date', inplace=True)
            return df
    except Exception as e:
        print(f"StatCounter Error: {e}")

    return pd.DataFrame()


def process_search_engine_data(df):
    """
    Google, Bing, Yahoo, Other 4개로 정리
    (Streamlit 버전과 동일한 로직)

    - Google: Google
    - Yahoo: Yahoo! 또는 Yahoo
    - Bing: bing 또는 Bing
    - Other: Baidu, YANDEX, DuckDuckGo 등 나머지 합산
    """
    if df.empty:
        return df

    cols = df.columns.tolist()

    # 컬럼명 대소문자 처리
    # Bing: 'bing' 또는 'Bing'
    bing_col = None
    for c in cols:
        if c.lower() == 'bing':
            bing_col = c
            break

    # Yahoo: 'Yahoo!' 또는 'Yahoo'
    yahoo_col = None
    for c in cols:
        if 'yahoo' in c.lower():
            yahoo_col = c
            break

    # Google
    google_col = None
    for c in cols:
        if c.lower() == 'google':
            google_col = c
            break

    # 유효한 타겟 컬럼 수집
    target_cols = []
    if google_col:
        target_cols.append(google_col)
    if yahoo_col:
        target_cols.append(yahoo_col)
    if bing_col:
        target_cols.append(bing_col)

    # Other 계산 (나머지 컬럼들 합산)
    other_cols = [c for c in cols if c not in target_cols]

    # 결과 DataFrame 생성
    df_processed = pd.DataFrame(index=df.index)

    if google_col:
        df_processed['Google'] = df[google_col]
    if yahoo_col:
        df_processed['Yahoo'] = df[yahoo_col]
    if bing_col:
        df_processed['Bing'] = df[bing_col]
    if other_cols:
        df_processed['Other'] = df[other_cols].sum(axis=1)

    # 순서 정렬: Google, Yahoo, Other, Bing
    desired_order = ['Google', 'Yahoo', 'Other', 'Bing']
    final_order = [c for c in desired_order if c in df_processed.columns]

    return df_processed[final_order]


def process_os_data(df):
    """
    OS 데이터에서 Android와 iOS만 추출

    Parameters:
    -----------
    df : DataFrame
        StatCounter에서 가져온 원본 OS 데이터

    Returns:
    --------
    DataFrame with Android and iOS columns only
    """
    if df.empty:
        return df

    cols = df.columns.tolist()

    # Android 컬럼 찾기
    android_col = None
    for c in cols:
        if 'android' in c.lower():
            android_col = c
            break

    # iOS 컬럼 찾기
    ios_col = None
    for c in cols:
        if 'ios' in c.lower():
            ios_col = c
            break

    # 결과 DataFrame 생성
    df_processed = pd.DataFrame(index=df.index)

    if android_col:
        df_processed['Android'] = df[android_col]
    if ios_col:
        df_processed['iOS'] = df[ios_col]

    # NaN 값을 0으로 처리하지 않고 그대로 유지 (차트에서 연결되지 않도록)
    return df_processed


# =============================================================================
# 2. Active ETF (TIMEFOLIO)
# =============================================================================
#
# [데이터 소스]
# - URL: https://timefolioetf.co.kr/m11_view.php?idx={ETF_ID}&pdfDate={YYYY-MM-DD}
# - 형식: HTML 테이블 (BeautifulSoup으로 파싱)
#
# [ETF ID 매핑]
# 해외주식형:
#   "글로벌탑픽": "22", "글로벌바이오": "9", "우주테크&방산": "20"
#   "S&P500": "5", "나스닥100": "2", "글로벌AI": "6"
# 국내주식형:
#   "K신재생에너지": "16", "K바이오": "13", "코스피": "11"
#
# [반환 데이터]
# - 종목코드: "NVDA US EQUITY", "AAPL US EQUITY" 등
# - 종목명: "NVIDIA CORP", "APPLE INC" 등
# - 수량: 보유 주식 수
# - 평가금액: 원화 기준 평가 금액
# - 비중: 포트폴리오 내 비중 (%)
#
# [리밸런싱 분석 로직]
# 1. yfinance로 각 종목의 전일→금일 가격 변동률(시장수익률) 수집
# 2. 가상 비중 = 전일비중 × (1 + 시장수익률)
# 3. 예상 비중 = 가상비중 / Σ가상비중 × 100 (정규화)
# 4. 순수 비중변화 = 실제비중 - 예상비중
#    → 이 값이 ±0.5%p 이상이면 매니저의 실제 매매로 판단
# =============================================================================

def fetch_etf_portfolio(etf_idx="22", date=None):
    """
    TIMEFOLIO 웹사이트에서 ETF 포트폴리오 데이터 크롤링

    Parameters:
    -----------
    etf_idx : str
        ETF 고유 ID (예: "22"=글로벌탑픽, "5"=S&P500)
    date : str
        조회 날짜 (YYYY-MM-DD), None이면 오늘 날짜

    Returns:
    --------
    DataFrame with columns: 종목코드, 종목명, 수량, 평가금액, 비중
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    base_url = "https://timefolioetf.co.kr/m11_view.php"

    params = {
        'idx': etf_idx,
        'cate': '',
        'pdfDate': date
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8',
        'Referer': 'https://timefolioetf.co.kr/'
    }

    try:
        response = requests.get(base_url, params=params, headers=headers,
                               timeout=30, verify=False)
        response.raise_for_status()
        response.encoding = 'utf-8'

        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # 구성종목 테이블 찾기 (class="table3")
        table = soup.find('table', class_='table3')
        if not table:
            return pd.DataFrame()

        # 테이블 데이터 추출
        rows = table.find('tbody').find_all('tr')
        data = []

        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 5:
                # 숫자 파싱 (쉼표 제거)
                quantity_text = cols[2].get_text(strip=True).replace(',', '')
                value_text = cols[3].get_text(strip=True).replace(',', '')
                weight_text = cols[4].get_text(strip=True)

                data.append({
                    '종목코드': cols[0].get_text(strip=True),
                    '종목명': cols[1].get_text(strip=True),
                    '수량': int(quantity_text) if quantity_text else 0,
                    '평가금액': int(value_text) if value_text else 0,
                    '비중': float(weight_text) if weight_text else 0.0
                })

        return pd.DataFrame(data)

    except Exception as e:
        print(f"ETF Crawling Error: {e}")
        return pd.DataFrame()


# =============================================================================
# 3. Earnings Trading (Idio Score)
# =============================================================================
#
# [데이터 소스]
# 1. 주가 데이터: Yahoo Finance (yfinance 라이브러리)
# 2. Fama-French Factors: Kenneth French Library
#    - SMB/HML: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip
#    - Momentum: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip
#
# [5-Factor Regression Model]
# R_stock = α + β_Mkt × R_Market + β_Sec × R_Sector + β_SMB × SMB + β_HML × HML + β_Mom × MOM + ε
#
# 각 Factor 설명:
# - R_Market: S&P 500 (SPY) 수익률 - 시장 전체 움직임
# - R_Sector: 섹터 ETF 수익률 (XLK=기술, XLV=헬스케어, XLF=금융 등)
# - SMB (Small Minus Big): 소형주 프리미엄 - 소형주가 대형주 대비 얼마나 outperform했는지
# - HML (High Minus Low): 가치주 프리미엄 - 가치주가 성장주 대비 얼마나 outperform했는지
# - MOM (Momentum): 모멘텀 팩터 - 최근 상승 종목이 계속 상승하는 경향
# - ε (잔차): Idiosyncratic Return - 위 요인으로 설명되지 않는 순수 종목 고유 수익률
#
# [Idio Score 계산 (Goldman Sachs Method)]
# 1. 실적발표일 감지: T-2 ~ T+2 (5일 윈도우)
# 2. Efficiency Score 계산:
#    Score = Mean(|ε|) × 252 / (Std(ε) × √252)
#    - 연간화된 평균 절대 잔차 / 연간화된 잔차 변동성
#    - Sharpe Ratio와 유사한 개념 (변동성 대비 수익 효율)
#
# 3. Delta Score (최종 점수):
#    Score_incl = 전체 기간 포함 효율성
#    Score_excl = 실적 발표일 제외 효율성
#    Delta Score = Score_incl - Score_excl
#
# 4. 해석:
#    - Delta > 0: 실적 발표가 알파 창출에 기여 (실적 이벤트 보유 유리)
#    - Delta < 0: 실적 발표가 변동성만 증가 (이벤트 회피 유리)
#
# [VIX 레벨별 조정]
# Goldman Sachs 리서치에 따르면 VIX 수준이 실적 이벤트 성과에 영향:
# - VIX 35~45 (최적 구간): ×1.2 보정 - 불확실성 해소 효과 극대화
# - VIX > 45 (위험 구간): ×0.8 보정 - 극단적 공포로 실적 무관한 하락
# - VIX < 35 (안정 구간): ×1.0 보정 - 기본 점수 유지
# =============================================================================

# 섹터별 벤치마크 ETF 매핑
SECTOR_BENCHMARKS = {
    'Technology': 'XLK',      # Technology Select Sector SPDR
    'Healthcare': 'XLV',      # Health Care Select Sector SPDR
    'Financials': 'XLF',      # Financial Select Sector SPDR
    'Consumer': 'XLY',        # Consumer Discretionary Select Sector SPDR
    'Energy': 'XLE',          # Energy Select Sector SPDR
    'Industrials': 'XLI',     # Industrial Select Sector SPDR
}

def fetch_vix_level():
    """
    현재 VIX 지수 조회 (Yahoo Finance)

    VIX (Volatility Index):
    - CBOE에서 산출하는 S&P 500 옵션의 내재 변동성 지수
    - "공포 지수"라고도 불림
    - 20 이하: 안정적, 20-30: 보통, 30 이상: 높은 불확실성

    Returns:
    --------
    float: VIX 값 (실패시 시뮬레이션 값)
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass

    # Fallback: 시뮬레이션 (15~25 사이)
    return np.random.uniform(15, 25)

def calculate_idio_score_simple(ticker, sector="Technology"):
    """
    간단한 Idio Score 계산 (시뮬레이션)

    실제 구현시 필요한 단계:
    1. yfinance로 3년치 일별 수익률 수집
    2. Fama-French Factor 데이터 다운로드 및 병합
    3. 5-Factor Regression 수행 (sklearn LinearRegression)
    4. 잔차(ε) 추출 및 Score 계산

    Parameters:
    -----------
    ticker : str
        종목 티커 (예: "AAPL", "NVDA")
    sector : str
        섹터 분류 (벤치마크 ETF 결정용)

    Returns:
    --------
    dict with score details
    """
    # 시뮬레이션 데이터 (실제로는 regression 결과 사용)
    np.random.seed(hash(ticker) % 2**32)

    score = round(np.random.uniform(-0.3, 1.2), 2)
    market_beta = round(np.random.uniform(0.8, 1.3), 2)
    sector_beta = round(np.random.uniform(0.1, 0.5), 2)
    smb_beta = round(np.random.uniform(-0.2, 0.3), 2)
    hml_beta = round(np.random.uniform(-0.3, 0.2), 2)
    mom_beta = round(np.random.uniform(-0.1, 0.2), 2)
    avg_return = round(np.random.uniform(-0.1, 0.3), 2)
    daily_vol = round(np.random.uniform(20, 40), 1)

    return {
        'ticker': ticker,
        'idio_score': score,
        'market_beta': market_beta,
        'sector_beta': sector_beta,
        'smb_beta': smb_beta,
        'hml_beta': hml_beta,
        'mom_beta': mom_beta,
        'avg_return': avg_return,
        'daily_volatility': daily_vol,
        'insight': get_score_insight(score, ticker)
    }

def get_score_insight(score, ticker):
    """Idio Score 기반 인사이트 메시지 생성"""
    if score > 0.5:
        return f"High Impact: {ticker}의 실적 발표가 변동성 대비 수익 효율을 크게 높여줍니다. (Delta: +{score})"
    elif score > 0:
        return f"Moderate Impact: {ticker}의 실적 발표가 긍정적 영향을 미칩니다. (Delta: +{score})"
    else:
        return f"Low Impact: {ticker}의 실적 발표를 제외해도 효율성 차이가 거의 없습니다. (Delta: {score})"


# VIX 관련 함수
import yfinance as yf

def fetch_vix_history(years=10):
    """
    VIX 지수 히스토리 데이터 가져오기

    Args:
        years: 조회할 연도 (default: 10년)

    Returns:
        dict: {dates: [...], values: [...]}
    """
    try:
        vix = yf.Ticker("^VIX")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        hist = vix.history(start=start_date, end=end_date)

        if not hist.empty:
            # 날짜와 Close 가격 추출
            dates = [d.strftime('%Y-%m-%d') for d in hist.index]
            values = hist['Close'].tolist()

            return {
                'dates': dates,
                'values': values,
                'current': round(values[-1], 2) if values else None
            }
    except Exception as e:
        print(f"VIX fetch error: {e}")

    return {'dates': [], 'values': [], 'current': None}

def get_vix_current():
    """현재 VIX 값 가져오기"""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 2)
    except:
        pass
    return None


# =============================================================================
# 라우트 (페이지 렌더링)
# =============================================================================

@app.route('/')
def index():
    """홈페이지 - 주요 기능 메뉴"""
    return render_template('index.html')

@app.route('/market')
def market():
    """Market Data - Search Engine으로 리다이렉트"""
    from flask import redirect, url_for
    return redirect(url_for('search_engine'))

@app.route('/market/search-engine')
def search_engine():
    """Market Data - Search Engine M/S 페이지"""
    return render_template('search_engine.html')

@app.route('/market/os-market-share')
def os_market_share():
    """Market Data - OS Market Share 페이지"""
    return render_template('os_market_share.html')

@app.route('/earnings')
def earnings():
    """Earnings Trading 페이지"""
    return render_template('earnings.html')

@app.route('/etf')
def etf():
    """Active ETF 페이지"""
    return render_template('etf.html')


# =============================================================================
# API 엔드포인트
# =============================================================================

@app.route('/api/statcounter/<metric>')
def api_statcounter(metric):
    """
    StatCounter 시장점유율 데이터 API

    Endpoint: GET /api/statcounter/{metric}?device={device}

    Parameters:
    - metric: "browser" | "os" | "search_engine"
    - device: "desktop" | "mobile" | "desktop+mobile" (default: "desktop")

    Response:
    {
        "dates": ["2024-01", "2024-02", ...],
        "series": {
            "Chrome": [65.1, 65.3, ...],
            "Safari": [18.4, 18.2, ...],
            ...
        }
    }
    """
    device = request.args.get('device', 'desktop')

    df = fetch_statcounter_data(metric=metric, device=device)

    if df.empty:
        return jsonify({'error': 'No data available'}), 404

    # 최근 24개월만 반환
    df = df.tail(24)

    data = {
        'dates': df.index.tolist(),
        'series': {}
    }

    for col in df.columns:
        data['series'][col] = df[col].tolist()

    return jsonify(data)

@app.route('/api/os-rivalry')
def api_os_rivalry():
    """
    OS 시장점유율 API (Android vs iOS 비교용)

    Endpoint: GET /api/os-rivalry?device={device}

    Parameters:
    - device: "mobile" | "tablet" (default: "mobile")
    - Note: mobile+tablet is NOT supported via this endpoint due to URL encoding issues

    Response: Android, iOS만 필터링하여 반환
    """
    device = request.args.get('device', 'mobile')

    # mobile+tablet 요청은 지원하지 않음 (프론트엔드에서 합산 처리)
    if device == 'mobile+tablet':
        return jsonify({'error': 'mobile+tablet not supported. Use mobile and tablet separately.'}), 400

    # 2019년부터 데이터 가져오기
    df = fetch_statcounter_data(metric="os", device=device, from_year="2019")

    if df.empty:
        return jsonify({'error': 'No data available'}), 404

    # process_os_data로 Android, iOS만 추출
    df_processed = process_os_data(df)

    if df_processed.empty:
        return jsonify({'error': 'No Android/iOS data'}), 404

    data = {
        'dates': df_processed.index.tolist(),
        'series': {}
    }

    for col in df_processed.columns:
        # NaN이 아닌 값만 유지 (null로 변환)
        values = df_processed[col].tolist()
        data['series'][col] = values

    return jsonify(data)

@app.route('/api/os-rivalry-combined')
def api_os_rivalry_combined():
    """
    OS 시장점유율 API - Mobile+Tablet 전용

    URL 인코딩 문제를 피하기 위해 별도 엔드포인트로 분리
    백엔드에서 params dict로 device='mobile+tablet' 전달

    Response: Android, iOS만 필터링하여 반환
    """
    # 2019년부터 데이터 가져오기 (device 파라미터를 params dict로 전달)
    df = fetch_statcounter_data(metric="os", device="mobile+tablet", from_year="2019")

    if df.empty:
        return jsonify({'error': 'No data available'}), 404

    # process_os_data로 Android, iOS만 추출
    df_processed = process_os_data(df)

    if df_processed.empty:
        return jsonify({'error': 'No Android/iOS data'}), 404

    data = {
        'dates': df_processed.index.tolist(),
        'series': {}
    }

    for col in df_processed.columns:
        # NaN이 아닌 값만 유지 (null로 변환)
        values = df_processed[col].tolist()
        data['series'][col] = values

    return jsonify(data)

# Old ETF endpoints removed - replaced with new Active ETF monitoring system below

@app.route('/api/vix')
def api_vix():
    """
    VIX 지수 API

    Endpoint: GET /api/vix

    Response:
    {
        "value": 18.5,
        "status": "Stable",
        "color": "success"
    }
    """
    vix = fetch_vix_level()

    if vix < 20:
        status, color = "Stable", "success"
    elif vix < 35:
        status, color = "Elevated", "warning"
    elif vix < 45:
        status, color = "Optimal Zone", "primary"
    else:
        status, color = "Danger", "danger"

    return jsonify({
        'value': round(vix, 2),
        'status': status,
        'color': color
    })

@app.route('/api/idio-score')
def api_idio_score():
    """
    Idio Score 계산 API

    Endpoint: GET /api/idio-score?ticker={ticker}&sector={sector}

    Parameters:
    - ticker: 종목 티커 (예: "AAPL", "NVDA")
    - sector: 섹터 (예: "Technology", "Healthcare")

    Response:
    {
        "ticker": "NVDA",
        "idio_score": 0.85,
        "market_beta": 1.15,
        "sector_beta": 0.32,
        "smb_beta": 0.05,
        "hml_beta": -0.12,
        "mom_beta": 0.08,
        "avg_return": 0.15,
        "daily_volatility": 28.5,
        "insight": "High Impact: NVDA의 실적 발표가..."
    }
    """
    ticker = request.args.get('ticker', 'AAPL').upper()
    sector = request.args.get('sector', 'Technology')

    result = calculate_idio_score_simple(ticker, sector)
    return jsonify(result)


# =============================================================================
# Active ETF API 엔드포인트
# =============================================================================

# ETF 설정
ETF_CONFIG = [
    {
        'id': 'time_sp500',
        'name': 'TIME S&P500',
        'type': 'time',
        'idx': '5',
        'description': 'TIME S&P500 ETF'
    },
    {
        'id': 'time_nasdaq100',
        'name': 'TIME NASDAQ100',
        'type': 'time',
        'idx': '2',
        'description': 'TIME NASDAQ100 ETF'
    },
    {
        'id': 'kiwoom_growth30',
        'name': 'KIWOOM 미국성장기업30액티브',
        'type': 'kiwoom',
        'code': '459790',
        'description': 'KIWOOM 미국성장기업30액티브'
    }
]

# ETF 모니터 인스턴스 생성
etf_monitors = {
    'time_sp500': TimeETFMonitor('5', 'TIME S&P500'),
    'time_nasdaq100': TimeETFMonitor('2', 'TIME NASDAQ100'),
    'kiwoom_growth30': KiwoomETFMonitor()
}

@app.route('/api/etf/list')
def api_etf_list():
    """
    ETF 목록 반환
    """
    return jsonify({'etfs': ETF_CONFIG})

@app.route('/api/etf/data/<etf_id>')
def api_etf_data(etf_id):
    """
    특정 ETF의 포트폴리오 데이터 및 리밸런싱 분석
    Query Parameters:
    - date: YYYY-MM-DD (default: today)

    Response:
    {
        "etf_info": {...},
        "base_date": "2026-01-30",
        "prev_date": "2026-01-29",
        "portfolio_today": [...],
        "rebalancing": {
            "new_stocks": [...],
            "removed_stocks": [...],
            "increased_stocks": [...],
            "decreased_stocks": [...]
        }
    }
    """
    if etf_id not in etf_monitors:
        return jsonify({'error': 'Invalid ETF ID'}), 404

    # 날짜 파라미터 처리
    date_str = request.args.get('date')
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')

    try:
        monitor = etf_monitors[etf_id]

        # 오늘 데이터 가져오기
        df_today = monitor.load_data(date_str)
        if df_today is None or df_today.empty:
            return jsonify({'error': f'No data available for {date_str}'}), 404

        # 이전 영업일 찾기
        prev_date = monitor.get_previous_business_day(date_str)

        # 리밸런싱 분석 (캐시 우선, 없으면 실시간 계산)
        rebalancing = None
        if prev_date:
            # 1. 캐시된 리밸런싱 결과 확인
            rebalancing_cache_dir = os.path.join(CACHE_DIR, 'rebalancing')
            cache_filename = f"{etf_id}_{date_str}_vs_{prev_date}.json"
            cache_filepath = os.path.join(rebalancing_cache_dir, cache_filename)

            if os.path.exists(cache_filepath):
                # 캐시 사용 (빠름)
                try:
                    with open(cache_filepath, 'r', encoding='utf-8') as f:
                        rebalancing = json.load(f)
                    print(f"✓ Rebalancing cache hit: {cache_filename}")
                except Exception as e:
                    print(f"✗ Rebalancing cache load error: {e}")
                    rebalancing = None

            # 2. 캐시 없으면 실시간 계산 (느림)
            if rebalancing is None:
                df_prev = monitor.load_data(prev_date)
                if df_prev is not None and not df_prev.empty:
                    print(f"⚠ Rebalancing cache miss, computing real-time (slow)...")
                    # Pass dates for Yahoo Finance price-based rebalancing analysis
                    rebalancing = monitor.analyze_rebalancing(df_today, df_prev, date_str, prev_date)

        # ETF 정보
        etf_info = next((etf for etf in ETF_CONFIG if etf['id'] == etf_id), None)

        # Top 10 Holdings
        top_holdings = df_today.nlargest(10, '비중')[['종목코드', '종목명', '비중', '평가금액']].to_dict('records')

        response = {
            'etf_info': etf_info,
            'base_date': date_str,
            'prev_date': prev_date,
            'top_holdings': top_holdings,
            'portfolio_today': df_today.to_dict('records'),
            'rebalancing': rebalancing,
            'summary': {
                'total_positions': len(df_today),
                'new_count': len(rebalancing['new_stocks']) if rebalancing else 0,
                'removed_count': len(rebalancing['removed_stocks']) if rebalancing else 0,
                'increased_count': len(rebalancing['increased_stocks']) if rebalancing else 0,
                'decreased_count': len(rebalancing['decreased_stocks']) if rebalancing else 0
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Search Engine M/S API 엔드포인트
# =============================================================================

@app.route('/api/search-engine/all')
def api_search_engine_all():
    """
    Search Engine 시장점유율 전체 데이터 API
    캐시 파일에서 로드 (Scheduled Task로 매일 갱신)

    Response:
    {
        "desktop_mobile": { "dates": [...], "Google": [...], "Yahoo": [...], "Bing": [...], "Other": [...] },
        "desktop": { ... },
        "mobile": { ... }
    }
    """
    # 캐시 파일에서 로드 시도
    cached_data = load_cache_file('search_engine.json')
    if cached_data:
        return jsonify(cached_data)

    # 캐시 파일이 없으면 API 직접 호출 (fallback)
    result = {}
    START_DATE = "2019-01"  # 2019년 1월부터 데이터만 사용

    for device_key, device_val in [('desktop_mobile', 'desktop-mobile'), ('desktop', 'desktop'), ('mobile', 'mobile')]:
        df = fetch_statcounter_data(metric="search_engine", device=device_val, from_year="2019")
        if not df.empty:
            df_processed = process_search_engine_data(df)
            df_processed = df_processed.sort_index()

            # 2019-01 이후 데이터만 필터링 (API가 from_year를 무시할 수 있음)
            df_processed = df_processed[df_processed.index >= START_DATE]

            data = {'dates': df_processed.index.tolist()}
            for col in df_processed.columns:
                values = df_processed[col].tolist()
                data[col] = [round(v, 2) if pd.notna(v) else None for v in values]
            result[device_key] = data
        else:
            result[device_key] = {'dates': [], 'error': 'No data'}

    return jsonify(result)


@app.route('/api/search-engine/download')
def api_search_engine_download():
    """
    Search Engine 데이터 Excel 다운로드 API
    Google, Yahoo, Other, Bing 정리된 표 3개 포함
    """
    output = io.BytesIO()

    # 캐시 파일에서 로드 시도
    cached_data = load_cache_file('search_engine.json')

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for device_key, sheet_name in [
            ('desktop_mobile', 'Desktop+Mobile'),
            ('desktop', 'Desktop'),
            ('mobile', 'Mobile')
        ]:
            if cached_data and device_key in cached_data:
                # 캐시 데이터 사용
                data = cached_data[device_key]
                df_export = pd.DataFrame({
                    'Date': data['dates'],
                    'Google': data.get('Google', []),
                    'Yahoo': data.get('Yahoo', []),
                    'Other': data.get('Other', []),
                    'Bing': data.get('Bing', [])
                })
            else:
                # Fallback: API 직접 호출
                device_val = 'desktop-mobile' if device_key == 'desktop_mobile' else device_key
                df = fetch_statcounter_data(metric="search_engine", device=device_val, from_year="2019")
                if not df.empty:
                    df_processed = process_search_engine_data(df)
                    df_processed = df_processed.sort_index()
                    df_export = df_processed.reset_index()
                else:
                    df_export = pd.DataFrame()

            if not df_export.empty:
                df_export.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)

    return Response(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': 'attachment; filename=search_engine_ms.xlsx'}
    )


@app.route('/api/os-rivalry/download')
def api_os_rivalry_download():
    """
    OS Market Share 데이터 Excel 다운로드 API
    3개 시트 (Mobile, Tablet, Mobile+Tablet) 포함
    """
    start_date = request.args.get('start_date', '2020-01')
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m'))

    output = io.BytesIO()

    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 3개 디바이스 각각 시트로 추가
            for device_key, sheet_name in [
                ('mobile', 'Mobile'),
                ('tablet', 'Tablet'),
                ('mobile+tablet', 'Mobile+Tablet')
            ]:
                # 2019년부터 데이터 가져오기
                df = fetch_statcounter_data(metric="os", device=device_key, from_year="2019")

                if not df.empty:
                    df_processed = process_os_data(df)
                    df_processed = df_processed.sort_index()

                    # 날짜 필터링
                    if start_date and end_date:
                        df_processed = df_processed[
                            (df_processed.index >= start_date) &
                            (df_processed.index <= end_date)
                        ]

                    df_export = df_processed.reset_index()
                    df_export.columns = ['Date'] + list(df_export.columns[1:])

                    df_export.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # 빈 데이터
                    pd.DataFrame({'Error': ['No data available']}).to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)

        return Response(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': 'attachment; filename=os_market_share.xlsx'}
        )
    except Exception as e:
        print(f"Excel download error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Earnings Trading API
# =============================================================================

@app.route('/api/vix/history')
def api_vix_history():
    """
    VIX 히스토리 데이터 API
    Query Parameters:
    - years: 조회 연도 (default: 10)
    """
    years = int(request.args.get('years', 10))
    data = fetch_vix_history(years)
    return jsonify(data)

@app.route('/api/vix/current')
def api_vix_current():
    """현재 VIX 값 API"""
    vix_val = get_vix_current()
    return jsonify({'vix': vix_val})


@app.route('/api/earnings/calendar')
def api_earnings_calendar():
    """
    Earnings Calendar API (Yahoo Finance 기반)

    Parameters:
    - date: YYYY-MM-DD (default: today)
    - days: Number of days (default: 7)

    Response:
    [
        {
            "Symbol": "AAPL",
            "Company": "Apple Inc.",
            "Date": "2026-01-29",
            "CallTime": "장후",
            "EPSEstimate": "2.67",
            "ReportedEPS": "-",
            "Surprise": "-",
            "MarketCap": "3.81T",
            "AnalystRating": "buy",
            "AnalystCount": 45,
            "AvgMove": 3.5,
            "Volatility": 2.8,
            "TradeAlert": "⭐ High Move + Analyst Support"
        },
        ...
    ]
    """
    import pytz
    from datetime import datetime

    date_str = request.args.get('date', None)
    days = int(request.args.get('days', 7))

    # 1. Check cache first
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    today_kst = now_kst.date().strftime('%Y-%m-%d')

    # Create cache key based on today's date (refresh daily)
    earnings_cache_dir = os.path.join(CACHE_DIR, 'earnings')
    os.makedirs(earnings_cache_dir, exist_ok=True)
    cache_filename = f"earnings_calendar_{today_kst}_days{days}.json"
    cache_filepath = os.path.join(earnings_cache_dir, cache_filename)

    # Try to load from cache
    if os.path.exists(cache_filepath):
        try:
            cache_mtime = os.path.getmtime(cache_filepath)
            cache_age_hours = (datetime.now().timestamp() - cache_mtime) / 3600

            # Use cache if less than 1 hour old
            if cache_age_hours < 1:
                with open(cache_filepath, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"✓ Earnings cache hit: {cache_filename} (age: {cache_age_hours:.1f}h)")
                return jsonify(cached_data)
            else:
                print(f"⚠ Earnings cache expired: {cache_filename} (age: {cache_age_hours:.1f}h)")
        except Exception as e:
            print(f"⚠ Earnings cache read error: {e}")

    try:
        # 2. Fetch calendar from Yahoo Finance
        df_calendar = fetch_yahoo_earnings_calendar(date_str=date_str, days=days)

        if df_calendar.empty:
            return jsonify([])

        # 2. Filter to show only current date onwards (Korean timezone)
        kst = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(kst)
        today_kst = now_kst.date().strftime('%Y-%m-%d')

        df_calendar = df_calendar[df_calendar['Date'] >= today_kst]

        if df_calendar.empty:
            return jsonify([])

        # 3. Enrich with analyst data and earnings metrics
        enriched_data = []

        print(f"[Earnings API] Processing {len(df_calendar)} events...")

        for idx, row in df_calendar.iterrows():
            ticker = row['Symbol']
            print(f"[{ticker}] Processing...")

            # Analyst data (with error handling)
            analyst_data = {'recommendKey': None, 'analystCount': 0}
            try:
                analyst_data = fetch_analyst_consensus(ticker)
                print(f"[{ticker}] Analyst: {analyst_data.get('recommendKey')}, Count: {analyst_data.get('analystCount')}")
            except Exception as e:
                print(f"[{ticker}] Analyst fetch error: {e}")
                pass

            # Earnings metrics (3-year avg move and volatility, with error handling)
            earnings_metrics = {'avg_move': None, 'volatility': None, 'num_earnings': 0}
            try:
                earnings_metrics = calculate_earnings_metrics(ticker, years=3)
                print(f"[{ticker}] Metrics: AvgMove={earnings_metrics.get('avg_move')}, Vol={earnings_metrics.get('volatility')}")
            except Exception as e:
                print(f"[{ticker}] Metrics fetch error: {e}")
                pass

            # Trade Alert
            trade_alert = calculate_trade_alert(
                avg_move=earnings_metrics.get('avg_move'),
                analyst_rating=analyst_data.get('recommendKey'),
                analyst_count=analyst_data.get('analystCount', 0)
            )

            enriched_data.append({
                'Symbol': ticker,
                'Company': row['Company'],
                'Date': row['Date'],
                'CallTime': row['CallTime'],
                'EPSEstimate': row['EPSEstimate'],
                'ReportedEPS': row['ReportedEPS'],
                'Surprise': row['Surprise'],
                'MarketCap': row['MarketCap'],
                'AnalystRating': analyst_data.get('recommendKey'),
                'AnalystCount': analyst_data.get('analystCount', 0),
                'AvgMove': earnings_metrics.get('avg_move'),
                'Volatility': earnings_metrics.get('volatility'),
                'NumEarnings': earnings_metrics.get('num_earnings', 0),
                'TradeAlert': trade_alert
            })

        print(f"[Earnings API] Completed. Returning {len(enriched_data)} events.")

        # 3. Save to cache
        try:
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Earnings cache saved: {cache_filename}")
        except Exception as e:
            print(f"⚠ Earnings cache save error: {e}")

        return jsonify(enriched_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
