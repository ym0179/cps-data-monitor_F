#!/usr/bin/env python3
"""
=============================================================================
캐시 갱신 스크립트 (Scheduled Task용)
=============================================================================
PythonAnywhere Scheduled Task에서 매일 실행하여 캐시 파일 갱신

사용법:
  python /home/ym96/my_finance_app/update_cache.py

설정 (PythonAnywhere > Tasks):
  Command: python3 /home/ym96/my_finance_app/update_cache.py
  Time: 09:00 UTC (한국시간 18:00) 또는 00:00 UTC (한국시간 09:00)
=============================================================================
"""

import os
import sys
import json
import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import urllib3
import pytz

# 현재 스크립트의 디렉토리를 sys.path에 추가 (etf_monitor import를 위해)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 캐시 디렉토리 설정
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')


def fetch_statcounter_data(metric="search_engine", device="desktop", from_year="2019", from_month="01"):
    """StatCounter에서 시장점유율 데이터 수집"""
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

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, verify=False, timeout=60)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
            df.set_index('Date', inplace=True)
            return df
    except Exception as e:
        print(f"StatCounter Error: {e}")

    return pd.DataFrame()


def process_search_engine_data(df):
    """Google, Bing, Yahoo, Other 4개로 정리"""
    if df.empty:
        return df

    cols = df.columns.tolist()

    # 컬럼명 대소문자 처리
    bing_col = None
    for c in cols:
        if c.lower() == 'bing':
            bing_col = c
            break

    yahoo_col = None
    for c in cols:
        if 'yahoo' in c.lower():
            yahoo_col = c
            break

    google_col = None
    for c in cols:
        if c.lower() == 'google':
            google_col = c
            break

    target_cols = []
    if google_col:
        target_cols.append(google_col)
    if yahoo_col:
        target_cols.append(yahoo_col)
    if bing_col:
        target_cols.append(bing_col)

    other_cols = [c for c in cols if c not in target_cols]

    df_processed = pd.DataFrame(index=df.index)

    if google_col:
        df_processed['Google'] = df[google_col]
    if yahoo_col:
        df_processed['Yahoo'] = df[yahoo_col]
    if bing_col:
        df_processed['Bing'] = df[bing_col]
    if other_cols:
        df_processed['Other'] = df[other_cols].sum(axis=1)

    desired_order = ['Google', 'Yahoo', 'Other', 'Bing']
    final_order = [c for c in desired_order if c in df_processed.columns]

    return df_processed[final_order]


def update_search_engine_cache():
    """Search Engine 데이터 캐시 갱신"""
    print(f"[{datetime.now()}] Search Engine 캐시 갱신 시작...")

    result = {}
    START_DATE = "2019-01"  # 2019년 1월부터 데이터만 사용

    for device_key, device_val in [('desktop_mobile', 'desktop-mobile'), ('desktop', 'desktop'), ('mobile', 'mobile')]:
        print(f"  - {device_key} 데이터 수집 중...")
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
            print(f"    완료: {len(df_processed)}개월 데이터 (2019-01 이후)")
        else:
            result[device_key] = {'dates': [], 'error': 'No data'}
            print(f"    실패: 데이터 없음")

    # 캐시 디렉토리 생성
    os.makedirs(CACHE_DIR, exist_ok=True)

    # JSON 파일로 저장
    cache_file = os.path.join(CACHE_DIR, 'search_engine.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"[{datetime.now()}] 캐시 저장 완료: {cache_file}")
    return result


def update_os_cache():
    """OS Market Share 데이터 캐시 갱신"""
    print(f"[{datetime.now()}] OS Market Share 캐시 갱신 시작...")

    result = {}
    START_DATE = "2019-01"

    for device_key, device_val in [('desktop', 'desktop'), ('mobile', 'mobile'), ('tablet', 'tablet')]:
        print(f"  - {device_key} 데이터 수집 중...")
        df = fetch_statcounter_data(metric="os", device=device_val, from_year="2019")

        if not df.empty:
            df = df.sort_index()
            df = df[df.index >= START_DATE]

            data = {'dates': df.index.tolist()}
            for col in df.columns:
                values = df[col].tolist()
                data[col] = [round(v, 2) if pd.notna(v) else None for v in values]
            result[device_key] = data
            print(f"    완료: {len(df)}개월 데이터 (2019-01 이후)")
        else:
            result[device_key] = {'dates': [], 'error': 'No data'}
            print(f"    실패: 데이터 없음")

    # 캐시 디렉토리 생성
    os.makedirs(CACHE_DIR, exist_ok=True)

    # JSON 파일로 저장
    cache_file = os.path.join(CACHE_DIR, 'os_market_share.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"[{datetime.now()}] 캐시 저장 완료: {cache_file}")
    return result


def update_active_etf_cache():
    """Active ETF 최근 7일 데이터 캐시 갱신 (포트폴리오 + 리밸런싱 결과)"""
    print(f"[{datetime.now()}] Active ETF 캐시 갱신 시작...")

    try:
        from etf_monitor import TimeETFMonitor, KiwoomETFMonitor
    except ImportError as e:
        print(f"  ERROR: etf_monitor import 실패 - {e}")
        return

    # ETF 설정
    etf_configs = [
        {'id': 'time_sp500', 'name': 'TIME S&P500', 'type': 'time', 'idx': '5'},
        {'id': 'time_nasdaq100', 'name': 'TIME NASDAQ100', 'type': 'time', 'idx': '2'},
        {'id': 'kiwoom_growth30', 'name': 'KIWOOM 미국성장기업30액티브', 'type': 'kiwoom', 'code': '459790'}
    ]

    # 데이터 디렉토리 설정 (PythonAnywhere 환경)
    data_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    rebalancing_cache_dir = os.path.join(CACHE_DIR, 'rebalancing')
    os.makedirs(rebalancing_cache_dir, exist_ok=True)

    # 최근 7일 날짜 생성 (한국시간 기준)
    # TIME ETF 홈페이지는 한국시간 오전 7시경 업데이트됨
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)

    # 오전 7시 이전이면 전날까지만 조회 (아직 업데이트 전)
    if now_kst.hour < 7:
        today_kst = (now_kst - timedelta(days=1)).date()
        print(f"  현재 KST 시간: {now_kst.strftime('%Y-%m-%d %H:%M:%S %Z')} (오전 7시 이전 - 전날까지 조회)")
    else:
        today_kst = now_kst.date()
        print(f"  현재 KST 시간: {now_kst.strftime('%Y-%m-%d %H:%M:%S %Z')} (오전 7시 이후 - 오늘까지 조회)")

    dates_to_fetch = [(today_kst - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    print(f"  조회 날짜: {dates_to_fetch}")

    for etf_config in etf_configs:
        etf_id = etf_config['id']
        etf_name = etf_config['name']
        etf_type = etf_config['type']

        print(f"\n  [{etf_name}] 데이터 수집 중...")

        # Monitor 인스턴스 생성
        if etf_type == 'time':
            data_dir = os.path.join(data_base_dir, 'time_etf')
            monitor = TimeETFMonitor(
                etf_idx=etf_config['idx'],
                etf_name=etf_name,
                data_dir=data_dir
            )
        elif etf_type == 'kiwoom':
            data_dir = os.path.join(data_base_dir, 'kiwoom_etf')
            monitor = KiwoomETFMonitor(data_dir=data_dir)
        else:
            print(f"    ERROR: Unknown ETF type: {etf_type}")
            continue

        # 최근 7일 포트폴리오 데이터 수집
        success_count = 0
        valid_dates = []
        for date_str in dates_to_fetch:
            try:
                df = monitor.get_portfolio_data(date_str)
                if not df.empty:
                    success_count += 1
                    valid_dates.append(date_str)
                    print(f"    ✓ {date_str}: {len(df)}개 종목")
                else:
                    print(f"    ✗ {date_str}: 데이터 없음")
            except Exception as e:
                print(f"    ✗ {date_str}: 오류 - {e}")

        print(f"    포트폴리오 수집 완료: {success_count}/{len(dates_to_fetch)}일")

        # 리밸런싱 결과 계산 및 캐시 (연속된 날짜 쌍)
        if len(valid_dates) >= 2:
            print(f"    리밸런싱 결과 캐시 생성 중...")
            rebalancing_count = 0

            for i in range(len(valid_dates) - 1):
                current_date = valid_dates[i]
                previous_date = valid_dates[i + 1]

                try:
                    # 리밸런싱 분석 실행 - DataFrame을 먼저 로드
                    df_today = monitor.load_data(current_date)
                    df_prev = monitor.load_data(previous_date)

                    if df_today is None or df_prev is None or df_today.empty or df_prev.empty:
                        print(f"      ✗ {current_date} vs {previous_date}: 데이터 로드 실패")
                        continue

                    rebalancing_result = monitor.analyze_rebalancing(df_today, df_prev, current_date, previous_date)

                    if rebalancing_result:
                        # 캐시 파일명: {etf_id}_{current_date}_vs_{previous_date}.json
                        cache_filename = f"{etf_id}_{current_date}_vs_{previous_date}.json"
                        cache_filepath = os.path.join(rebalancing_cache_dir, cache_filename)

                        with open(cache_filepath, 'w', encoding='utf-8') as f:
                            json.dump(rebalancing_result, f, ensure_ascii=False, indent=2)

                        rebalancing_count += 1
                        print(f"      ✓ {current_date} vs {previous_date}")
                    else:
                        print(f"      ✗ {current_date} vs {previous_date}: 결과 없음")

                except Exception as e:
                    print(f"      ✗ {current_date} vs {previous_date}: 오류 - {e}")

            print(f"    리밸런싱 캐시 완료: {rebalancing_count}개")
        else:
            print(f"    리밸런싱 캐시 건너뜀 (유효 날짜 부족)")

    print(f"\n[{datetime.now()}] Active ETF 캐시 갱신 완료")
    print(f"  포트폴리오 저장 위치: {data_base_dir}")
    print(f"  리밸런싱 캐시 위치: {rebalancing_cache_dir}")
    return True


def main():
    """메인 함수 - 모든 캐시 갱신"""
    print("=" * 60)
    print(f"캐시 갱신 스크립트 시작: {datetime.now()}")
    print("=" * 60)

    try:
        # 1. Search Engine 캐시 갱신
        update_search_engine_cache()
        print()

        # 2. OS Market Share 캐시 갱신
        update_os_cache()
        print()

        # 3. Active ETF 캐시 갱신 (최근 7일)
        update_active_etf_cache()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"캐시 갱신 완료: {datetime.now()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
