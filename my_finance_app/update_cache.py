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
import json
import pandas as pd
import requests
import io
from datetime import datetime
import urllib3

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


def main():
    """메인 함수 - 모든 캐시 갱신"""
    print("=" * 60)
    print(f"캐시 갱신 스크립트 시작: {datetime.now()}")
    print("=" * 60)

    # Search Engine 캐시 갱신
    update_search_engine_cache()

    # 추후 다른 데이터 캐시도 여기에 추가 가능
    # update_browser_cache()
    # update_os_cache()

    print("=" * 60)
    print(f"캐시 갱신 완료: {datetime.now()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
