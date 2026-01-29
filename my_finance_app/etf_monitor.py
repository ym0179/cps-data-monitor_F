import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz
import urllib3
import yfinance as yf
import numpy as np

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TimeETFMonitor:
    """
    Monitor for TIME ETF products
    - S&P500: idx=5
    - NASDAQ100: idx=2
    Source: https://timeetf.co.kr/m11_view.php
    """

    BASE_URL = "https://timeetf.co.kr/m11_view.php"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    KST = pytz.timezone('Asia/Seoul')

    # ISIN 코드 → yfinance 티커 매핑 테이블
    ISIN_TO_TICKER = {
        'CA13321L1085': 'CCJ',      # Cameco Corp
        'US02079K3059': 'GOOGL',    # Alphabet Inc Class A
        'US02079K1079': 'GOOG',     # Alphabet Inc Class C
        # 필요시 추가 매핑
    }

    def __init__(self, etf_idx: str, etf_name: str, data_dir: str = "./data/time_etf"):
        self.etf_idx = etf_idx
        self.etf_name = etf_name
        self.data_dir = os.path.join(data_dir, f"etf_{etf_idx}")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_data_from_web(self, date_str: str) -> pd.DataFrame:
        """
        Fetch portfolio data for a specific date (YYYY-MM-DD) via web scraping.
        """
        # TIME ETF expects date in YYYY-MM-DD format (with hyphens)
        params = {
            "idx": self.etf_idx,
            "cate": "",
            "pdfDate": date_str  # Keep hyphens!
        }

        try:
            full_url = f"{self.BASE_URL}?idx={self.etf_idx}&cate=&pdfDate={date_str}"
            print(f"[TIME ETF] Fetching: {full_url}")

            resp = requests.get(self.BASE_URL, params=params, headers=self.HEADERS, timeout=10)
            print(f"[TIME ETF] Response status: {resp.status_code}")

            if resp.status_code != 200:
                print(f"[TIME ETF] Status {resp.status_code} for {date_str}")
                return pd.DataFrame()

            soup = BeautifulSoup(resp.content, 'html.parser')

            # Find portfolio table (class="table3 moreList1")
            # Try multiple ways to find the table
            table = soup.find('table', {'class': 'table3'})
            if not table:
                table = soup.find('table', {'class': 'moreList1'})
            if not table:
                table = soup.find('table', class_=lambda x: x and ('table3' in x or 'moreList1' in x))
            if not table:
                # Try finding any table
                tables = soup.find_all('table')
                print(f"[TIME ETF] Found {len(tables)} tables total")
                if tables:
                    table = tables[0]  # Use first table
                else:
                    print(f"[TIME ETF] No table found for {date_str}")
                    # Print first 500 chars of HTML for debugging
                    print(f"[TIME ETF] HTML preview: {str(soup)[:500]}")
                    return pd.DataFrame()

            tbody = table.find('tbody')
            if not tbody:
                print(f"[TIME ETF] No tbody found for {date_str}, trying to find rows directly")
                # tbody가 없으면 table에서 직접 tr 찾기
                rows = table.find_all('tr')
            else:
                rows = tbody.find_all('tr')

            print(f"[TIME ETF] Found {len(rows)} rows")

            data = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 5:
                    continue

                try:
                    # Extract data from columns
                    ticker = cols[0].text.strip()
                    name = cols[1].text.strip()
                    shares_str = cols[2].text.strip().replace(',', '')
                    amount_str = cols[3].text.strip().replace(',', '').replace('원', '').replace('$', '')
                    weight_str = cols[4].text.strip().replace('%', '')

                    # Skip cash holdings and empty rows
                    if not ticker or 'CASH' in ticker.upper() or '현금' in name:
                        continue

                    row_data = {
                        '종목코드': ticker,
                        '종목명': name,
                        '보유수량': float(shares_str) if shares_str and shares_str != '0' else 0,
                        '평가금액': float(amount_str) if amount_str else 0,
                        '비중': float(weight_str) if weight_str else 0,
                        '날짜': date_str
                    }
                    data.append(row_data)

                except Exception as e:
                    print(f"[TIME ETF] Error parsing row: {e} - {[c.text.strip() for c in cols]}")
                    continue

            if not data:
                print(f"[TIME ETF] No data parsed for {date_str}")
                return pd.DataFrame()

            return pd.DataFrame(data)

        except Exception as e:
            print(f"[TIME ETF] Scraping Error: {e}")
            return pd.DataFrame()
    
    def get_portfolio_data(self, date: str) -> pd.DataFrame:
        """
        Public method to get data. Writes to file cache if new.
        """
        df = self.fetch_data_from_web(date)
        if not df.empty:
            self.save_data(df, date)
        return df
    
    def save_data(self, df: pd.DataFrame, date: str):
        filename = os.path.join(self.data_dir, f"portfolio_{date}.json")
        # Replace NaN with None for proper JSON serialization
        df = df.where(pd.notna(df), None)
        df.to_json(filename, orient='records', force_ascii=False, indent=2)
    
    def load_data(self, date: str) -> pd.DataFrame:
        # Check cache first
        filename = os.path.join(self.data_dir, f"portfolio_{date}.json")
        if os.path.exists(filename):
            return pd.read_json(filename)
        
        # If not in cache, try fetching
        df = self.fetch_data_from_web(date)
        if not df.empty:
            self.save_data(df, date)
            return df
        return None
    
    def get_previous_business_day(self, date_str: str, lookback_days: int = 7) -> Optional[str]:
        """
        Finds the nearest previous date with valid data.
        """
        curr = datetime.strptime(date_str, "%Y-%m-%d")
        
        for i in range(1, lookback_days + 1):
            prev = curr - timedelta(days=i)
            prev_str = prev.strftime("%Y-%m-%d")
            
            # Check if data exists
            df = self.load_data(prev_str)
            if df is not None and not df.empty:
                return prev_str
        
        return None
    
    def _ticker_from_code(self, code: str) -> str:
        """
        종목코드를 yfinance 티커로 변환

        Args:
            code: PDF 종목코드 (예: "NVDA US EQUITY", "ESZ5 Index", "BRK/B US EQUITY", "CA13321L1085")

        Returns:
            yfinance 티커 (예: "NVDA", "BRK-B", "^GSPC", "CCJ")
        """
        code = code.strip()

        # ISIN 코드 먼저 체크 (길이가 12자이고 공백이 없는 경우)
        # 예: CA13321L1085 (ISIN)
        # 제외: "PG US EQUITY" (12자이지만 공백 있음)
        if len(code) == 12 and ' ' not in code:
            if code in self.ISIN_TO_TICKER:
                return self.ISIN_TO_TICKER[code]
            else:
                # 매핑되지 않은 ISIN 코드
                return None

        # 선물 처리
        if 'Index' in code or 'FUT' in code:
            # S&P500 선물
            if 'S&P' in code or 'ES' in code:
                return '^GSPC'  # S&P 500 Index로 대체
            # NASDAQ 100 선물 (NQZ5, NQH6 등)
            if 'NQ' in code:
                return 'NQ=F'  # NASDAQ 100 E-MINI Futures
            # 기타 선물은 기초자산 반환 또는 None
            return None

        # US EQUITY 제거
        if 'US EQUITY' in code:
            ticker = code.replace('US EQUITY', '').strip()
        # CT EQUITY 제거 (캐나다 주식 - 토론토 증권거래소)
        elif 'CT EQUITY' in code:
            ticker = code.replace('CT EQUITY', '').strip() + '.TO'
        else:
            ticker = code

        # 티커 형식 변환: "/" → "-" (BRK/B → BRK-B, BRK/A → BRK-A)
        # yfinance는 클래스 주식을 하이픈으로 표기
        if '/' in ticker:
            ticker = ticker.replace('/', '-')

        return ticker if ticker else None

    def get_market_returns(self, df_prev: pd.DataFrame, df_today: pd.DataFrame,
                          date_prev: str, date_today: str) -> Dict[str, float]:
        """
        yfinance로 각 종목의 시장 수익률 가져오기 (텔레그램 로직 적용)

        대시보드 환경 고려사항:
        - date_prev, date_today는 사용자가 선택한 날짜와 그 이전 영업일
        - yfinance는 항상 최신 데이터만 제공하므로, period="5d"로 최근 5일 데이터 사용
        - 선택한 날짜가 과거인 경우 PDF 데이터로 fallback
        """
        market_returns = {}
        print(f"📊 yfinance로 시장 수익률 수집 중...")

        for _, row in df_prev.iterrows():
            code = row['종목코드']
            stock_name = row['종목명']

            # 현금은 0% 처리
            if stock_name == '현금' or code == '':
                market_returns[code] = 0.0
                continue

            ticker_symbol = self._ticker_from_code(code)

            # 티커 변환 실패 시 PDF fallback
            if not ticker_symbol:
                try:
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['보유수량'] > 0 and today_row.iloc[0]['보유수량'] > 0:
                        prev_price = row['평가금액'] / row['보유수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"ℹ️  {code[:20]} ({stock_name}): yfinance 미지원, PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                        print(f"ℹ️  {code[:20]} ({stock_name}): yfinance 미지원, 0% 사용")
                except:
                    market_returns[code] = 0.0
                continue

            try:
                # yfinance로 특정 날짜 범위 데이터 가져오기
                # 중요: ETF 포트폴리오 날짜는 전날 미국 종가 기준
                # 예: 1/30 포트폴리오 = 1/29 종가 기준 → date_today를 하루 빼서 조회
                from datetime import datetime, timedelta

                date_prev_dt = datetime.strptime(date_prev, '%Y-%m-%d')
                date_today_dt = datetime.strptime(date_today, '%Y-%m-%d')

                # ETF 포트폴리오는 전날 종가 기준이므로 하루 빼기
                date_prev_price_dt = date_prev_dt - timedelta(days=1)
                date_today_price_dt = date_today_dt - timedelta(days=1)

                start_date = (date_prev_price_dt - timedelta(days=5)).strftime('%Y-%m-%d')
                end_date = (date_today_price_dt + timedelta(days=5)).strftime('%Y-%m-%d')

                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(start=start_date, end=end_date)

                if len(hist) < 2:
                    # 데이터 부족 시 PDF fallback
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['보유수량'] > 0 and today_row.iloc[0]['보유수량'] > 0:
                        prev_price = row['평가금액'] / row['보유수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"ℹ️  {ticker_symbol} ({stock_name}): yfinance 데이터 부족, PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                        print(f"⚠️  {ticker_symbol} ({stock_name}): yfinance 데이터 부족, 0% 사용")
                    continue

                # 요청한 날짜에 가장 가까운 영업일 찾기
                hist.index = hist.index.tz_localize(None)  # timezone 제거

                # date_prev 포트폴리오에 해당하는 종가 (date_prev - 1)
                prev_candidates = hist[hist.index <= date_prev_price_dt]
                if len(prev_candidates) == 0:
                    prev_candidates = hist  # fallback
                prev_close = prev_candidates.iloc[-1]['Close']
                prev_date_used = prev_candidates.iloc[-1].name.strftime('%Y-%m-%d')

                # date_today 포트폴리오에 해당하는 종가 (date_today - 1)
                today_candidates = hist[hist.index <= date_today_price_dt]
                if len(today_candidates) == 0:
                    today_candidates = hist  # fallback
                today_close = today_candidates.iloc[-1]['Close']
                today_date_used = today_candidates.iloc[-1].name.strftime('%Y-%m-%d')

                # 수익률 계산
                market_return = (today_close / prev_close - 1) if prev_close > 0 else 0.0
                market_returns[code] = market_return
                print(f"✓ {ticker_symbol} ({stock_name}): {market_return*100:+.2f}% ({prev_date_used} → {today_date_used})")

            except Exception as e:
                # 오류 발생 시 PDF fallback
                try:
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['보유수량'] > 0 and today_row.iloc[0]['보유수량'] > 0:
                        prev_price = row['평가금액'] / row['보유수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"⚠️  {ticker_symbol} ({stock_name}): yfinance 오류, PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                        print(f"⚠️  {ticker_symbol} ({stock_name}): yfinance 오류, 0% 사용")
                except:
                    market_returns[code] = 0.0

        return market_returns

    def analyze_rebalancing(self, df_today: pd.DataFrame, df_prev: pd.DataFrame,
                          date_today: str = None, date_prev: str = None) -> Dict:
        """
        리밸런싱 분석 (텔레그램 로직 적용)

        시장 가격 변동만으로 설명되지 않는 비중 변화를 리밸런싱으로 감지
        AUM 변화와 가격 변동 효과를 모두 제거

        대시보드 환경 고려:
        - 사용자가 선택한 날짜(date_today)와 이전 영업일(date_prev) 비교
        - yfinance는 최신 데이터만 제공하므로 과거 날짜 선택 시 PDF 데이터로 보정
        """
        # 종목코드 기준으로 병합
        merged = pd.merge(
            df_today[['종목코드', '종목명', '보유수량', '평가금액', '비중']],
            df_prev[['종목코드', '종목명', '보유수량', '평가금액', '비중']],
            on='종목코드',
            how='outer',
            suffixes=('_today', '_prev')
        )

        # 종목명 통합 (금일 우선)
        merged['종목명'] = merged['종목명_today'].fillna(merged['종목명_prev'])

        # 숫자 컬럼만 0으로 채우기
        numeric_columns = ['보유수량_today', '보유수량_prev', '평가금액_today', '평가금액_prev', '비중_today', '비중_prev']
        merged[numeric_columns] = merged[numeric_columns].fillna(0)

        # 1단계: 시장 수익률 가져오기
        if date_prev and date_today:
            market_returns = self.get_market_returns(df_prev, df_today, date_prev, date_today)
        else:
            # 날짜 없으면 PDF 데이터로 fallback
            print(f"⚠️  날짜 정보 없음, PDF 데이터로 수익률 계산")
            market_returns = {}
            for _, row in df_prev.iterrows():
                code = row['종목코드']
                prev_price = row['평가금액'] / row['보유수량'] if row['보유수량'] > 0 else 0
                today_row = df_today[df_today['종목코드'] == code]
                if len(today_row) > 0:
                    today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량'] if today_row.iloc[0]['보유수량'] > 0 else 0
                    market_returns[code] = (today_price / prev_price - 1) if prev_price > 0 else 0
                else:
                    market_returns[code] = 0

        # 시장 수익률을 merged에 추가
        merged['시장_수익률'] = merged['종목코드'].map(market_returns).fillna(0)

        # 2단계: 가상 비중 계산 (시장 변동만 반영)
        merged['가상_비중'] = merged['비중_prev'] * (1 + merged['시장_수익률'])

        # 3단계: 정규화 (100%로 스케일링) ⭐ 핵심!
        total_virtual_weight = merged['가상_비중'].sum()
        if total_virtual_weight > 0:
            merged['예상_비중'] = merged['가상_비중'] / total_virtual_weight * 100
        else:
            merged['예상_비중'] = 0

        # 4단계: 실제 비중 변화 vs 예상 비중 변화
        merged['순수_비중변화'] = merged['비중_today'] - merged['예상_비중']

        # 5단계: 수량 변화 확인
        merged['수량_변화'] = merged['보유수량_today'] - merged['보유수량_prev']

        # 리밸런싱 감지
        # - 의미있는 비중 변화 (±0.5%p 이상)
        # - 또는 편입/편출 (수량이 0에서 변화)
        # - 현금 제외
        threshold = 0.5
        rebalanced = merged[
            ((abs(merged['순수_비중변화']) >= threshold) |
             (merged['보유수량_prev'] == 0) |
             (merged['보유수량_today'] == 0)) &
            (merged['종목명'] != '현금')
        ].copy()

        # 편입/편출/비중확대/비중축소 구분
        new_stocks = rebalanced[(rebalanced['보유수량_prev'] == 0) & (rebalanced['보유수량_today'] > 0)]
        removed_stocks = rebalanced[(rebalanced['보유수량_today'] == 0) & (rebalanced['보유수량_prev'] > 0)]

        # 비중 확대/축소는 순수 비중 변화 + 수량 변화 모두 체크 ⭐ 핵심!
        # 수량이 증가했고, 비중도 의미있게 증가한 경우만
        increased_stocks = rebalanced[(rebalanced['순수_비중변화'] > threshold) &
                                     (rebalanced['수량_변화'] > 0) &
                                     (rebalanced['보유수량_prev'] > 0) &
                                     (rebalanced['보유수량_today'] > 0)]
        decreased_stocks = rebalanced[(rebalanced['순수_비중변화'] < -threshold) &
                                     (rebalanced['수량_변화'] < 0) &
                                     (rebalanced['보유수량_prev'] > 0) &
                                     (rebalanced['보유수량_today'] > 0)]

        # Clean data for JSON serialization
        def clean_records(df):
            df = df.where(pd.notna(df), None)
            return df.to_dict('records')

        return {
            'new_stocks': clean_records(new_stocks),
            'removed_stocks': clean_records(removed_stocks),
            'increased_stocks': clean_records(increased_stocks),
            'decreased_stocks': clean_records(decreased_stocks)
        }


class KiwoomETFMonitor:
    """
    Monitor for Kiwoom KOSEF Active ETF (US Growth 30)
    Target: 459790 (KOSEF 미국성장기업30 Active)
    Source: AJAX API (https://www.kiwoometf.com/service/etf/KO02010200MAjax4)

    Note: 미국 주식을 편입하므로 TIME ETF와 동일한 리밸런싱 로직 사용
    """

    API_URL = "https://www.kiwoometf.com/service/etf/KO02010200MAjax4"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://www.kiwoometf.com',
        'Referer': 'https://www.kiwoometf.com/service/etf/KO02010200M?gcode=459790'
    }
    KST = pytz.timezone('Asia/Seoul')

    def __init__(self, data_dir: str = "./data/kiwoom_etf"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.etf_code = "459790"
        self.etf_name = "KIWOOM 미국성장기업30액티브"
    
    def fetch_data_from_api(self, date_str: str) -> pd.DataFrame:
        """
        Fetch portfolio data for a specific date (YYYY-MM-DD) via API.
        """
        # API expects YYYYMMDD
        date_api = date_str.replace("-", "")
        
        payload = {
            "schGubun1": self.etf_code,
            "startDate": date_api
        }
        
        try:
            resp = requests.post(self.API_URL, data=payload, headers=self.HEADERS, verify=False, timeout=10)
            if resp.status_code != 200:
                print(f"[Kiwoom] Status {resp.status_code} for {date_str}")
                return pd.DataFrame()
            
            js = resp.json()
            if 'pdfList' not in js or not js['pdfList']:
                return pd.DataFrame()
            
            data = []
            for item in js['pdfList']:
                try:
                    vol_str = item.get('volume', '0').replace(',', '')
                    amt_str = item.get('assessment', '0').replace(',', '')
                    ratio_str = item.get('ratio', '0').replace('%', '')
                    
                    # Skip cash
                    item_code = item.get('itemCode', '')
                    if 'CASH' in item_code.upper():
                        continue
                    
                    row = {
                        '종목코드': item_code,
                        '종목명': item.get('itemTitle', ''),
                        '보유수량': float(vol_str),
                        '평가금액': float(amt_str),
                        '비중': float(ratio_str),
                        '날짜': date_str
                    }
                    data.append(row)
                except Exception as e:
                    continue
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"[Kiwoom] API Error: {e}")
            return pd.DataFrame()
    
    def get_portfolio_data(self, date: str) -> pd.DataFrame:
        """
        Public method to get data. Writes to file cache if new.
        """
        df = self.fetch_data_from_api(date)
        if not df.empty:
            self.save_data(df, date)
        return df
    
    def save_data(self, df: pd.DataFrame, date: str):
        filename = os.path.join(self.data_dir, f"portfolio_{date}.json")
        # Replace NaN with None for proper JSON serialization
        df = df.where(pd.notna(df), None)
        df.to_json(filename, orient='records', force_ascii=False, indent=2)
    
    def load_data(self, date: str) -> pd.DataFrame:
        # Check cache first
        filename = os.path.join(self.data_dir, f"portfolio_{date}.json")
        if os.path.exists(filename):
            return pd.read_json(filename)
        
        # If not in cache, try fetching
        df = self.fetch_data_from_api(date)
        if not df.empty:
            self.save_data(df, date)
            return df
        return None
    
    def get_previous_business_day(self, date_str: str, lookback_days: int = 7) -> Optional[str]:
        """
        Finds the nearest previous date with valid data.
        """
        curr = datetime.strptime(date_str, "%Y-%m-%d")
        
        for i in range(1, lookback_days + 1):
            prev = curr - timedelta(days=i)
            prev_str = prev.strftime("%Y-%m-%d")
            
            # Check if data exists
            df = self.load_data(prev_str)
            if df is not None and not df.empty:
                return prev_str
        
        return None
    
    def _ticker_from_code(self, code: str) -> str:
        """
        종목코드를 yfinance 티커로 변환 (TIME ETF와 동일)
        """
        code = code.strip()

        # 선물 처리
        if 'Index' in code or 'FUT' in code:
            if 'S&P' in code or 'ES' in code:
                return '^GSPC'
            if 'NQ' in code:
                return 'NQ=F'
            return None

        # US EQUITY 제거
        if 'US EQUITY' in code:
            ticker = code.replace('US EQUITY', '').strip()
        else:
            ticker = code

        # "/" → "-" 변환
        if '/' in ticker:
            ticker = ticker.replace('/', '-')

        return ticker if ticker else None

    def get_market_returns(self, df_prev: pd.DataFrame, df_today: pd.DataFrame,
                          date_prev: str, date_today: str) -> Dict[str, float]:
        """
        yfinance로 각 종목의 시장 수익률 가져오기 (TIME ETF와 동일 로직)
        """
        market_returns = {}
        print(f"📊 [Kiwoom] yfinance로 시장 수익률 수집 중...")

        for _, row in df_prev.iterrows():
            code = row['종목코드']
            stock_name = row['종목명']

            # 현금 처리
            if stock_name == '현금' or code == '':
                market_returns[code] = 0.0
                continue

            ticker_symbol = self._ticker_from_code(code)

            # 티커 변환 실패 시 PDF fallback
            if not ticker_symbol:
                try:
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['보유수량'] > 0 and today_row.iloc[0]['보유수량'] > 0:
                        prev_price = row['평가금액'] / row['보유수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"ℹ️  {code[:20]} ({stock_name}): PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                except:
                    market_returns[code] = 0.0
                continue

            try:
                # yfinance로 특정 날짜 범위 데이터 가져오기
                # ETF 포트폴리오는 전날 미국 종가 기준
                from datetime import datetime, timedelta

                date_prev_dt = datetime.strptime(date_prev, '%Y-%m-%d')
                date_today_dt = datetime.strptime(date_today, '%Y-%m-%d')

                # ETF 포트폴리오는 전날 종가 기준이므로 하루 빼기
                date_prev_price_dt = date_prev_dt - timedelta(days=1)
                date_today_price_dt = date_today_dt - timedelta(days=1)

                start_date = (date_prev_price_dt - timedelta(days=5)).strftime('%Y-%m-%d')
                end_date = (date_today_price_dt + timedelta(days=5)).strftime('%Y-%m-%d')

                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(start=start_date, end=end_date)

                if len(hist) < 2:
                    # PDF fallback
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['보유수량'] > 0 and today_row.iloc[0]['보유수량'] > 0:
                        prev_price = row['평가금액'] / row['보유수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"ℹ️  {ticker_symbol} ({stock_name}): PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                    continue

                # 요청한 날짜에 가장 가까운 영업일 찾기
                hist.index = hist.index.tz_localize(None)  # timezone 제거

                # date_prev 포트폴리오에 해당하는 종가 (date_prev - 1)
                prev_candidates = hist[hist.index <= date_prev_price_dt]
                if len(prev_candidates) == 0:
                    prev_candidates = hist
                prev_close = prev_candidates.iloc[-1]['Close']
                prev_date_used = prev_candidates.iloc[-1].name.strftime('%Y-%m-%d')

                # date_today 포트폴리오에 해당하는 종가 (date_today - 1)
                today_candidates = hist[hist.index <= date_today_price_dt]
                if len(today_candidates) == 0:
                    today_candidates = hist
                today_close = today_candidates.iloc[-1]['Close']
                today_date_used = today_candidates.iloc[-1].name.strftime('%Y-%m-%d')

                market_return = (today_close / prev_close - 1) if prev_close > 0 else 0.0
                market_returns[code] = market_return
                print(f"✓ {ticker_symbol} ({stock_name}): {market_return*100:+.2f}% ({prev_date_used} → {today_date_used})")

            except:
                try:
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['보유수량'] > 0 and today_row.iloc[0]['보유수량'] > 0:
                        prev_price = row['평가금액'] / row['보유수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"⚠️  {ticker_symbol} ({stock_name}): PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                except:
                    market_returns[code] = 0.0

        return market_returns

    def analyze_rebalancing(self, df_today: pd.DataFrame, df_prev: pd.DataFrame,
                          date_today: str = None, date_prev: str = None) -> Dict:
        """
        리밸런싱 분석 (텔레그램 로직 적용 - TIME ETF와 동일)
        Kiwoom ETF도 미국 주식 편입이므로 동일한 가격 보정 로직 사용
        """
        merged = pd.merge(
            df_today[['종목코드', '종목명', '보유수량', '평가금액', '비중']],
            df_prev[['종목코드', '종목명', '보유수량', '평가금액', '비중']],
            on='종목코드',
            how='outer',
            suffixes=('_today', '_prev')
        )

        merged['종목명'] = merged['종목명_today'].fillna(merged['종목명_prev'])

        numeric_columns = ['보유수량_today', '보유수량_prev', '평가금액_today', '평가금액_prev', '비중_today', '비중_prev']
        merged[numeric_columns] = merged[numeric_columns].fillna(0)

        # 시장 수익률 가져오기
        if date_prev and date_today:
            market_returns = self.get_market_returns(df_prev, df_today, date_prev, date_today)
        else:
            market_returns = {}
            for _, row in df_prev.iterrows():
                code = row['종목코드']
                prev_price = row['평가금액'] / row['보유수량'] if row['보유수량'] > 0 else 0
                today_row = df_today[df_today['종목코드'] == code]
                if len(today_row) > 0:
                    today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['보유수량'] if today_row.iloc[0]['보유수량'] > 0 else 0
                    market_returns[code] = (today_price / prev_price - 1) if prev_price > 0 else 0
                else:
                    market_returns[code] = 0

        merged['시장_수익률'] = merged['종목코드'].map(market_returns).fillna(0)
        merged['가상_비중'] = merged['비중_prev'] * (1 + merged['시장_수익률'])

        # 정규화
        total_virtual_weight = merged['가상_비중'].sum()
        if total_virtual_weight > 0:
            merged['예상_비중'] = merged['가상_비중'] / total_virtual_weight * 100
        else:
            merged['예상_비중'] = 0

        merged['순수_비중변화'] = merged['비중_today'] - merged['예상_비중']
        merged['수량_변화'] = merged['보유수량_today'] - merged['보유수량_prev']

        threshold = 0.5
        rebalanced = merged[
            ((abs(merged['순수_비중변화']) >= threshold) |
             (merged['보유수량_prev'] == 0) |
             (merged['보유수량_today'] == 0)) &
            (merged['종목명'] != '현금')
        ].copy()

        new_stocks = rebalanced[(rebalanced['보유수량_prev'] == 0) & (rebalanced['보유수량_today'] > 0)]
        removed_stocks = rebalanced[(rebalanced['보유수량_today'] == 0) & (rebalanced['보유수량_prev'] > 0)]
        increased_stocks = rebalanced[(rebalanced['순수_비중변화'] > threshold) &
                                     (rebalanced['수량_변화'] > 0) &
                                     (rebalanced['보유수량_prev'] > 0) &
                                     (rebalanced['보유수량_today'] > 0)]
        decreased_stocks = rebalanced[(rebalanced['순수_비중변화'] < -threshold) &
                                     (rebalanced['수량_변화'] < 0) &
                                     (rebalanced['보유수량_prev'] > 0) &
                                     (rebalanced['보유수량_today'] > 0)]

        def clean_records(df):
            df = df.where(pd.notna(df), None)
            return df.to_dict('records')

        return {
            'new_stocks': clean_records(new_stocks),
            'removed_stocks': clean_records(removed_stocks),
            'increased_stocks': clean_records(increased_stocks),
            'decreased_stocks': clean_records(decreased_stocks)
        }
