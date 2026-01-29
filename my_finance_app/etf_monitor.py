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
    
    def __init__(self, etf_idx: str, etf_name: str, data_dir: str = "./data/time_etf"):
        self.etf_idx = etf_idx
        self.etf_name = etf_name
        self.data_dir = os.path.join(data_dir, f"etf_{etf_idx}")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_data_from_web(self, date_str: str) -> pd.DataFrame:
        """
        Fetch portfolio data for a specific date (YYYY-MM-DD) via web scraping.
        """
        # Convert date format for URL parameter
        date_param = date_str.replace("-", "")

        params = {
            "idx": self.etf_idx,
            "pdfDate": date_param
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, headers=self.HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f"[TIME ETF] Status {resp.status_code} for {date_str}")
                return pd.DataFrame()

            soup = BeautifulSoup(resp.content, 'html.parser')

            # Find portfolio table (class="table3 moreList1")
            table = soup.find('table', {'class': ['table3', 'moreList1']})
            if not table:
                print(f"[TIME ETF] No table found for {date_str}")
                return pd.DataFrame()

            tbody = table.find('tbody')
            if not tbody:
                print(f"[TIME ETF] No tbody found for {date_str}")
                return pd.DataFrame()

            data = []
            rows = tbody.find_all('tr')

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
    
    def fetch_yahoo_prices(self, tickers: List[str], date_str: str) -> Dict[str, float]:
        """
        Fetch closing prices from Yahoo Finance for given date.
        Converts TIME ETF ticker format to Yahoo format (e.g., "AAPL US EQUITY" -> "AAPL")
        """
        prices = {}
        target_date = datetime.strptime(date_str, "%Y-%m-%d")

        for ticker in tickers:
            try:
                # Extract base ticker (remove " US EQUITY" suffix, handle futures/ETFs)
                clean_ticker = ticker.split()[0]

                # Skip futures and indices
                if 'Index' in ticker or 'FUT' in ticker.upper():
                    continue

                # Fetch data from Yahoo Finance
                stock = yf.Ticker(clean_ticker)
                hist = stock.history(start=target_date - timedelta(days=7), end=target_date + timedelta(days=1))

                if not hist.empty:
                    # Get the closest available price
                    closest_price = hist.loc[hist.index <= target_date, 'Close']
                    if not closest_price.empty:
                        prices[ticker] = float(closest_price.iloc[-1])

            except Exception as e:
                print(f"[Yahoo Finance] Error fetching {ticker}: {e}")
                continue

        return prices

    def analyze_rebalancing(self, df_today: pd.DataFrame, df_prev: pd.DataFrame,
                          date_today: str = None, date_prev: str = None) -> Dict:
        """
        Analyze changes based on actual price-adjusted rebalancing.

        Logic:
        1. Get yesterday's closing prices from Yahoo Finance
        2. Calculate theoretical portfolio weights if no trading occurred (price changes only)
        3. Compare with actual portfolio weights to detect rebalancing trades
        """
        merged = pd.merge(
            df_today[['종목명', '종목코드', '보유수량', '비중', '평가금액']],
            df_prev[['종목명', '종목코드', '보유수량', '비중', '평가금액']],
            on='종목코드',
            how='outer',
            suffixes=('_today', '_prev')
        )

        merged['종목명'] = merged['종목명_today'].fillna(merged['종목명_prev'])

        # Fill NaNs with 0
        for col in ['보유수량_today', '보유수량_prev', '비중_today', '비중_prev', '평가금액_today', '평가금액_prev']:
            merged[col] = merged[col].fillna(0)

        # Calculate deltas
        merged['수량변화'] = merged['보유수량_today'] - merged['보유수량_prev']
        merged['비중변화'] = merged['비중_today'] - merged['비중_prev']

        # Fetch Yahoo Finance prices if dates provided
        if date_today and date_prev:
            all_tickers = merged['종목코드'].unique().tolist()
            prices_today = self.fetch_yahoo_prices(all_tickers, date_today)
            prices_prev = self.fetch_yahoo_prices(all_tickers, date_prev)

            # Calculate price-adjusted expected weights
            merged['price_return'] = merged['종목코드'].apply(
                lambda t: (prices_today.get(t, 1) / prices_prev.get(t, 1)) if prices_prev.get(t) else 1
            )

            # Expected weight = prev_weight * price_return (if no trading)
            # Normalize after price changes
            merged['expected_weight'] = merged['비중_prev'] * merged['price_return']
            total_expected = merged['expected_weight'].sum()
            if total_expected > 0:
                merged['expected_weight'] = merged['expected_weight'] / total_expected * 100

            # True rebalancing = actual weight - expected weight
            merged['true_rebalancing'] = merged['비중_today'] - merged['expected_weight']
        else:
            # Fallback to simple weight change
            merged['true_rebalancing'] = merged['비중변화']

        # Thresholds
        share_threshold = 1.0
        weight_threshold = 0.5  # 0.5%p true rebalancing

        # Categorize changes
        new_stocks = merged[(merged['보유수량_prev'] == 0) & (merged['보유수량_today'] > 0)]
        removed_stocks = merged[(merged['보유수량_today'] == 0) & (merged['보유수량_prev'] > 0)]

        # Weight changes for existing positions (using true rebalancing)
        existing = merged[(merged['보유수량_prev'] > 0) & (merged['보유수량_today'] > 0)]
        increased = existing[existing['true_rebalancing'] >= weight_threshold]
        decreased = existing[existing['true_rebalancing'] <= -weight_threshold]

        # Clean data for JSON serialization
        def clean_records(df):
            df = df.where(pd.notna(df), None)
            return df.to_dict('records')

        return {
            'new_stocks': clean_records(new_stocks),
            'removed_stocks': clean_records(removed_stocks),
            'increased_stocks': clean_records(increased),
            'decreased_stocks': clean_records(decreased)
        }


class KiwoomETFMonitor:
    """
    Monitor for Kiwoom KOSEF Active ETF (US Growth 30)
    Target: 459790 (KOSEF 미국성장기업30 Active)
    Source: AJAX API (https://www.kiwoometf.com/service/etf/KO02010200MAjax4)
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
        self.etf_name = "KOSEF 미국성장기업30 Active"
    
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
    
    def analyze_rebalancing(self, df_today: pd.DataFrame, df_prev: pd.DataFrame,
                          date_today: str = None, date_prev: str = None) -> Dict:
        """
        Analyze changes based on SHARES and WEIGHTS.
        For Kiwoom ETF (Korean stocks), we use simple weight change analysis.
        """
        merged = pd.merge(
            df_today[['종목명', '종목코드', '보유수량', '비중']],
            df_prev[['종목명', '종목코드', '보유수량', '비중']],
            on='종목코드',
            how='outer',
            suffixes=('_today', '_prev')
        )

        merged['종목명'] = merged['종목명_today'].fillna(merged['종목명_prev'])

        # Fill NaNs
        for col in ['보유수량_today', '보유수량_prev', '비중_today', '비중_prev']:
            merged[col] = merged[col].fillna(0)

        # Calculate deltas
        merged['수량변화'] = merged['보유수량_today'] - merged['보유수량_prev']
        merged['비중변화'] = merged['비중_today'] - merged['비중_prev']

        # Thresholds
        share_threshold = 1.0
        weight_threshold = 0.5

        # Categorize changes
        new_stocks = merged[(merged['보유수량_prev'] == 0) & (merged['보유수량_today'] > 0)]
        removed_stocks = merged[(merged['보유수량_today'] == 0) & (merged['보유수량_prev'] > 0)]

        # Weight changes for existing positions
        existing = merged[(merged['보유수량_prev'] > 0) & (merged['보유수량_today'] > 0)]
        increased = existing[existing['비중변화'] >= weight_threshold]
        decreased = existing[existing['비중변화'] <= -weight_threshold]

        # Clean data for JSON serialization
        def clean_records(df):
            df = df.where(pd.notna(df), None)
            return df.to_dict('records')

        return {
            'new_stocks': clean_records(new_stocks),
            'removed_stocks': clean_records(removed_stocks),
            'increased_stocks': clean_records(increased),
            'decreased_stocks': clean_records(decreased)
        }
