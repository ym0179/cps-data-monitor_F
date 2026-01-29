import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz
import urllib3

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
            "date": date_param
        }
        
        try:
            resp = requests.get(self.BASE_URL, params=params, headers=self.HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f"[TIME ETF] Status {resp.status_code} for {date_str}")
                return pd.DataFrame()
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find portfolio table
            table = soup.find('table', {'class': 'table'})
            if not table:
                return pd.DataFrame()
            
            data = []
            rows = table.find('tbody').find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 5:
                    continue
                
                try:
                    # Extract data from columns
                    ticker = cols[0].text.strip()
                    name = cols[1].text.strip()
                    shares_str = cols[2].text.strip().replace(',', '')
                    amount_str = cols[3].text.strip().replace(',', '').replace('$', '')
                    weight_str = cols[4].text.strip().replace('%', '')
                    
                    # Skip cash holdings
                    if 'CASH' in ticker.upper() or '현금' in name:
                        continue
                    
                    row_data = {
                        '종목코드': ticker,
                        '종목명': name,
                        '보유수량': float(shares_str) if shares_str else 0,
                        '평가금액': float(amount_str) if amount_str else 0,
                        '비중': float(weight_str) if weight_str else 0,
                        '날짜': date_str
                    }
                    data.append(row_data)
                    
                except Exception as e:
                    print(f"[TIME ETF] Error parsing row: {e}")
                    continue
            
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
    
    def analyze_rebalancing(self, df_today: pd.DataFrame, df_prev: pd.DataFrame) -> Dict:
        """
        Analyze changes based on SHARES and WEIGHTS.
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
        weight_threshold = 0.5  # 0.5%p
        
        # Categorize changes
        new_stocks = merged[(merged['보유수량_prev'] == 0) & (merged['보유수량_today'] > 0)]
        removed_stocks = merged[(merged['보유수량_today'] == 0) & (merged['보유수량_prev'] > 0)]
        
        # Weight changes for existing positions
        existing = merged[(merged['보유수량_prev'] > 0) & (merged['보유수량_today'] > 0)]
        increased = existing[existing['비중변화'] >= weight_threshold]
        decreased = existing[existing['비중변화'] <= -weight_threshold]
        
        return {
            'new_stocks': new_stocks.to_dict('records'),
            'removed_stocks': removed_stocks.to_dict('records'),
            'increased_stocks': increased.to_dict('records'),
            'decreased_stocks': decreased.to_dict('records')
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
    
    def analyze_rebalancing(self, df_today: pd.DataFrame, df_prev: pd.DataFrame) -> Dict:
        """
        Analyze changes based on SHARES and WEIGHTS.
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
        
        return {
            'new_stocks': new_stocks.to_dict('records'),
            'removed_stocks': removed_stocks.to_dict('records'),
            'increased_stocks': increased.to_dict('records'),
            'decreased_stocks': decreased.to_dict('records')
        }
