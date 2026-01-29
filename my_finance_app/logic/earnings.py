"""
Earnings Trading 데이터 수집 로직 (Yahoo Finance 기반)
"""
import requests
import pandas as pd
import datetime
import urllib3
from bs4 import BeautifulSoup
import yfinance as yf

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_yahoo_earnings_calendar(date_str=None, days=7):
    """
    Yahoo Finance Earnings Calendar 크롤링

    Args:
        date_str: Starting date in 'YYYY-MM-DD' format (default: today)
        days: Number of days to fetch (default: 7)

    Returns:
        DataFrame with columns: Symbol, Company, CallTime, EPSEstimate, ReportedEPS, Surprise, MarketCap, Date
    """
    if date_str is None:
        start_date = datetime.date.today()
    else:
        start_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

    all_data = []

    for i in range(days):
        target_date = start_date + datetime.timedelta(days=i)
        date_str_query = target_date.strftime("%Y-%m-%d")

        url = f"https://finance.yahoo.com/calendar/earnings?day={date_str_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://finance.yahoo.com/'
        }

        try:
            response = requests.get(url, headers=headers, verify=False, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find table with earnings data
                table = soup.find('table', {'class': 'yf-1uayyp1'})

                if table:
                    tbody = table.find('tbody')
                    if tbody:
                        rows = tbody.find_all('tr', {'class': 'row'})

                        for row in rows:
                            cells = row.find_all('td')

                            if len(cells) >= 8:
                                # Extract data from cells
                                symbol_cell = cells[0].find('a')
                                symbol = symbol_cell.get_text(strip=True) if symbol_cell else cells[0].get_text(strip=True)

                                company = cells[1].get_text(strip=True)
                                call_time_span = cells[3].find('span')
                                call_time = call_time_span.get('title') if call_time_span and call_time_span.get('title') else cells[3].get_text(strip=True)

                                eps_estimate = cells[4].get_text(strip=True)
                                reported_eps = cells[5].get_text(strip=True)
                                surprise = cells[6].get_text(strip=True)
                                market_cap = cells[7].get_text(strip=True)

                                all_data.append({
                                    'Symbol': symbol,
                                    'Company': company,
                                    'CallTime': call_time,
                                    'EPSEstimate': eps_estimate,
                                    'ReportedEPS': reported_eps,
                                    'Surprise': surprise,
                                    'MarketCap': market_cap,
                                    'Date': date_str_query
                                })
        except Exception as e:
            # Silently pass - some dates may not have data
            pass

    if all_data:
        df = pd.DataFrame(all_data)

        # Convert Call Time to Korean
        df['CallTime'] = df['CallTime'].apply(convert_call_time_to_korean)

        return df
    else:
        return pd.DataFrame()


def convert_call_time_to_korean(call_time):
    """
    Convert Yahoo Finance Call Time to Korean

    Args:
        call_time: 'After market close' (AMC) or 'Before market open' (BMO) or abbreviation

    Returns:
        Korean string: '장후' or '장전'
    """
    ct_lower = str(call_time).lower()

    if 'amc' in ct_lower or 'after' in ct_lower:
        return '장후'
    elif 'bmo' in ct_lower or 'tas' in ct_lower or 'before' in ct_lower or 'pre' in ct_lower:
        return '장전'
    else:
        return call_time  # Return original if unknown


def fetch_analyst_consensus(ticker):
    """
    Fetch Analyst Consensus from Yahoo Finance (yfinance)

    Args:
        ticker: Stock symbol

    Returns:
        dict with keys: targetMean, targetHigh, targetLow, recommendMean, recommendKey, analystCount
    """
    result = {
        'targetMean': None,
        'targetHigh': None,
        'targetLow': None,
        'recommendMean': None,  # 1.0 (Strong Buy) ~ 5.0 (Sell)
        'recommendKey': None,
        'analystCount': 0
    }

    try:
        session = requests.Session()
        session.verify = False
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        t = yf.Ticker(ticker, session=session)
        info = t.info

        if info and 'targetMeanPrice' in info:
            result['targetMean'] = info.get('targetMeanPrice')
            result['targetHigh'] = info.get('targetHighPrice')
            result['targetLow'] = info.get('targetLowPrice')
            result['recommendMean'] = info.get('recommendationMean')
            result['recommendKey'] = info.get('recommendationKey')
            result['analystCount'] = info.get('numberOfAnalystOpinions', 0)

    except Exception:
        pass

    return result


def fetch_historical_price(ticker, years=3):
    """
    Fetch historical price data from Yahoo Finance

    Args:
        ticker: Stock symbol
        years: Number of years of data (default: 3)

    Returns:
        DataFrame with Date index and Close column
    """
    try:
        session = requests.Session()
        session.verify = False
        session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })

        t = yf.Ticker(ticker, session=session)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * years)

        hist = t.history(start=start_date, end=end_date)

        if not hist.empty:
            df = pd.DataFrame()
            df['Close'] = hist['Close']
            df.index = pd.to_datetime(df.index).normalize()
            return df

    except Exception:
        pass

    return pd.DataFrame()


def fetch_historical_earnings_dates(ticker):
    """
    Fetch historical earnings dates from Yahoo Finance

    Args:
        ticker: Stock symbol

    Returns:
        List of datetime objects (earnings dates)
    """
    dates_list = []

    try:
        session = requests.Session()
        session.verify = False
        session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })

        t = yf.Ticker(ticker, session=session)

        # earnings_dates returns DataFrame with index = datetime
        earnings_dates = t.earnings_dates

        if earnings_dates is not None and not earnings_dates.empty:
            dates_list = [pd.Timestamp(dt).normalize() for dt in earnings_dates.index]

    except Exception:
        pass

    # Deduplicate and sort
    unique_dates = sorted(list(set(dates_list)))

    return unique_dates


def calculate_earnings_metrics(ticker, years=3):
    """
    Calculate Avg Earnings Move and Earnings Volatility

    Args:
        ticker: Stock symbol
        years: Number of years to analyze (default: 3)

    Returns:
        dict with keys: avg_move (%), volatility (%), num_earnings
    """
    result = {
        'avg_move': None,
        'volatility': None,
        'num_earnings': 0
    }

    try:
        # 1. Get historical earnings dates
        earnings_dates = fetch_historical_earnings_dates(ticker)

        if not earnings_dates:
            return result

        # Filter to last N years
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=365 * years)
        earnings_dates = [d for d in earnings_dates if d >= cutoff_date]

        if not earnings_dates:
            return result

        # 2. Get historical prices
        price_df = fetch_historical_price(ticker, years=years)

        if price_df.empty:
            return result

        # 3. Calculate returns on earnings dates
        returns = []

        for earnings_date in earnings_dates:
            # Find price on earnings date (or closest available date)
            if earnings_date in price_df.index:
                price_current = price_df.loc[earnings_date, 'Close']

                # Find previous day price
                prev_dates = price_df.index[price_df.index < earnings_date]
                if len(prev_dates) > 0:
                    prev_date = prev_dates[-1]
                    price_prev = price_df.loc[prev_date, 'Close']

                    # Calculate return
                    ret = (price_current - price_prev) / price_prev * 100
                    returns.append(ret)

        if returns:
            # Avg Move = Mean of absolute returns
            result['avg_move'] = round(sum(abs(r) for r in returns) / len(returns), 2)

            # Volatility = Std of returns
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            result['volatility'] = round(variance ** 0.5, 2)

            result['num_earnings'] = len(returns)

    except Exception:
        pass

    return result


def calculate_trade_alert(avg_move, analyst_rating, analyst_count):
    """
    Calculate Trade Alert flag based on Goldman Sachs methodology

    Conditions:
    - High Volatility: Avg Move > 5%
    - Analyst Support: Rating is 'strong buy' or 'buy'
    - Minimum Coverage: # Analysts >= 5

    Args:
        avg_move: Average earnings move (%)
        analyst_rating: 'strong buy', 'buy', 'hold', 'sell', 'strong sell'
        analyst_count: Number of analysts

    Returns:
        String: '⭐ High Move + Analyst Support' or '-'
    """
    if avg_move is None or analyst_rating is None:
        return '-'

    rating_lower = str(analyst_rating).lower()

    high_vol = avg_move > 5.0
    analyst_support = ('strong buy' in rating_lower or 'buy' in rating_lower)
    min_coverage = analyst_count >= 5

    if high_vol and analyst_support and min_coverage:
        return '⭐ High Move + Analyst Support'
    else:
        return '-'
