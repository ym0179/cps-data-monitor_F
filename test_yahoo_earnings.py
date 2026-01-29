#!/usr/bin/env python3
"""
Yahoo Finance Earnings Calendar 크롤링 테스트
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

def fetch_yahoo_earnings_calendar(date_str="2026-01-29"):
    """
    Yahoo Finance Earnings Calendar 크롤링

    Args:
        date_str: YYYY-MM-DD 형식

    Returns:
        DataFrame with columns: Symbol, Company, Event, CallTime, EPSEstimate, ReportedEPS, Surprise, MarketCap
    """
    url = f"https://finance.yahoo.com/calendar/earnings?day={date_str}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://finance.yahoo.com/'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find table
            table = soup.find('table', {'class': 'yf-1uayyp1'})

            if not table:
                print("Table not found!")
                # Print page structure for debugging
                print("\nPage title:", soup.find('title'))
                print("\nTables found:", len(soup.find_all('table')))

                # Save HTML for inspection
                with open('/Users/lynn/cps-data-monitor_F/yahoo_earnings_page.html', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print("HTML saved to yahoo_earnings_page.html")
                return pd.DataFrame()

            # Parse table rows
            rows = table.find('tbody').find_all('tr')
            print(f"Rows found: {len(rows)}")

            data = []
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 8:
                    symbol = cells[0].get_text(strip=True)
                    company = cells[1].get_text(strip=True)
                    event = cells[2].get_text(strip=True)
                    call_time = cells[3].get_text(strip=True)
                    eps_estimate = cells[4].get_text(strip=True)
                    reported_eps = cells[5].get_text(strip=True)
                    surprise = cells[6].get_text(strip=True)
                    market_cap = cells[7].get_text(strip=True)

                    data.append({
                        'Symbol': symbol,
                        'Company': company,
                        'Event': event,
                        'CallTime': call_time,
                        'EPSEstimate': eps_estimate,
                        'ReportedEPS': reported_eps,
                        'Surprise': surprise,
                        'MarketCap': market_cap
                    })

            df = pd.DataFrame(data)
            return df

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    return pd.DataFrame()


if __name__ == '__main__':
    print("=" * 60)
    print("Yahoo Finance Earnings Calendar 크롤링 테스트")
    print("=" * 60)

    df = fetch_yahoo_earnings_calendar("2026-01-29")

    if not df.empty:
        print(f"\n✓ {len(df)}개 종목 데이터 수집 성공!")
        print("\n" + "=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)

        # Save to JSON
        df.to_json('/Users/lynn/cps-data-monitor_F/yahoo_earnings_test.json', orient='records', indent=2)
        print("\nJSON 저장 완료: yahoo_earnings_test.json")
    else:
        print("\n✗ 데이터 수집 실패")
