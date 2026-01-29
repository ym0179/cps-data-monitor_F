"""
Active ETF Portfolio Monitor
타임폴리오 Active ETF의 구성종목(PDF) 변화를 모니터링하는 모듈
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from typing import Dict, List, Tuple
import re
import yfinance as yf
import pytz


class ActiveETFMonitor:
    """Active ETF 포트폴리오 모니터링 클래스"""

    BASE_URL = "https://timefolioetf.co.kr/m11_view.php"
    KST = pytz.timezone('Asia/Seoul')  # 한국 표준시

    # ISIN 코드 → yfinance 티커 매핑 테이블
    ISIN_TO_TICKER = {
        'CA13321L1085': 'CCJ',  # Cameco Corp
        # 필요시 추가 매핑 추가
    }

    def __init__(self, data_dir: str = "./data", url: str = None, etf_name: str = None):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
            url: ETF URL (예: https://timefolioetf.co.kr/m11_view.php?idx=2)
                 None이면 기본값 (idx=5) 사용
            etf_name: ETF 이름 (예: "Active ETF", "Value ETF")
                      None이면 기본값 사용
        """
        # URL에서 idx 추출 (먼저 수행)
        if url:
            # URL 파싱해서 idx 추출
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            self.idx = query_params.get('idx', ['5'])[0]
        else:
            self.idx = '5'  # 기본값

        # ETF별로 데이터 디렉토리 분리
        # 예: ./data/idx_5/, ./data/idx_2/
        self.data_dir = os.path.join(data_dir, f"idx_{self.idx}")
        os.makedirs(self.data_dir, exist_ok=True)

        # ETF 이름 설정
        self.etf_name = etf_name if etf_name else 'Active ETF'

    def get_portfolio_data(self, date: str = None) -> pd.DataFrame:
        """
        특정 날짜의 포트폴리오 데이터를 크롤링

        Args:
            date: 조회할 날짜 (YYYY-MM-DD), None이면 오늘 날짜

        Returns:
            DataFrame: 종목코드, 종목명, 수량, 평가금액, 비중
        """
        if date is None:
            date = datetime.now(self.KST).strftime("%Y-%m-%d")

        # URL 파라미터 구성
        params = {
            'idx': self.idx,
            'cate': '',
            'pdfDate': date
        }

        try:
            # 브라우저 헤더 추가
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Referer': 'https://timefolioetf.co.kr/'
            }

            # HTTP 요청
            response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'

            # HTML 파싱
            soup = BeautifulSoup(response.text, 'html.parser')

            # 구성종목 테이블 찾기
            table = soup.find('table', class_='table3')
            if not table:
                raise ValueError(f"테이블을 찾을 수 없습니다. (날짜: {date})")

            # 데이터 추출 (display:none인 행도 모두 포함)
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

            df = pd.DataFrame(data)
            df['날짜'] = date

            print(f"✓ {date} 데이터 수집 완료: {len(df)}개 종목")
            return df

        except requests.RequestException as e:
            print(f"✗ 네트워크 오류: {e}")
            raise
        except Exception as e:
            print(f"✗ 데이터 수집 오류: {e}")
            raise

    def save_data(self, df: pd.DataFrame, date: str):
        """데이터를 JSON 파일로 저장"""
        filename = os.path.join(self.data_dir, f"portfolio_{date}.json")
        df.to_json(filename, orient='records', force_ascii=False, indent=2)
        print(f"✓ 데이터 저장 완료: {filename}")

    def load_data(self, date: str) -> pd.DataFrame:
        """저장된 데이터 로드"""
        filename = os.path.join(self.data_dir, f"portfolio_{date}.json")
        if os.path.exists(filename):
            return pd.read_json(filename)
        return None

    def get_previous_business_day(self, date: str, lookback_days: int = 10) -> str:
        """
        이전 영업일 찾기 (데이터가 있는 날짜 기준)

        Args:
            date: 기준 날짜
            lookback_days: 최대 조회 일수

        Returns:
            이전 영업일 (YYYY-MM-DD)
        """
        current_date = datetime.strptime(date, "%Y-%m-%d")

        for i in range(1, lookback_days + 1):
            prev_date = current_date - timedelta(days=i)
            prev_date_str = prev_date.strftime("%Y-%m-%d")

            # 저장된 데이터가 있는지 확인
            if self.load_data(prev_date_str) is not None:
                return prev_date_str

            # 없으면 크롤링 시도
            try:
                df = self.get_portfolio_data(prev_date_str)
                if len(df) > 0:
                    self.save_data(df, prev_date_str)
                    return prev_date_str
            except:
                continue

        raise ValueError(f"{date}로부터 {lookback_days}일 이내에 이전 영업일을 찾을 수 없습니다.")

    def _ticker_from_code(self, code: str) -> str:
        """
        종목코드를 yfinance 티커로 변환

        Args:
            code: PDF 종목코드 (예: "NVDA US EQUITY", "ESZ5 Index", "BRK/B US EQUITY", "CA13321L1085")

        Returns:
            yfinance 티커 (예: "NVDA", "BRK-B", "^GSPC", "CCJ")
        """
        # 공백과 접미사 제거
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
        yfinance로 각 종목의 시장 수익률 가져오기

        Args:
            df_prev: 전일 포트폴리오
            df_today: 금일 포트폴리오 (PDF fallback용)
            date_prev: 전일 날짜
            date_today: 금일 날짜

        Returns:
            {종목코드: 수익률} 딕셔너리
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

            # 티커 변환 실패 (ISIN, 지원안하는 선물 등)
            if not ticker_symbol:
                # PDF 데이터로 fallback
                try:
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['수량'] > 0 and today_row.iloc[0]['수량'] > 0:
                        prev_price = row['평가금액'] / row['수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"ℹ️  {code[:20]} ({stock_name}): yfinance 미지원, PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                        print(f"ℹ️  {code[:20]} ({stock_name}): yfinance 미지원, 0% 사용")
                except Exception as e:
                    market_returns[code] = 0.0
                    print(f"⚠️  {code[:20]} ({stock_name}): PDF fallback 실패 - {type(e).__name__}: {str(e)[:50]}")
                continue

            try:
                # yfinance로 데이터 가져오기
                # 미국 장 기준: 항상 최신 2개 영업일 (D-1, D-2) 사용
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period="5d")  # 최근 5일 데이터 (영업일 2일 확보)

                if len(hist) < 2:
                    # 데이터 부족 - PDF 데이터로 fallback
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['수량'] > 0 and today_row.iloc[0]['수량'] > 0:
                        prev_price = row['평가금액'] / row['수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"ℹ️  {ticker_symbol} ({stock_name}): yfinance 데이터 부족, PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                        print(f"⚠️  {ticker_symbol} ({stock_name}): yfinance 데이터 부족, 0% 사용")
                    continue

                # 최신 2개 영업일 사용 (D-1, D-2)
                prev_close = hist.iloc[-2]['Close']   # D-2 (전전일 종가)
                today_close = hist.iloc[-1]['Close']  # D-1 (전일 종가)
                prev_date_used = hist.iloc[-2].name.strftime('%Y-%m-%d')
                today_date_used = hist.iloc[-1].name.strftime('%Y-%m-%d')

                # 수익률 계산
                market_return = (today_close / prev_close - 1) if prev_close > 0 else 0.0
                market_returns[code] = market_return
                print(f"✓ {ticker_symbol} ({stock_name}): {market_return*100:+.2f}% ({prev_date_used} → {today_date_used})")

            except Exception as e:
                # 오류 발생 시 PDF 데이터로 fallback
                error_type = type(e).__name__
                error_msg = str(e)[:100]

                try:
                    today_row = df_today[df_today['종목코드'] == code]
                    if len(today_row) > 0 and row['수량'] > 0 and today_row.iloc[0]['수량'] > 0:
                        prev_price = row['평가금액'] / row['수량']
                        today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['수량']
                        pdf_return = (today_price / prev_price - 1) if prev_price > 0 else 0
                        market_returns[code] = pdf_return
                        print(f"⚠️  {ticker_symbol} ({stock_name}): yfinance 오류 ({error_type}: {error_msg}), PDF 가격 사용 ({pdf_return*100:.2f}%)")
                    else:
                        market_returns[code] = 0.0
                        print(f"⚠️  {ticker_symbol} ({stock_name}): yfinance 오류 ({error_type}: {error_msg}), 0% 사용")
                except:
                    market_returns[code] = 0.0
                    print(f"⚠️  {ticker_symbol} ({stock_name}): yfinance 및 PDF fallback 실패 ({error_type}), 0% 사용")

        return market_returns

    def analyze_rebalancing(self, df_today: pd.DataFrame, df_prev: pd.DataFrame,
                           date_prev: str = None, date_today: str = None) -> Dict:
        """
        리밸런싱 분석 (시장 수익률 기반)

        시장 가격 변동만으로 설명되지 않는 비중 변화를 리밸런싱으로 감지
        AUM 변화와 가격 변동 효과를 모두 제거

        Args:
            df_today: 금일 포트폴리오
            df_prev: 전일 포트폴리오

        Returns:
            분석 결과 딕셔너리
        """
        # 종목코드를 기준으로 병합 (양쪽 모두 종목명 포함)
        merged = pd.merge(
            df_today[['종목코드', '종목명', '수량', '평가금액', '비중']],
            df_prev[['종목코드', '종목명', '수량', '평가금액', '비중']],
            on='종목코드',
            how='outer',
            suffixes=('_today', '_prev')
        )

        # 종목명 통합 (금일 우선, 없으면 전일 사용)
        merged['종목명'] = merged['종목명_today'].fillna(merged['종목명_prev'])

        # 숫자 컬럼만 0으로 채우기
        numeric_columns = ['수량_today', '수량_prev', '평가금액_today', '평가금액_prev', '비중_today', '비중_prev']
        merged[numeric_columns] = merged[numeric_columns].fillna(0)

        # 1단계: yfinance로 시장 수익률 가져오기
        if date_prev and date_today:
            market_returns = self.get_market_returns(df_prev, df_today, date_prev, date_today)
        else:
            # 날짜 정보가 없으면 PDF 데이터로 fallback
            print(f"⚠️  날짜 정보 없음, PDF 데이터로 수익률 계산")
            market_returns = {}
            for _, row in df_prev.iterrows():
                code = row['종목코드']
                # PDF 데이터로 수익률 계산
                prev_price = row['평가금액'] / row['수량'] if row['수량'] > 0 else 0
                today_row = df_today[df_today['종목코드'] == code]
                if len(today_row) > 0:
                    today_price = today_row.iloc[0]['평가금액'] / today_row.iloc[0]['수량'] if today_row.iloc[0]['수량'] > 0 else 0
                    market_returns[code] = (today_price / prev_price - 1) if prev_price > 0 else 0
                else:
                    market_returns[code] = 0

        # 시장 수익률을 merged에 추가
        merged['시장_수익률'] = merged['종목코드'].map(market_returns).fillna(0)

        # 2단계: 가상 비중 계산 (시장 변동만 반영)
        merged['가상_비중'] = merged['비중_prev'] * (1 + merged['시장_수익률'])

        # 3단계: 정규화 (100%로 스케일링)
        total_virtual_weight = merged['가상_비중'].sum()
        if total_virtual_weight > 0:
            merged['예상_비중'] = merged['가상_비중'] / total_virtual_weight * 100
        else:
            merged['예상_비중'] = 0

        # 4단계: 실제 비중 변화 vs 예상 비중 변화
        merged['순수_비중변화'] = merged['비중_today'] - merged['예상_비중']

        # 5단계: 수량 변화 확인
        merged['수량_변화'] = merged['수량_today'] - merged['수량_prev']

        # 리밸런싱 감지
        # - 의미있는 비중 변화 (±0.5%p 이상)
        # - 또는 편입/편출 (수량이 0에서 변화)
        # - 현금 제외
        threshold = 0.5
        rebalanced = merged[
            ((abs(merged['순수_비중변화']) >= threshold) |
             (merged['수량_prev'] == 0) |
             (merged['수량_today'] == 0)) &
            (merged['종목명'] != '현금')
        ].copy()

        # 편입/편출/비중확대/비중축소 구분
        new_stocks = rebalanced[(rebalanced['수량_prev'] == 0) & (rebalanced['수량_today'] > 0)]  # 신규 편입
        removed_stocks = rebalanced[(rebalanced['수량_today'] == 0) & (rebalanced['수량_prev'] > 0)]  # 편출

        # 비중 확대/축소는 순수 비중 변화 + 수량 변화 모두 체크
        # 수량이 증가했고, 비중도 의미있게 증가한 경우만
        increased_stocks = rebalanced[(rebalanced['순수_비중변화'] > threshold) &
                                     (rebalanced['수량_변화'] > 0) &
                                     (rebalanced['수량_prev'] > 0) &
                                     (rebalanced['수량_today'] > 0)]  # 비중 확대
        decreased_stocks = rebalanced[(rebalanced['순수_비중변화'] < -threshold) &
                                     (rebalanced['수량_변화'] < 0) &
                                     (rebalanced['수량_prev'] > 0) &
                                     (rebalanced['수량_today'] > 0)]  # 비중 축소

        # 주식 비중 계산 (현금 제외)
        stock_weight_prev = df_prev[df_prev['종목명'] != '현금']['비중'].sum()
        stock_weight_today = df_today[df_today['종목명'] != '현금']['비중'].sum()

        return {
            'new_stocks': new_stocks.to_dict('records'),
            'removed_stocks': removed_stocks.to_dict('records'),
            'increased_stocks': increased_stocks.to_dict('records'),
            'decreased_stocks': decreased_stocks.to_dict('records'),
            'total_changes': len(rebalanced),
            'stock_weight_prev': stock_weight_prev,
            'stock_weight_today': stock_weight_today,
        }

    def format_summary(self, analysis: Dict, df_today: pd.DataFrame,
                      date_today: str, date_prev: str) -> str:
        """
        분석 결과를 가독성 있게 포맷팅 (간결한 요약 형식)

        Args:
            analysis: analyze_rebalancing 결과
            df_today: 금일 포트폴리오
            date_today: 금일 날짜
            date_prev: 전일 날짜

        Returns:
            포맷된 텍스트
        """
        lines = []

        # ETF 이름
        lines.append(f"<b>{self.etf_name}</b>")
        lines.append("")

        # 기준일
        lines.append(f"• <b>기준일: {date_today} (vs {date_prev})</b>")
        lines.append("")

        # 리밸런싱 요약
        lines.append("• <b>리밸런싱 요약(±0.5%p 이상*만 표시)</b>:")
        lines.append(f"편입 {len(analysis['new_stocks'])}개, "
                    f"편출 {len(analysis['removed_stocks'])}개, "
                    f"비중 확대 {len(analysis['increased_stocks'])}개, "
                    f"비중 축소 {len(analysis['decreased_stocks'])}개")
        lines.append("")

        # 편입 종목
        if analysis['new_stocks']:
            for stock in analysis['new_stocks']:
                code = stock['종목코드'].replace(' US EQUITY', '').replace(' Index', '').strip()
                lines.append(f"- {stock['종목명']}({code}) 편입 "
                            f"(0.0 ▶ {stock['비중_today']:.1f}%) "
                            f"{stock['순수_비중변화']:+.1f}")
            lines.append("")

        # 편출 종목
        if analysis['removed_stocks']:
            for stock in analysis['removed_stocks']:
                code = stock['종목코드'].replace(' US EQUITY', '').replace(' Index', '').strip()
                lines.append(f"- {stock['종목명']}({code}) 편출 "
                            f"({stock['비중_prev']:.1f} ▶ 0.0%) "
                            f"{stock['순수_비중변화']:+.1f}")
            lines.append("")

        # 비중 확대 종목
        if analysis['increased_stocks']:
            for stock in analysis['increased_stocks']:
                code = stock['종목코드'].replace(' US EQUITY', '').replace(' Index', '').strip()
                lines.append(f"- {stock['종목명']}({code}) 비중 확대 "
                            f"({stock['비중_prev']:.1f} ▶ {stock['비중_today']:.1f}%) "
                            f"{stock['순수_비중변화']:+.1f}")
            lines.append("")

        # 비중 축소 종목
        if analysis['decreased_stocks']:
            for stock in analysis['decreased_stocks']:
                code = stock['종목코드'].replace(' US EQUITY', '').replace(' Index', '').strip()
                lines.append(f"- {stock['종목명']}({code}) 비중 축소 "
                            f"({stock['비중_prev']:.1f} ▶ {stock['비중_today']:.1f}%) "
                            f"{stock['순수_비중변화']:+.1f}")
            lines.append("")

        # 리밸런싱 후 주식비중 변화
        # stock_prev = analysis['stock_weight_prev']
        # stock_today = analysis['stock_weight_today']
        # stock_change = stock_today - stock_prev
        # lines.append(f"• 리밸런싱 후 주식비중 변화 ({stock_prev:.1f} ▶ {stock_today:.1f}%)   {stock_change:+.1f}")
        # lines.append("")

        # 구성종목 Top 10
        lines.append("• <b>구성종목 Top 10</b>:")
        top10 = df_today.nlargest(10, '비중')
        for idx, row in enumerate(top10.itertuples(), 1):
            lines.append(f"{idx}. {row.종목명}  {row.비중:.2f}%")
        lines.append("")

        # 주석
        lines.append("* 수량이 변한 종목 중, 가격효과(전일 비중×(1+시장수익률))를 제거한 예상비중 대비 실제비중의 차이")

        return "\n".join(lines)


if __name__ == "__main__":
    # 테스트 코드
    monitor = ActiveETFMonitor()

    # 오늘 데이터 수집
    today = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d")
    df_today = monitor.get_portfolio_data(today)
    monitor.save_data(df_today, today)

    # 전일 데이터 가져오기
    try:
        prev_day = monitor.get_previous_business_day(today)
        df_prev = monitor.load_data(prev_day)

        # 리밸런싱 분석
        analysis = monitor.analyze_rebalancing(df_today, df_prev)

        # 결과 출력
        summary = monitor.format_summary(analysis, df_today, today, prev_day)
        print(summary)
    except Exception as e:
        print(f"전일 데이터를 찾을 수 없습니다: {e}")
        print(f"\n금일 포트폴리오 ({today}):")
        print(df_today.to_string())
