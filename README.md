# MAS Finance Dashboard

미래에셋 고객자산배분부 투자 분석 대시보드

## 개발 현황

| 버전 | 플랫폼 | 상태 | 경로 |
|------|--------|------|------|
| V1 (Streamlit) | Streamlit Cloud | 운영중 | `app.py` |
| V2 (Flask) | PythonAnywhere | 개발중 | `my_finance_app/` |

## 프로젝트 구조

```
├── app.py                  # Streamlit 앱 (기존)
├── etf.py                  # TIMEFOLIO ETF 크롤링
├── etf_kiwoom.py           # 키움 ETF API
├── logic_crawler.py        # 어닝 캘린더 크롤링
├── logic_earnings.py       # 컨센서스 분석
├── logic_idio.py           # 5-Factor 회귀분석 (Idio Score)
├── universe_stocks.csv     # 주식 유니버스 (129개)
├── universe_themes.csv     # 테마 ETF 유니버스 (97개)
│
└── my_finance_app/         # Flask 앱 (신규)
    ├── app.py              # Flask 메인
    ├── templates/          # HTML 템플릿
    ├── static/             # CSS, JS
    └── logic/              # 로직 모듈 (이동 예정)
```

## 주요 기능

| 기능 | 설명 | 모듈 |
|------|------|------|
| MS Monitoring | 검색엔진/OS 점유율 추적 (StatCounter) | `app.py` |
| Earnings Trading | GS 5-Factor Idio Score 알파 발굴 | `logic_idio.py` |
| Active ETF | TOP 1% 매니저 포트폴리오 추적 | `etf.py`, `etf_kiwoom.py` |

## 실행 방법

**Streamlit (기존)**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Flask (신규)**
```bash
cd my_finance_app
pip install -r requirements.txt
python app.py
```

## 배포

- **Streamlit**: Streamlit Cloud 연동
- **Flask**: PythonAnywhere (WSGI 설정 필요)

## TODO

- [ ] Flask 앱에 기존 로직 연결
- [ ] 실제 데이터 API 연동
- [ ] 사용자 인증 구현
- [ ] 스케줄러로 자동 데이터 수집
