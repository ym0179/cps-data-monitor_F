# PythonAnywhere 스케줄러 설정 가이드

이 문서는 매일 아침 자동으로 캐시를 갱신하는 스케줄러 설정 방법을 설명합니다.

## 📋 캐시 갱신 내용

`update_cache.py` 스크립트는 다음 데이터를 자동으로 갱신합니다:

1. **Search Engine Market Share** (2019-01 ~ 현재)
   - Desktop+Mobile, Desktop, Mobile 각각
   - 캐시 파일: `cache/search_engine.json`

2. **OS Market Share** (2019-01 ~ 현재)
   - Desktop, Mobile, Tablet 각각
   - 캐시 파일: `cache/os_market_share.json`

3. **Active ETF 포트폴리오** (최근 7일)
   - TIME S&P500, TIME NASDAQ100, KIWOOM 미국성장기업30액티브
   - 캐시 파일: `data/time_etf/etf_*/portfolio_*.json`, `data/kiwoom_etf/portfolio_*.json`

---

## 🚀 PythonAnywhere 스케줄러 설정 방법

### 1. PythonAnywhere 웹사이트 접속

1. [PythonAnywhere](https://www.pythonanywhere.com) 로그인
2. 대시보드로 이동

### 2. Tasks 메뉴 열기

1. 상단 메뉴에서 **"Tasks"** 클릭
2. "Scheduled tasks" 섹션으로 이동

### 3. 새 스케줄 작업 추가

#### 설정 값:

| 항목 | 값 |
|------|-----|
| **Description** | Daily cache update (Search Engine, OS, Active ETF) |
| **Command** | `python3 /home/ym96/cps-data-monitor_F/my_finance_app/update_cache.py` |
| **Hour** | `00` (UTC) |
| **Minute** | `30` |

> **중요**: PythonAnywhere는 UTC 시간을 사용합니다!
> - UTC 00:30 = 한국시간 09:30 (오전)
> - UTC 09:00 = 한국시간 18:00 (저녁)

#### 추천 실행 시간:
- **UTC 00:30** (한국시간 오전 9시 30분) - 미국 시장 데이터가 업데이트된 후 실행

### 4. 저장하기

1. "Create" 버튼 클릭
2. 작업이 목록에 추가되었는지 확인

---

## ⏰ 시간대 변환 참고

| UTC 시간 | 한국 시간 (KST) | 설명 |
|----------|-----------------|------|
| 00:00 | 오전 09:00 | 출근 시간 |
| 00:30 | 오전 09:30 | **추천** |
| 01:00 | 오전 10:00 | 업무 시작 |
| 09:00 | 저녁 18:00 | 퇴근 시간 |
| 15:00 | 자정 00:00 | 심야 |

---

## 🔍 스케줄러 동작 확인

### 1. 수동 실행 테스트

스케줄러 설정 전에 먼저 수동으로 테스트해보세요:

```bash
# PythonAnywhere Bash Console에서 실행
cd ~/cps-data-monitor_F/my_finance_app
python3 update_cache.py
```

### 2. 로그 확인

스케줄 작업이 실행되면 PythonAnywhere Tasks 페이지에서 로그를 확인할 수 있습니다:

1. Tasks 페이지로 이동
2. 해당 작업의 "Show output" 링크 클릭
3. 실행 로그 확인

**예상 출력:**
```
============================================================
캐시 갱신 스크립트 시작: 2026-01-30 00:30:15.123456
============================================================
[2026-01-30 00:30:15] Search Engine 캐시 갱신 시작...
  - desktop_mobile 데이터 수집 중...
    완료: 85개월 데이터 (2019-01 이후)
  - desktop 데이터 수집 중...
    완료: 85개월 데이터 (2019-01 이후)
  - mobile 데이터 수집 중...
    완료: 85개월 데이터 (2019-01 이후)
[2026-01-30 00:30:25] 캐시 저장 완료: .../cache/search_engine.json

[2026-01-30 00:30:25] OS Market Share 캐시 갱신 시작...
  - desktop 데이터 수집 중...
    완료: 85개월 데이터 (2019-01 이후)
  - mobile 데이터 수집 중...
    완료: 85개월 데이터 (2019-01 이후)
  - tablet 데이터 수집 중...
    완료: 85개월 데이터 (2019-01 이후)
[2026-01-30 00:30:35] 캐시 저장 완료: .../cache/os_market_share.json

[2026-01-30 00:30:35] Active ETF 캐시 갱신 시작...
  조회 날짜: ['2026-01-30', '2026-01-29', '2026-01-28', ...]

  [TIME S&P500] 데이터 수집 중...
    ✓ 2026-01-30: 52개 종목
    ✓ 2026-01-29: 52개 종목
    ...
    완료: 7/7일 데이터 수집

  [TIME NASDAQ100] 데이터 수집 중...
    ✓ 2026-01-30: 48개 종목
    ...
    완료: 7/7일 데이터 수집

  [KIWOOM 미국성장기업30액티브] 데이터 수집 중...
    ✓ 2026-01-30: 30개 종목
    ...
    완료: 7/7일 데이터 수집

[2026-01-30 00:32:15] Active ETF 캐시 갱신 완료
  데이터 저장 위치: .../data

============================================================
캐시 갱신 완료: 2026-01-30 00:32:15.789012
============================================================
```

### 3. 웹사이트에서 확인

1. 대시보드 페이지 새로고침
2. Active ETF 페이지에서 최근 7일 데이터 조회 테스트
3. Search Engine, OS Market Share 페이지 정상 작동 확인

---

## 🛠️ 트러블슈팅

### 문제: 스케줄 작업이 실행되지 않음

**해결 방법:**
1. PythonAnywhere Tasks 페이지에서 작업이 "Enabled" 상태인지 확인
2. Command 경로가 올바른지 확인 (절대 경로 사용)
3. 다음 실행 예정 시간 확인

### 문제: 실행은 되지만 오류 발생

**해결 방법:**
1. Tasks 페이지에서 로그 확인
2. Bash Console에서 수동 실행하여 오류 메시지 확인:
   ```bash
   cd ~/cps-data-monitor_F/my_finance_app
   python3 update_cache.py
   ```
3. 의존성 패키지 설치 확인:
   ```bash
   pip3 install --user requests pandas beautifulsoup4 yfinance pytz urllib3
   ```

### 문제: Active ETF 데이터가 없음

**원인:** TIME ETF 또는 KIWOOM 웹사이트에서 데이터를 제공하지 않는 날짜

**해결 방법:**
- 정상적인 현상입니다 (주말, 공휴일 등)
- 이전 영업일 데이터가 자동으로 조회됩니다

---

## 📊 캐시 파일 구조

```
my_finance_app/
├── cache/
│   ├── search_engine.json       # Search Engine 데이터
│   └── os_market_share.json     # OS Market Share 데이터
├── data/
│   ├── time_etf/
│   │   ├── etf_5/                # TIME S&P500
│   │   │   ├── portfolio_2026-01-30.json
│   │   │   ├── portfolio_2026-01-29.json
│   │   │   └── ...
│   │   └── etf_2/                # TIME NASDAQ100
│   │       └── ...
│   └── kiwoom_etf/
│       ├── portfolio_2026-01-30.json
│       └── ...
└── update_cache.py
```

---

## 🔄 스케줄 작업 수정/삭제

### 수정:
1. Tasks 페이지에서 해당 작업 찾기
2. 시간 또는 Command 수정
3. "Update" 버튼 클릭

### 삭제:
1. Tasks 페이지에서 해당 작업 찾기
2. "Delete" 버튼 클릭
3. 확인

### 일시 중지:
1. Tasks 페이지에서 해당 작업 찾기
2. "Disable" 버튼 클릭
3. 나중에 "Enable"로 재활성화 가능

---

## 💡 추가 정보

- **무료 계정 제한**: PythonAnywhere 무료 계정은 1개의 스케줄 작업만 생성 가능
- **실행 시간**: 전체 캐시 갱신은 약 2-3분 소요
- **저장 공간**: 캐시 파일은 약 5-10MB 사용

---

## 📞 문의

문제가 발생하면 로그를 확인하고, 필요시 수동 실행으로 디버깅하세요.
