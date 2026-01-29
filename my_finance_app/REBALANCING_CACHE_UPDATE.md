# Active ETF 리밸런싱 캐시 최적화

## 🎯 문제 해결

### 기존 문제
- Active ETF 페이지 로딩 시간이 **2-3분** 소요
- 원인: yfinance API를 사용한 실시간 주가 조회 (50+ 종목)
- 포트폴리오 데이터는 캐시되었지만, **리밸런싱 분석은 매번 실시간 계산**

### 해결 방법
- 리밸런싱 결과도 **캐시로 저장**
- 스케줄러가 매일 아침 리밸런싱 결과 미리 계산
- API는 캐시된 결과 반환 → **로딩 시간 1초 이내**

---

## 📁 변경된 파일

### 1. `update_cache.py`
리밸런싱 결과를 캐시에 저장하도록 수정

**변경 내용:**
```python
def update_active_etf_cache():
    """Active ETF 최근 7일 데이터 캐시 갱신 (포트폴리오 + 리밸런싱 결과)"""

    # 1. 포트폴리오 데이터 수집 (기존과 동일)
    for date_str in dates_to_fetch:
        df = monitor.get_portfolio_data(date_str)
        # data/time_etf/etf_5/portfolio_2026-01-30.json 저장

    # 2. 리밸런싱 결과 계산 및 캐시 (새로 추가)
    for i in range(len(valid_dates) - 1):
        current_date = valid_dates[i]
        previous_date = valid_dates[i + 1]

        # 리밸런싱 분석 (yfinance 호출)
        rebalancing_result = monitor.analyze_rebalancing(current_date, previous_date)

        # 캐시 저장
        cache_filename = f"{etf_id}_{current_date}_vs_{previous_date}.json"
        cache_filepath = "cache/rebalancing/{cache_filename}"
        json.dump(rebalancing_result, f)
```

**캐시 파일 위치:**
```
my_finance_app/
├── cache/
│   ├── search_engine.json
│   ├── os_market_share.json
│   └── rebalancing/                    # 새로 생성
│       ├── time_sp500_2026-01-30_vs_2026-01-29.json
│       ├── time_sp500_2026-01-29_vs_2026-01-28.json
│       ├── time_nasdaq100_2026-01-30_vs_2026-01-29.json
│       └── kiwoom_growth30_2026-01-30_vs_2026-01-29.json
```

### 2. `app.py`
API 엔드포인트가 캐시 우선 사용하도록 수정

**변경 내용:**
```python
@app.route('/api/etf/data/<etf_id>')
def api_etf_data(etf_id):
    # 기존: 무조건 실시간 계산 (느림)
    # rebalancing = monitor.analyze_rebalancing(df_today, df_prev, date_str, prev_date)

    # 신규: 캐시 우선, 없으면 실시간 계산
    rebalancing = None
    if prev_date:
        # 1. 캐시 확인
        cache_filename = f"{etf_id}_{date_str}_vs_{prev_date}.json"
        cache_filepath = os.path.join(CACHE_DIR, 'rebalancing', cache_filename)

        if os.path.exists(cache_filepath):
            # 캐시 사용 (빠름 - 1초 이내)
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                rebalancing = json.load(f)
            print(f"✓ Rebalancing cache hit: {cache_filename}")
        else:
            # 캐시 없으면 실시간 계산 (느림 - 2-3분)
            print(f"⚠ Rebalancing cache miss, computing real-time (slow)...")
            rebalancing = monitor.analyze_rebalancing(df_today, df_prev, date_str, prev_date)
```

---

## 🚀 배포 및 테스트 방법

### 1. PythonAnywhere에 파일 업로드

```bash
# 로컬에서 변경사항 커밋 및 푸시
git add my_finance_app/update_cache.py my_finance_app/app.py
git commit -m "Add rebalancing result caching for faster Active ETF loading"
git push
```

```bash
# PythonAnywhere Bash Console에서
cd ~/cps-data-monitor_F
git pull
```

### 2. 캐시 수동 생성 (첫 배포 시)

```bash
# PythonAnywhere Bash Console에서
cd ~/cps-data-monitor_F/my_finance_app
python3 update_cache.py
```

**예상 실행 시간:** 약 **10-15분** (리밸런싱 계산 포함)

**예상 출력:**
```
============================================================
캐시 갱신 스크립트 시작: 2026-01-30 05:45:00
============================================================

[2026-01-30 05:45:00] Search Engine 캐시 갱신 시작...
  - desktop_mobile 데이터 수집 중...
    완료: 85개월 데이터
  ...
[2026-01-30 05:45:10] 캐시 저장 완료: .../cache/search_engine.json

[2026-01-30 05:45:10] OS Market Share 캐시 갱신 시작...
  ...
[2026-01-30 05:45:20] 캐시 저장 완료: .../cache/os_market_share.json

[2026-01-30 05:45:20] Active ETF 캐시 갱신 시작...
  조회 날짜: ['2026-01-30', '2026-01-29', '2026-01-28', ...]

  [TIME S&P500] 데이터 수집 중...
    ✓ 2026-01-30: 52개 종목
    ✓ 2026-01-29: 52개 종목
    ...
    포트폴리오 수집 완료: 7/7일

    리밸런싱 결과 캐시 생성 중...
      ✓ 2026-01-30 vs 2026-01-29
      ✓ 2026-01-29 vs 2026-01-28
      ...
    리밸런싱 캐시 완료: 6개

  [TIME NASDAQ100] 데이터 수집 중...
    ...
    리밸런싱 캐시 완료: 6개

  [KIWOOM 미국성장기업30액티브] 데이터 수집 중...
    ...
    리밸런싱 캐시 완료: 6개

[2026-01-30 05:58:00] Active ETF 캐시 갱신 완료
  포트폴리오 저장 위치: .../data
  리밸런싱 캐시 위치: .../cache/rebalancing

============================================================
캐시 갱신 완료: 2026-01-30 05:58:00
============================================================
```

### 3. 웹 앱 재시작

```
PythonAnywhere > Web > Reload
```

### 4. 성능 테스트

#### **Before (캐시 적용 전)**
1. Active ETF 페이지 접속
2. 날짜 선택 (예: 2026-01-30)
3. ETF 선택 (예: TIME S&P500)
4. ⏱️ **로딩 시간: 2-3분**

#### **After (캐시 적용 후)**
1. Active ETF 페이지 접속
2. 날짜 선택 (예: 2026-01-30)
3. ETF 선택 (예: TIME S&P500)
4. ⚡ **로딩 시간: 1초 이내**

---

## 📊 캐시 구조

### 포트폴리오 캐시 (기존)
```
data/
├── time_etf/
│   ├── etf_5/                          # TIME S&P500
│   │   ├── portfolio_2026-01-30.json  # 52개 종목, 비중
│   │   ├── portfolio_2026-01-29.json
│   │   └── ...
│   └── etf_2/                          # TIME NASDAQ100
│       └── ...
└── kiwoom_etf/
    ├── portfolio_2026-01-30.json       # 30개 종목, 비중
    └── ...
```

**용량:** 약 100-200KB per file

### 리밸런싱 캐시 (신규)
```
cache/
└── rebalancing/
    ├── time_sp500_2026-01-30_vs_2026-01-29.json
    ├── time_sp500_2026-01-29_vs_2026-01-28.json
    ├── time_nasdaq100_2026-01-30_vs_2026-01-29.json
    └── kiwoom_growth30_2026-01-30_vs_2026-01-29.json
```

**파일 내용 예시:**
```json
{
  "new_stocks": [
    {
      "ticker": "NVDA",
      "name": "NVIDIA Corp",
      "after_weight": 5.2,
      "market_return": 2.5
    }
  ],
  "removed_stocks": [...],
  "increased_stocks": [...],
  "decreased_stocks": [...]
}
```

**용량:** 약 50-100KB per file

**총 캐시 크기 (7일치):**
- 포트폴리오: ~2MB
- 리밸런싱: ~2MB
- **합계: ~4MB**

---

## 🔄 스케줄러 동작

### 기존 스케줄 (변경 없음)
- **시간:** UTC 20:45 (한국시간 05:45)
- **Command:** `python3 /home/ym96/cps-data-monitor_F/my_finance_app/update_cache.py`

### 실행 내용 (업데이트)
1. ✅ Search Engine 캐시 갱신 (~10초)
2. ✅ OS Market Share 캐시 갱신 (~10초)
3. ✅ Active ETF 포트폴리오 캐시 갱신 (~1분)
4. ✨ **Active ETF 리밸런싱 캐시 생성 (~10-12분)** ← 새로 추가

**총 실행 시간:** 약 **12-13분**

---

## 🛠️ 트러블슈팅

### 문제: 캐시 생성 중 오류 발생

**로그 확인:**
```bash
# PythonAnywhere Tasks 페이지에서 "Show output" 클릭
```

**일반적인 오류:**

#### 1. yfinance 오류
```
✗ 2026-01-30 vs 2026-01-29: 오류 - No data found
```

**원인:** yfinance API 일시적 오류 또는 Rate Limit

**해결:**
- 정상적인 현상 (일부 날짜는 실패할 수 있음)
- 다음 스케줄 실행 시 재시도됨

#### 2. 메모리 부족
```
MemoryError: Unable to allocate array
```

**해결:**
- PythonAnywhere 무료 계정 제한 (512MB)
- ETF 개수 줄이기 또는 유료 계정 업그레이드

### 문제: 캐시가 있는데도 느림

**확인 사항:**
1. 캐시 파일 존재 확인
   ```bash
   ls -lh ~/cps-data-monitor_F/my_finance_app/cache/rebalancing/
   ```

2. 로그 확인
   - "✓ Rebalancing cache hit" → 캐시 사용 (정상)
   - "⚠ Rebalancing cache miss" → 캐시 없음 (느림)

3. 날짜 파라미터 확인
   - 캐시는 최근 7일만 저장
   - 7일 이전 날짜 조회 시 실시간 계산

---

## 💡 추가 최적화 옵션

### 옵션 1: 캐시 기간 연장
현재 7일 → 30일로 확장

```python
# update_cache.py
dates_to_fetch = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
```

**장단점:**
- ✅ 더 많은 과거 데이터 빠르게 조회 가능
- ❌ 캐시 크기 증가 (~15MB)
- ❌ 스케줄러 실행 시간 증가 (~40분)

### 옵션 2: 선택적 ETF 캐시
자주 조회하는 ETF만 캐시

```python
# update_cache.py
etf_configs = [
    {'id': 'time_sp500', 'name': 'TIME S&P500', 'type': 'time', 'idx': '5'},
    # 나머지 ETF는 주석 처리
]
```

**장단점:**
- ✅ 스케줄러 실행 시간 단축
- ❌ 일부 ETF는 여전히 느림

---

## 📞 문의

문제가 발생하면:
1. Tasks 페이지에서 로그 확인
2. Bash Console에서 수동 실행으로 디버깅
3. 캐시 파일 직접 확인

**캐시 파일 확인 명령어:**
```bash
# 리밸런싱 캐시 목록
ls -lh ~/cps-data-monitor_F/my_finance_app/cache/rebalancing/

# 캐시 파일 내용 확인
cat ~/cps-data-monitor_F/my_finance_app/cache/rebalancing/time_sp500_2026-01-30_vs_2026-01-29.json | python3 -m json.tool
```
