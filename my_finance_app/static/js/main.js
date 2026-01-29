/**
 * MAS Finance - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('MAS Finance App Loaded');

    // ============================================
    // 프로모션 카드 닫기
    // ============================================
    const promoClose = document.querySelector('.promo-close');
    if (promoClose) {
        promoClose.addEventListener('click', function() {
            const promoCard = this.closest('.promo-card');
            promoCard.style.display = 'none';
            // 로컬스토리지에 저장 (다시 안보이게)
            localStorage.setItem('promo_closed', 'true');
        });

        // 이미 닫은 경우 숨기기
        if (localStorage.getItem('promo_closed') === 'true') {
            document.querySelector('.promo-card').style.display = 'none';
        }
    }

    // ============================================
    // 시장 데이터 자동 갱신 (5분마다)
    // ============================================
    function updateMarketData() {
        fetch('/api/market-data')
            .then(response => response.json())
            .then(data => {
                console.log('Market data updated:', data);
                // TODO: DOM 업데이트 로직 추가
            })
            .catch(error => {
                console.error('Failed to update market data:', error);
            });
    }

    // 5분마다 갱신
    // setInterval(updateMarketData, 5 * 60 * 1000);

    // ============================================
    // 검색 자동완성 (추후 구현)
    // ============================================
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        let debounceTimer;
        searchInput.addEventListener('input', function() {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                const query = this.value.trim();
                if (query.length >= 2) {
                    // TODO: 자동완성 API 호출
                    console.log('Search query:', query);
                }
            }, 300);
        });
    }

    // ============================================
    // 모바일 터치 피드백
    // ============================================
    const touchElements = document.querySelectorAll('.index-card, .feature-card, .bottom-nav-item');
    touchElements.forEach(el => {
        el.addEventListener('touchstart', function() {
            this.style.opacity = '0.7';
        });
        el.addEventListener('touchend', function() {
            this.style.opacity = '1';
        });
    });
});
