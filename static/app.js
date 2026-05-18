document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('results-container');
    const loadBtn = document.getElementById('load-btn');
    const filterMatch = document.getElementById('filter-match');
    const filterNomatch = document.getElementById('filter-nomatch');
    const statsDisplay = document.getElementById('stats-display');

    let allResults = [];

    async function loadResults() {
        const originalText = loadBtn.innerText;
        loadBtn.innerText = '読み込み中...';
        loadBtn.disabled = true;
        
        try {
            // Cache buster to always get fresh results
            const response = await fetch('/api/results?t=' + new Date().getTime());
            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            allResults = data.results;
            render();
        } catch (error) {
            console.error('Error fetching results:', error);
            container.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px; color: #ef4444;">
                    <h3>エラーが発生しました</h3>
                    <p>バックエンドAPIに接続できません。FastAPIサーバーが起動しているか確認してください。</p>
                </div>
            `;
            statsDisplay.innerText = 'エラー';
        } finally {
            loadBtn.innerText = originalText;
            loadBtn.disabled = false;
        }
    }

    function render() {
        container.innerHTML = '';
        
        const showMatch = filterMatch.checked;
        const showNomatch = filterNomatch.checked;

        const filtered = allResults.filter(r => {
            if (r.is_match && showMatch) return true;
            if (!r.is_match && showNomatch) return true;
            return false;
        });

        // Update stats
        const matchCount = allResults.filter(r => r.is_match).length;
        statsDisplay.innerHTML = `全 <strong>${allResults.length}</strong> 件中: MATCH <strong>${matchCount}</strong> / NO MATCH <strong>${allResults.length - matchCount}</strong> (表示中: ${filtered.length}件)`;

        if (filtered.length === 0) {
            container.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #64748b; padding: 3rem;">表示する結果がありません。フィルターを変更するか、main.py を実行してください。</div>';
            return;
        }

        filtered.forEach(result => {
            const card = document.createElement('div');
            card.className = 'card';
            
            const statusClass = result.is_match ? 'match' : 'nomatch';
            const statusText = result.is_match ? 'MATCH' : 'NO MATCH';
            const simPercentage = (result.similarity_score * 100).toFixed(1);

            card.innerHTML = `
                <div class="image-container">
                    <img src="${result.result_image}?t=${new Date().getTime()}" alt="Result Image" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'200\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%23333\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'white\\' text-anchor=\\'middle\\'>画像なし</text></svg>'">
                </div>
                <div class="card-content">
                    <div>
                        <span class="status-badge ${statusClass}">${statusText}</span>
                        <span class="sim-score">${simPercentage}%</span>
                    </div>
                    <div class="details">
                        <p><strong>Origin Path</strong> ${result.origin_path}</p>
                        <p><strong>Dummy Path</strong> ${result.dummy_path}</p>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    }

    loadBtn.addEventListener('click', loadResults);
    filterMatch.addEventListener('change', render);
    filterNomatch.addEventListener('change', render);

    // Initial load
    loadResults();
});
