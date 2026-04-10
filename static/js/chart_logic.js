let myChart;
const BIN_COUNT = 96;
let isDrawing = false;

document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    const canvas = document.getElementById('mainChart');

    myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({length: BIN_COUNT}, (_, i) => i),
            datasets: [{
                label: 'Výstup z detektora',
                data: Array(BIN_COUNT).fill(10),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: { y: { min: 0, max: 100 } },
            plugins: { legend: { display: false } },
            onHover: (e) => { if (isDrawing) updateBinValue(e.native); }
        }
    });

    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 0) { isDrawing = true; updateBinValue(e); e.preventDefault(); }
    });
    canvas.addEventListener('mousemove', (e) => { if (isDrawing) updateBinValue(e); });
    window.addEventListener('mouseup', () => { isDrawing = false; });
});

function updateBinValue(event) {
    const rect = myChart.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const index = myChart.scales.x.getValueForPixel(x);
    let value = myChart.scales.y.getValueForPixel(y);

    if (index >= 0 && index < BIN_COUNT) {
        myChart.data.datasets[0].data[Math.floor(index)] = Math.max(0, Math.min(100, value));
        myChart.update('none');
    }
}

async function evaluateModel() {
    const rawData = myChart.data.datasets[0].data;
    const qValue = document.getElementById('qRange').value;
    const resultDiv = document.getElementById('result');
    const heatmapContainer = document.getElementById('heatmap-container');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ histogram: rawData, q: qValue })
        });

        const result = await response.json();
        
        // 1. Aktualizácia semaforu
        resultDiv.style.display = 'block';
        resultDiv.style.backgroundColor = result.color;
        resultDiv.innerHTML = `<strong>STAV: ${result.status}</strong><br>Quantile Score: ${result.score.toFixed(6)}`;

        // 2. Aktualizácia farieb v hlavnom grafe
        myChart.data.datasets[0].backgroundColor = result.color + "80"; // Pridaná priehľadnosť
        myChart.data.datasets[0].borderColor = result.color;
        myChart.update();

        // 3. Zobrazenie heatmapy (ak existuje)
        if (result.heatmap) {
            heatmapContainer.innerHTML = `<h3>Detailná analýza rezíduí:</h3><img src="${result.heatmap}" style="max-width:100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">`;
            heatmapContainer.style.display = 'block';
        } else {
            heatmapContainer.style.display = 'none';
        }

    } catch (error) {
        console.error("Chyba:", error);
    }
}

function resetData() {
    // 1. Vrátime dáta na pôvodnú hladinu
    myChart.data.datasets[0].data = Array(BIN_COUNT).fill(10);
    
    // 2. Resetujeme farby na pôvodnú modrú
    myChart.data.datasets[0].backgroundColor = 'rgba(54, 162, 235, 0.5)';
    myChart.data.datasets[0].borderColor = 'rgba(54, 162, 235, 1)';
    
    // 3. Schováme výsledkový panel
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    
    // 4. Prekreslíme graf
    myChart.update();
}