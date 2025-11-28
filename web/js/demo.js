// Demo Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    const modelSelect = document.getElementById('modelSelect');
    let selectedFile = null;

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', function(e) {
        handleFile(e.target.files[0]);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (file && file.type === 'text/csv' || file.name.endsWith('.csv')) {
            selectedFile = file;
            uploadArea.innerHTML = `
                <i class="fas fa-check-circle" style="font-size: 48px; color: var(--success-color); margin-bottom: 15px;"></i>
                <p><strong>${file.name}</strong></p>
                <p style="color: var(--text-light); font-size: 14px;">${(file.size / 1024).toFixed(2)} KB</p>
                <button class="btn btn-secondary" style="margin-top: 10px;" onclick="event.stopPropagation(); location.reload();">Change File</button>
            `;
            analyzeBtn.disabled = false;
        } else {
            showNotification('Please upload a CSV file.', 'error');
        }
    }

    // Analyze button
    analyzeBtn.addEventListener('click', async function() {
        if (!selectedFile) {
            showNotification('Please select a file first.', 'error');
            return;
        }

        loading.classList.add('show');
        results.classList.remove('show');
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model', modelSelect.value);

        try {
            // In production, this would call the Flask API
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                displayResults(data);
            } else {
                // Simulate results for demo (when API is not available)
                simulateResults();
            }
        } catch (error) {
            console.error('Error:', error);
            // Simulate results for demo
            simulateResults();
        } finally {
            loading.classList.remove('show');
            analyzeBtn.disabled = false;
        }
    });

    function simulateResults() {
        // Simulate API response for demo purposes
        const mockResults = {
            total_samples: Math.floor(Math.random() * 100) + 50,
            predictions: {
                normal: Math.floor(Math.random() * 50) + 30,
                dos: Math.floor(Math.random() * 20) + 5,
                probe: Math.floor(Math.random() * 15) + 3,
                r2l: Math.floor(Math.random() * 10) + 2,
                u2r: Math.floor(Math.random() * 5) + 1
            },
            accuracy: (Math.random() * 10 + 90).toFixed(2),
            model: modelSelect.value
        };

        displayResults(mockResults);
    }

    function displayResults(data) {
        const attacks = Object.entries(data.predictions || {})
            .filter(([key, value]) => key !== 'normal' && value > 0);

        let html = `
            <div class="result-card">
                <h4><i class="fas fa-chart-pie"></i> Summary</h4>
                <p><strong>Total Samples:</strong> ${data.total_samples || 0}</p>
                <p><strong>Model Used:</strong> ${data.model || 'random_forest'}</p>
                ${data.accuracy ? `<p><strong>Accuracy:</strong> ${data.accuracy}%</p>` : ''}
            </div>
        `;

        if (attacks.length > 0) {
            html += `
                <div class="result-card">
                    <h4><i class="fas fa-exclamation-triangle" style="color: var(--primary-color);"></i> Potential Attacks Detected</h4>
                    ${attacks.map(([type, count]) => `
                        <div style="margin: 10px 0;">
                            <span class="prediction-badge prediction-attack">${type.toUpperCase()}</span>
                            <span style="margin-left: 10px;">${count} instances</span>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            html += `
                <div class="result-card">
                    <h4><i class="fas fa-check-circle" style="color: var(--success-color);"></i> No Attacks Detected</h4>
                    <p>All traffic appears to be normal.</p>
                </div>
            `;
        }

        resultsContent.innerHTML = html;
        results.classList.add('show');
    }

    function showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#27ae60' : '#e74c3c'};
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
});

