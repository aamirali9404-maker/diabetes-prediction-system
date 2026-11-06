// ============================================
// Diabetes Prediction System - Client Script
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsCard = document.getElementById('resultsCard');
    const resultsDiv = document.getElementById('results');
    const predictBtn = document.getElementById('predictBtn');
    
    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = {
            pregnancies: document.getElementById('pregnancies').value,
            glucose: document.getElementById('glucose').value,
            bloodPressure: document.getElementById('bloodPressure').value,
            skinThickness: document.getElementById('skinThickness').value,
            insulin: document.getElementById('insulin').value,
            bmi: document.getElementById('bmi').value,
            diabetesPedigree: document.getElementById('diabetesPedigree').value,
            age: document.getElementById('age').value
        };
        
        // Validate inputs
        if (!validateInputs(formData)) {
            showError('Please fill in all fields with valid values.');
            return;
        }
        
        // Show loading state
        setLoadingState(true);
        
        try {
            // Make prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                displayResults(data);
            } else {
                showError(data.error || 'An error occurred during prediction.');
            }
        } catch (error) {
            showError('Failed to connect to the server. Please try again.');
            console.error('Error:', error);
        } finally {
            setLoadingState(false);
        }
    });
    
    // Input validation
    function validateInputs(data) {
        for (let key in data) {
            if (!data[key] || data[key] === '' || isNaN(data[key])) {
                return false;
            }
        }
        return true;
    }
    
    // Loading state management
    function setLoadingState(isLoading) {
        if (isLoading) {
            predictBtn.classList.add('loading');
            predictBtn.disabled = true;
            predictBtn.querySelector('.btn-text').textContent = 'Processing';
        } else {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
            predictBtn.querySelector('.btn-text').textContent = 'Assess Risk';
        }
    }
    
    // Display prediction results
    function displayResults(data) {
        const riskClass = data.risk_level.toLowerCase();
        const diabetesProb = (data.probability.diabetes * 100).toFixed(1);
        const noDiabetesProb = (data.probability.no_diabetes * 100).toFixed(1);
        
        resultsDiv.innerHTML = `
            <div class="result-header ${riskClass}-risk">
                <span class="risk-badge ${riskClass}">${data.risk_level} Risk</span>
                <div class="result-percentage">${diabetesProb}%</div>
                <p class="result-message">${data.message}</p>
            </div>
            
            <div class="probability-bars">
                <div class="probability-item">
                    <div class="probability-label">
                        <span>No Diabetes</span>
                        <span>${noDiabetesProb}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill no-diabetes" style="width: ${noDiabetesProb}%"></div>
                    </div>
                </div>
                
                <div class="probability-item">
                    <div class="probability-label">
                        <span>Diabetes</span>
                        <span>${diabetesProb}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill diabetes" style="width: ${diabetesProb}%"></div>
                    </div>
                </div>
            </div>
        `;
        
        // Show results card with animation
        resultsCard.style.display = 'block';
        
        // Scroll to results
        setTimeout(() => {
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 300);
        
        // Animate probability bars
        setTimeout(() => {
            const fills = document.querySelectorAll('.probability-fill');
            fills.forEach(fill => {
                const width = fill.style.width;
                fill.style.width = '0%';
                setTimeout(() => {
                    fill.style.width = width;
                }, 50);
            });
        }, 400);
    }
    
    // Error display
    function showError(message) {
        resultsDiv.innerHTML = `
            <div class="result-header high-risk">
                <p class="result-message" style="color: var(--color-danger);">
                    ⚠ ${message}
                </p>
            </div>
        `;
        resultsCard.style.display = 'block';
    }
    
    // Add input formatting helpers
    const inputs = document.querySelectorAll('.form-input');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
});

// Sample data for testing
function fillSampleData() {
    document.getElementById('pregnancies').value = '6';
    document.getElementById('glucose').value = '148';
    document.getElementById('bloodPressure').value = '72';
    document.getElementById('skinThickness').value = '35';
    document.getElementById('insulin').value = '0';
    document.getElementById('bmi').value = '33.6';
    document.getElementById('diabetesPedigree').value = '0.627';
    document.getElementById('age').value = '50';
}

// Expose to console for testing
console.log('Diabetes Prediction System loaded successfully!');
console.log('Use fillSampleData() to test with sample values.');
