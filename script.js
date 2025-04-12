document.addEventListener('DOMContentLoaded', function() {
    // Theme switcher
    const themeSwitch = document.getElementById('theme-switch');
    
    // Check for saved theme preference
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
        themeSwitch.checked = true;
    }
    
    themeSwitch.addEventListener('change', function() {
        if (this.checked) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        }
    });
    
    // Navigation highlighting
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('nav a');
    
    window.addEventListener('scroll', function() {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === current) {
                link.classList.add('active');
            }
        });
    });
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                window.scrollTo({
                    top: targetSection.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // "Get Started" button scrolls to recommendations section
    document.getElementById('getStartedBtn').addEventListener('click', function() {
        const recommendationsSection = document.getElementById('recommendations');
        window.scrollTo({
            top: recommendationsSection.offsetTop - 80,
            behavior: 'smooth'
        });
    });
    
    // Fetch market data from backend
    fetchMarketData();
    
    // Generate recommendations when button is clicked
    document.getElementById('generateRecommendations').addEventListener('click', function() {
        generateRecommendations();
    });
    
    // Generate prediction when button is clicked
    document.getElementById('predictBtn').addEventListener('click', function() {
        generatePrediction();
    });
    
    // Update timeframe for price chart
    document.getElementById('timeframeSelect').addEventListener('change', function() {
        updatePriceChart(this.value);
    });
    
    // Load initial data
    loadInitialData();
});

// Function to load initial data
function loadInitialData() {
    // Initial market data with placeholder values
    updateMarketStats({
        btc: { price: '$63,245.78', change: '+2.34%' },
        eth: { price: '$3,487.12', change: '+1.76%' },
        marketCap: { value: '$2.34T', change: '+0.89%' },
        volume: { value: '$142.6B', change: '-1.24%' }
    });
    
    // Generate initial recommendations
    setTimeout(() => {
        generateRecommendations();
    }, 1500);
    
    // Load price chart for 7 days by default
    updatePriceChart('7d');
    
    // Load analysis charts
    loadAnalysisCharts();
}

// Function to fetch market data from backend
function fetchMarketData() {
    // In a real implementation, this would be an API call to the backend
    // For now, we'll simulate the data fetching with a timeout
    setTimeout(() => {
        const marketData = {
            btc: { price: '$63,245.78', change: '+2.34%' },
            eth: { price: '$3,487.12', change: '+1.76%' },
            marketCap: { value: '$2.34T', change: '+0.89%' },
            volume: { value: '$142.6B', change: '-1.24%' }
        };
        
        updateMarketStats(marketData);
    }, 1000);
}

// Function to update market statistics
function updateMarketStats(data) {
    document.getElementById('btcPrice').textContent = data.btc.price;
    document.getElementById('btcChange').textContent = data.btc.change;
    document.getElementById('ethPrice').textContent = data.eth.price;
    document.getElementById('ethChange').textContent = data.eth.change;
    document.getElementById('marketCap').textContent = data.marketCap.value;
    document.getElementById('marketCapChange').textContent = data.marketCap.change;
    document.getElementById('volume').textContent = data.volume.value;
    document.getElementById('volumeChange').textContent = data.volume.change;
    
    // Add appropriate classes for positive/negative changes
    document.getElementById('btcChange').className = data.btc.change.includes('+') ? 'stat-change positive' : 'stat-change negative';
    document.getElementById('ethChange').className = data.eth.change.includes('+') ? 'stat-change positive' : 'stat-change negative';
    document.getElementById('marketCapChange').className = data.marketCap.change.includes('+') ? 'stat-change positive' : 'stat-change negative';
    document.getElementById('volumeChange').className = data.volume.change.includes('+') ? 'stat-change positive' : 'stat-change negative';
}

// Function to update price chart based on timeframe
function updatePriceChart(timeframe) {
    const chartContainer = document.getElementById('priceChart');
    chartContainer.innerHTML = '<div class="chart-loading">Loading chart data...</div>';
    
    // In a real implementation, this would be an API call to the backend
    // For now, we'll simulate the data fetching with a timeout
    setTimeout(() => {
        // This is where you'd normally create the chart with real data
        // For this example, we'll just use a placeholder image
        chartContainer.innerHTML = `
            <img src="/api/placeholder/800/400" alt="Price chart for ${timeframe}" class="placeholder-chart">
        `;
    }, 1200);
}

// Function to generate cryptocurrency recommendations
function generateRecommendations() {
    const riskTolerance = document.getElementById('riskTolerance').value;
    const timeHorizon = document.getElementById('timeHorizon').value;
    
    const recommendationsContainer = document.getElementById('recommendationsContainer');
    recommendationsContainer.innerHTML = `
        <div class="recommendation-loading">
            <div class="loading-spinner"></div>
            <p>Analyzing market data and generating recommendations...</p>
        </div>
    `;
    
    // In a real implementation, this would be an API call to the backend
    // For now, we'll simulate the recommendation generation with a timeout
    setTimeout(() => {
        // Sample recommendation data - in a real app, this would come from the ML model
        const recommendations = [
            {
                symbol: 'BTC',
                name: 'Bitcoin',
                price: '$63,245.78',
                prediction: '+5.2%',
                confidence: '87%',
                risk: 'Medium'
            },
            {
                symbol: 'ETH',
                name: 'Ethereum',
                price: '$3,487.12',
                prediction: '+8.7%',
                confidence: '82%',
                risk: 'Medium'
            },
            {
                symbol: 'SOL',
                name: 'Solana',
                price: '$142.34',
                prediction: '+12.3%',
                confidence: '76%',
                risk: 'High'
            }
        ];
        
        // Apply some filtering based on user preferences
        let filteredRecommendations = recommendations;
        if (riskTolerance === 'low') {
            filteredRecommendations = recommendations.filter(rec => rec.risk === 'Low');
        } else if (riskTolerance === 'medium') {
            filteredRecommendations = recommendations.filter(rec => rec.risk === 'Low' || rec.risk === 'Medium');
        }
        
        // Generate recommendation cards
        if (filteredRecommendations.length === 0) {
            recommendationsContainer.innerHTML = `
                <div class="recommendation-loading">
                    <p>No recommendations match your criteria. Try adjusting your risk tolerance or time horizon.</p>
                </div>
            `;
        } else {
            let recommendationsHTML = '';
            
            filteredRecommendations.forEach(rec => {
                recommendationsHTML += `
                    <div class="recommendation-card">
                        <div class="recommendation-header">
                            <h3><i class="fas fa-${getIconForCrypto(rec.symbol)}"></i> ${rec.name} (${rec.symbol})</h3>
                            <p>Current Price: ${rec.price}</p>
                        </div>
                        <div class="recommendation-body">
                            <div class="recommendation-stat">
                                <span class="recommendation-stat-label">Expected Growth (${getTimeHorizonText(timeHorizon)}):</span>
                                <span class="recommendation-stat-value">${rec.prediction}</span>
                            </div>
                            <div class="recommendation-stat">
                                <span class="recommendation-stat-label">Confidence:</span>
                                <span class="recommendation-stat-value">${rec.confidence}</span>
                            </div>
                            <div class="recommendation-stat">
                                <span class="recommendation-stat-label">Risk Level:</span>
                                <span class="recommendation-stat-value">${rec.risk}</span>
                            </div>
                        </div>
                        <div class="recommendation-actions">
                            <button class="btn-secondary view-details-btn" data-symbol="${rec.symbol}">View Details</button>
                        </div>
                    </div>
                `;
            });
            
            recommendationsContainer.innerHTML = recommendationsHTML;
            
            // Add event listeners to view details buttons
            document.querySelectorAll('.view-details-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const symbol = this.getAttribute('data-symbol');
                    document.getElementById('cryptoSelect').value = symbol;
                    document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });
                    generatePrediction();
                });
            });
        }
    }, 2000);
}

// Function to generate prediction for selected cryptocurrency
function generatePrediction() {
    const crypto = document.getElementById('cryptoSelect').value;
    const daysAhead = document.getElementById('daysAhead').value;
    
    const predictionResult = document.getElementById('predictionResult');
    const predictionChart = predictionResult.querySelector('.prediction-chart');
    
    predictionChart.innerHTML = `
        <div class="chart-loading">Generating prediction...</div>
    `;
    
    document.getElementById('predictedPrice').textContent = 'Calculating...';
    document.getElementById('confidenceLevel').textContent = 'Calculating...';
    document.getElementById('potentialUpside').textContent = 'Calculating...';
    document.getElementById('riskAssessment').textContent = 'Calculating...';
    
    // In a real implementation, this would be an API call to the backend
    // For now, we'll simulate the prediction generation with a timeout
    setTimeout(() => {
        // Sample prediction data - in a real app, this would come from the ML model
        const predictions = {
            'BTC': {
                predictedPrice: '$67,831.45',
                confidence: '87%',
                upside: '+7.3%',
                risk: 'Medium'
            },
            'ETH': {
                predictedPrice: '$3,787.24',
                confidence: '82%',
                upside: '+8.7%',
                risk: 'Medium'
            },
            'BNB': {
                predictedPrice: '$618.32',
                confidence: '79%',
                upside: '+5.2%',
                risk: 'Medium-Low'
            },
            'XRP': {
                predictedPrice: '$0.741',
                confidence: '73%',
                upside: '+12.4%',
                risk: 'High'
            },
            'ADA': {
                predictedPrice: '$0.567',
                confidence: '75%',
                upside: '+9.1%',
                risk: 'Medium-High'
            },
            'SOL': {
                predictedPrice: '$159.78',
                confidence: '76%',
                upside: '+12.3%',
                risk: 'High'
            },
            'DOT': {
                predictedPrice: '$8.92',
                confidence: '71%',
                upside: '+7.5%',
                risk: 'Medium-High'
            },
            'DOGE': {
                predictedPrice: '$0.167',
                confidence: '68%',
                upside: '+14.2%',
                risk: 'Very High'
            }
        };
        
        const prediction = predictions[crypto];
        
        // Update prediction metrics
        document.getElementById('predictedPrice').textContent = prediction.predictedPrice;
        document.getElementById('confidenceLevel').textContent = prediction.confidence;
        document.getElementById('potentialUpside').textContent = prediction.upside;
        document.getElementById('riskAssessment').textContent = prediction.risk;
        
        // Update prediction chart
        predictionChart.innerHTML = `
            <img src="/api/placeholder/800/400" alt="${crypto} price prediction chart" class="placeholder-chart">
        `;
    }, 2500);
}

// Function to load analysis charts
function loadAnalysisCharts() {
    const correlationChart = document.getElementById('correlationChart');
    const volatilityChart = document.getElementById('volatilityChart');
    const sentimentChart = document.getElementById('sentimentChart');
    
    // In a real implementation, these would be API calls to the backend
    // For now, we'll use placeholder images
    setTimeout(() => {
        correlationChart.innerHTML = `
            <img src="/api/placeholder/400/250" alt="Correlation heatmap" class="placeholder-chart">
        `;
        
        volatilityChart.innerHTML = `
            <img src="/api/placeholder/400/250" alt="Volatility chart" class="placeholder-chart">
        `;
        
        sentimentChart.innerHTML = `
            <img src="/api/placeholder/400/250" alt="Sentiment analysis" class="placeholder-chart">
        `;
    }, 1500);
}

// Helper function to get appropriate icon for cryptocurrency
function getIconForCrypto(symbol) {
    switch(symbol) {
        case 'BTC':
            return 'bitcoin-sign';
        case 'ETH':
            return 'ethereum';
        case 'BNB':
            return 'b';
        case 'SOL':
            return 's';
        case 'DOGE':
            return 'd';
        default:
            return 'coins';
    }
}

// Helper function to get text for time horizon
function getTimeHorizonText(timeHorizon) {
    switch(timeHorizon) {
        case 'short':
            return '1-7 days';
        case 'medium':
            return '1-4 weeks';
        case 'long':
            return '1-6 months';
        default:
            return '1-4 weeks';
    }
}