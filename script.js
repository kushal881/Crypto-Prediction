document.addEventListener('DOMContentLoaded', function() {
    // Theme Toggle
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    const themeIcon = themeToggle.querySelector('i');
    
    themeToggle.addEventListener('click', function() {
        body.classList.toggle('dark-mode');
        
        if (body.classList.contains('dark-mode')) {
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.classList.remove('fa-sun');
            themeIcon.classList.add('fa-moon');
            localStorage.setItem('theme', 'light');
        }
    });
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        body.classList.add('dark-mode');
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
    }
    
    // Navigation active state
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Prediction Button Click
    const predictBtn = document.getElementById('predict-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const predictionResults = document.getElementById('prediction-results');
    
    predictBtn.addEventListener('click', function() {
        // Show loading overlay
        loadingOverlay.classList.add('active');
        
        // Simulate API call and model processing
        setTimeout(function() {
            loadingOverlay.classList.remove('active');
            predictionResults.classList.remove('hidden');
            
            // Scroll to results
            predictionResults.scrollIntoView({ behavior: 'smooth' });
            
            // Update prediction data
            updatePredictionData();
            
            // Initialize chart
            initPredictionChart();
        }, 2500);
    });
    
    // Testimonial Carousel
    const carousel = document.querySelector('.testimonial-carousel');
    const prevBtn = document.getElementById('prev-testimonial');
    const nextBtn = document.getElementById('next-testimonial');
    const indicators = document.querySelectorAll('.carousel-indicators .indicator');
    let currentSlide = 0;
    
    nextBtn.addEventListener('click', function() {
        currentSlide = (currentSlide + 1) % indicators.length;
        updateCarousel();
    });
    
    prevBtn.addEventListener('click', function() {
        currentSlide = (currentSlide - 1 + indicators.length) % indicators.length;
        updateCarousel();
    });
    
    function updateCarousel() {
        carousel.scrollTo({
            left: currentSlide * carousel.offsetWidth,
            behavior: 'smooth'
        });
        
        indicators.forEach((ind, i) => {
            ind.classList.toggle('active', i === currentSlide);
        });
    }
    
    // Auto rotate carousel
    setInterval(function() {
        currentSlide = (currentSlide + 1) % indicators.length;
        updateCarousel();
    }, 8000);
    
    // Handle view toggle (chart/table)
    const viewButtons = document.querySelectorAll('.control-btn[data-view]');
    viewButtons.forEach(button => {
        button.addEventListener('click', function() {
            viewButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Handle view switch here if needed
        });
    });
    
    // Initialize indicator bars
    const indicatorBars = document.querySelectorAll('.indicator-bar');
    indicatorBars.forEach(bar => {
        const value = bar.getAttribute('data-value');
        bar.style.setProperty('--value', `${value}%`);
    });
    
    // Newsletter form submission
    const newsletterForm = document.querySelector('.newsletter-form');
    newsletterForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const email = this.querySelector('input').value;
        
        if (validateEmail(email)) {
            alert('Thank you for subscribing! You will receive our weekly predictions soon.');
            this.reset();
        } else {
            alert('Please enter a valid email address.');
        }
    });
    
    function validateEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }
});

// Function to update prediction data with random but realistic values
function updatePredictionData() {
    const selectedCrypto = document.getElementById('crypto-select').value;
    const timeframe = document.getElementById('timeframe-select').value;
    
    // Get elements to update
    const coinName = document.getElementById('coin-name');
    const coinIcon = document.getElementById('coin-icon');
    const currentPrice = document.getElementById('current-price');
    const change24h = document.getElementById('change-24h');
    const change7d = document.getElementById('change-7d');
    const marketCap = document.getElementById('market-cap');
    const volume = document.getElementById('volume');
    const predictedHigh = document.getElementById('predicted-high');
    const predictedTarget = document.getElementById('predicted-target');
    const predictedLow = document.getElementById('predicted-low');
    const targetDate = document.getElementById('target-date');
    
    // Crypto data mapping (simplified)
    const cryptoData = {
        BTC: {
            name: 'Bitcoin (BTC)',
            basePrice: 48000 + Math.random() * 5000,
            marketCap: '918.4 B',
            volume: '32.7 B'
        },
        ETH: {
            name: 'Ethereum (ETH)',
            basePrice: 3200 + Math.random() * 400,
            marketCap: '372.5 B',
            volume: '18.3 B'
        },
        BNB: {
            name: 'Binance Coin (BNB)',
            basePrice: 450 + Math.random() * 50,
            marketCap: '71.2 B',
            volume: '2.1 B'
        },
        SOL: {
            name: 'Solana (SOL)',
            basePrice: 100 + Math.random() * 30,
            marketCap: '42.8 B',
            volume: '3.5 B'
        },
        ADA: {
            name: 'Cardano (ADA)',
            basePrice: 0.5 + Math.random() * 0.2,
            marketCap: '18.3 B',
            volume: '1.2 B'
        },
        DOT: {
            name: 'Polkadot (DOT)',
            basePrice: 12 + Math.random() * 4,
            marketCap: '14.7 B',
            volume: '0.9 B'
        },
        DOGE: {
            name: 'Dogecoin (DOGE)',
            basePrice: 0.12 + Math.random() * 0.05,
            marketCap: '16.8 B',
            volume: '1.3 B'
        },
        AVAX: {
            name: 'Avalanche (AVAX)',
            basePrice: 28 + Math.random() * 8,
            marketCap: '11.2 B',
            volume: '0.8 B'
        }
    };
    
    const data = cryptoData[selectedCrypto];
    const currentPriceValue = data.basePrice;
    const change24hValue = (Math.random() * 10 - 3).toFixed(2); // Random change between -3% and +7%
    const change7dValue = (Math.random() * 15 - 5).toFixed(2); // Random change between -5% and +10%
    
    // Calculate predicted prices based on
    const volatilityFactor = parseInt(timeframe) / 30;
    const predictionVariance = currentPriceValue * (0.1 * Math.sqrt(volatilityFactor));
    
    const predictedHighValue = currentPriceValue * (1 + (0.15 * Math.sqrt(volatilityFactor) + (Math.random() * 0.05)));
    const predictedTargetValue = currentPriceValue * (1 + (0.08 * Math.sqrt(volatilityFactor) + (Math.random() * 0.03)));
    const predictedLowValue = currentPriceValue * (1 - (0.05 * Math.sqrt(volatilityFactor) + (Math.random() * 0.05)));
    
    // Format numbers based on price range
    function formatPrice(price) {
        if (price >= 1000) {
            return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 0 });
        } else if (price >= 1) {
            return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 2 });
        } else {
            return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 4 });
        }
    }
    
    // Update DOM elements
    coinName.textContent = data.name;
    currentPrice.textContent = formatPrice(currentPriceValue);
    
    if (parseFloat(change24hValue) >= 0) {
        change24h.innerHTML = `+${change24hValue}% <i class="fas fa-caret-up"></i>`;
        change24h.classList.add('up');
        change24h.classList.remove('down');
    } else {
        change24h.innerHTML = `${change24hValue}% <i class="fas fa-caret-down"></i>`;
        change24h.classList.add('down');
        change24h.classList.remove('up');
    }
    
    if (parseFloat(change7dValue) >= 0) {
        change7d.innerHTML = `+${change7dValue}% <i class="fas fa-caret-up"></i>`;
        change7d.classList.add('up');
        change7d.classList.remove('down');
    } else {
        change7d.innerHTML = `${change7dValue}% <i class="fas fa-caret-down"></i>`;
        change7d.classList.add('down');
        change7d.classList.remove('up');
    }
    
    marketCap.textContent = '$' + data.marketCap;
    volume.textContent = '$' + data.volume;
    
    predictedHigh.textContent = formatPrice(predictedHighValue);
    predictedTarget.textContent = formatPrice(predictedTargetValue);
    predictedLow.textContent = formatPrice(predictedLowValue);
    
    const highChangePercent = ((predictedHighValue / currentPriceValue - 1) * 100).toFixed(2);
    const targetChangePercent = ((predictedTargetValue / currentPriceValue - 1) * 100).toFixed(2);
    const lowChangePercent = ((predictedLowValue / currentPriceValue - 1) * 100).toFixed(2);
    
    predictedHigh.nextElementSibling.textContent = `+${highChangePercent}%`;
    predictedTarget.nextElementSibling.textContent = `+${targetChangePercent}%`;
    predictedLow.nextElementSibling.textContent = `${lowChangePercent}% worst case`;
    
    targetDate.textContent = `in ${timeframe} days`;
}

// Function to initialize and render the price prediction chart
function initPredictionChart() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    const selectedTimeframe = parseInt(document.getElementById('timeframe-select').value);
    
    // Generate dates for the chart
    const dates = [];
    const currentDate = new Date();
    
    // Past dates (30 days historical data)
    for (let i = 30; i >= 1; i--) {
        const date = new Date();
        date.setDate(currentDate.getDate() - i);
        dates.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    
    // Current date
    dates.push(currentDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    
    // Future dates based on selected timeframe
    for (let i = 1; i <= selectedTimeframe; i++) {
        const date = new Date();
        date.setDate(currentDate.getDate() + i);
        dates.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    
    // Generate price data
    const selectedCrypto = document.getElementById('crypto-select').value;
    let basePrice;
    
    switch (selectedCrypto) {
        case 'BTC':
            basePrice = 48000 + Math.random() * 5000;
            break;
        case 'ETH':
            basePrice = 3200 + Math.random() * 400;
            break;
        case 'BNB':
            basePrice = 450 + Math.random() * 50;
            break;
        case 'SOL':
            basePrice = 100 + Math.random() * 30;
            break;
        case 'ADA':
            basePrice = 0.5 + Math.random() * 0.2;
            break;
        case 'DOT':
            basePrice = 12 + Math.random() * 4;
            break;
        case 'DOGE':
            basePrice = 0.12 + Math.random() * 0.05;
            break;
        case 'AVAX':
            basePrice = 28 + Math.random() * 8;
            break;
        default:
            basePrice = 50000;
    }
    
    // Generate historical data with realistic patterns
    const historicalData = [];
    let currentValue = basePrice * 0.85; // Start a bit lower than current
    
    for (let i = 0; i < 31; i++) {
        // Add some random walk with momentum
        const dailyChange = (Math.random() - 0.48) * (basePrice * 0.02); // Slightly bullish bias
        
        // Add some volatility patterns
        const volatility = Math.sin(i / 5) * (basePrice * 0.005);
        
        // Add trend
        const trend = i / 30 * (basePrice * 0.15);
        
        currentValue += dailyChange + volatility + (trend / 30);
        historicalData.push(currentValue);
    }
    
    // Current price is the last item in historical data
    const currentPrice = historicalData[historicalData.length - 1];
    
    // Generate prediction data
    const predictionData = new Array(selectedTimeframe).fill(null);
    const predictionUpper = new Array(selectedTimeframe).fill(null);
    const predictionLower = new Array(selectedTimeframe).fill(null);
    
    // Calculate end target price with some randomness
    const volatilityFactor = selectedTimeframe / 30;
    const endTargetChange = 0.08 * Math.sqrt(volatilityFactor) + (Math.random() * 0.03);
    const endPrice = currentPrice * (1 + endTargetChange);
    
    // Generate prediction line - smooth curve to target
    for (let i = 0; i < selectedTimeframe; i++) {
        const progress = i / (selectedTimeframe - 1);
        // Cubic easing for more realistic curve
        const easedProgress = progress < 0.5 
            ? 4 * progress * progress * progress 
            : 1 - Math.pow(-2 * progress + 2, 3) / 2;
            
        predictionData[i] = currentPrice + (endPrice - currentPrice) * easedProgress;
        
        // Generate confidence intervals (narrower near current time, wider in the future)
        const uncertaintyFactor = 0.005 + (progress * 0.03);
        predictionUpper[i] = predictionData[i] * (1 + uncertaintyFactor * (1 + Math.sqrt(progress)));
        predictionLower[i] = predictionData[i] * (1 - uncertaintyFactor * (1 + Math.sqrt(progress)));
    }
    
    // Combine datasets for the chart
    const completeData = {
        labels: dates,
        datasets: [
            // Historical data
            {
                label: 'Historical Price',
                data: [...historicalData, ...new Array(selectedTimeframe).fill(null)],
                borderColor: getComputedStyle(document.documentElement).getPropertyValue('--chart-line').trim(),
                backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--chart-fill').trim(),
                pointRadius: 0,
                borderWidth: 2,
                fill: false,
                tension: 0.4
            },
            // Prediction line
            {
                label: 'Predicted Price',
                data: [...new Array(31).fill(null), ...predictionData],
                borderColor: getComputedStyle(document.documentElement).getPropertyValue('--chart-prediction').trim(),
                backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--chart-prediction-fill').trim(),
                pointRadius: 0,
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                tension: 0.4
            },
            // Upper confidence bound
            {
                label: 'Upper Bound',
                data: [...new Array(31).fill(null), ...predictionUpper],
                borderColor: 'transparent',
                backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--chart-uncertainty').trim(),
                pointRadius: 0,
                fill: '+1', // Fill to the next dataset
                tension: 0.4
            },
            // Lower confidence bound
            {
                label: 'Lower Bound',
                data: [...new Array(31).fill(null), ...predictionLower],
                borderColor: 'transparent',
                backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--chart-uncertainty').trim(),
                pointRadius: 0,
                fill: false,
                tension: 0.4
            }
        ]
    };
    
    // Check if a chart already exists and destroy it
    if (window.predictionChart) {
        window.predictionChart.destroy();
    }
    
    // Create new chart
    window.predictionChart = new Chart(ctx, {
        type: 'line',
        data: completeData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        maxTicksLimit: 8,
                        font: {
                            size: 10
                        }
                    }
                },
                y: {
                    position: 'right', 
                    grid: {
                        color: getComputedStyle(document.documentElement).getPropertyValue('--border').trim()
                    },
                    ticks: {
                        callback: function(value) {
                            if (value >= 1000) {
                                return '$' + (value / 1000).toFixed(1) + 'k';
                            } else {
                                return '$' + value.toFixed(0);
                            }
                        },
                        font: {
                            size: 10
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        title: function(tooltipItems) {
                            return tooltipItems[0].label;
                        },
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', {
                                    style: 'currency',
                                    currency: 'USD',
                                    minimumFractionDigits: 0,
                                    maximumFractionDigits: 2
                                }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// Function to simulate fetch from a backend API
async function fetchPredictionFromBackend(cryptoSymbol, timeframeDays) {
    // This would be replaced with a real API call to the Python backend
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate machine learning model response
            resolve({
                success: true,
                data: {
                    // Data would be filled by the Python backend
                }
            });
        }, 1500);
    });
}

// This is where we would connect to the Python backend
// For demonstration purposes, we'll simulate the backend functionality
class CryptoPredictionBackend {
    constructor() {
        // Initialize backend connection
        console.log('Connecting to ML prediction service...');
    }
    
    async getPrediction(cryptoSymbol, timeframeDays) {
        // In a real implementation, this would make an API request to the Python backend
        try {
            const response = await fetchPredictionFromBackend(cryptoSymbol, timeframeDays);
            return response;
        } catch (error) {
            console.error('Error fetching prediction data:', error);
            return {
                success: false,
                error: 'Failed to connect to prediction service'
            };
        }
    }
}

// Initialize backend service when the page loads
const backendService = new CryptoPredictionBackend();