// Chart Management System using Chart.js
(function() {
    'use strict';
    
    class ChartManager {
    constructor() {
        this.priceChart = null;
        this.accuracyChart = null;
        this.chartColors = {
            primary: '#3B82F6',
            secondary: '#10B981',
            warning: '#F59E0B',
            danger: '#EF4444',
            purple: '#8B5CF6',
            gray: '#6B7280'
        };
        this.init();
    }

    init() {
        // Set Chart.js global defaults
        Chart.defaults.font.family = 'Inter, sans-serif';
        Chart.defaults.font.size = 12;
        Chart.defaults.color = '#374151';
        Chart.defaults.plugins.legend.display = true;
        Chart.defaults.plugins.legend.position = 'top';
    }

    // Initialize price chart
    initPriceChart() {
        const ctx = document.getElementById('priceChart');
        if (!ctx) return;

        this.destroyChart('price');

        this.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '實際價格',
                    data: [],
                    borderColor: this.chartColors.primary,
                    backgroundColor: this.chartColors.primary + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    tension: 0.1,
                    fill: true
                }, {
                    label: '預測價格',
                    data: [],
                    borderColor: this.chartColors.danger,
                    backgroundColor: this.chartColors.danger + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    tension: 0.1,
                    borderDash: [5, 5],
                    fill: false
                }, {
                    label: 'SMA (20)',
                    data: [],
                    borderColor: this.chartColors.secondary,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                }, {
                    label: 'EMA (20)',
                    data: [],
                    borderColor: this.chartColors.warning,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: '股價走勢與AI預測',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#3B82F6',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                return `${label}: $${value.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: '日期'
                        },
                        grid: {
                            color: '#E5E7EB'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: '股價 (USD)'
                        },
                        grid: {
                            color: '#E5E7EB'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                },
                elements: {
                    point: {
                        hoverBackgroundColor: '#fff',
                        hoverBorderWidth: 3
                    }
                }
            }
        });
    }

    // Initialize accuracy chart
    initAccuracyChart() {
        const ctx = document.getElementById('accuracyChart');
        if (!ctx) return;

        this.destroyChart('accuracy');

        this.accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['MAE', 'RMSE', 'R²', '方向準確率'],
                datasets: [{
                    label: '模型評估指標',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        this.chartColors.primary + '80',
                        this.chartColors.secondary + '80',
                        this.chartColors.purple + '80',
                        this.chartColors.warning + '80'
                    ],
                    borderColor: [
                        this.chartColors.primary,
                        this.chartColors.secondary,
                        this.chartColors.purple,
                        this.chartColors.warning
                    ],
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '模型準確率評估',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#3B82F6',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const label = context.label;
                                const value = context.parsed.y;
                                if (label === 'R²' || label === '方向準確率') {
                                    return `${label}: ${(value * 100).toFixed(2)}%`;
                                }
                                return `${label}: ${value.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: '數值'
                        },
                        grid: {
                            color: '#E5E7EB'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Update price chart with new data
    updatePriceChart(stockData, predictions = [], smaData = [], emaData = []) {
        if (!this.priceChart) {
            this.initPriceChart();
        }

        // Prepare data
        const labels = stockData.map(item => this.formatDateForChart(item.date));
        const prices = stockData.map(item => item.close);
        
        // Update chart data
        this.priceChart.data.labels = labels;
        this.priceChart.data.datasets[0].data = prices;
        
        // Add prediction data
        if (predictions.length > 0) {
            const predictionLabels = predictions.map(item => this.formatDateForChart(item.date));
            const predictionPrices = predictions.map(item => item.price);
            
            // Extend labels to include prediction dates
            const allLabels = [...labels, ...predictionLabels];
            this.priceChart.data.labels = allLabels;
            
            // Create prediction array: nulls for historical data, then prediction values
            const predictionArray = new Array(prices.length).fill(null);
            // Add connection point (last actual price)
            predictionArray[prices.length - 1] = prices[prices.length - 1];
            // Add prediction values
            predictionPrices.forEach(price => {
                predictionArray.push(price);
            });
            
            this.priceChart.data.datasets[1].data = predictionArray;
        } else {
            // Clear prediction data if no predictions
            this.priceChart.data.datasets[1].data = [];
        }
        
        // Add SMA data
        if (smaData.length > 0) {
            const smaArray = new Array(labels.length).fill(null);
            smaData.forEach(item => {
                const index = labels.indexOf(this.formatDateForChart(item.date));
                if (index !== -1) {
                    smaArray[index] = item.value;
                }
            });
            this.priceChart.data.datasets[2].data = smaArray;
        }
        
        // Add EMA data
        if (emaData.length > 0) {
            const emaArray = new Array(labels.length).fill(null);
            emaData.forEach(item => {
                const index = labels.indexOf(this.formatDateForChart(item.date));
                if (index !== -1) {
                    emaArray[index] = item.value;
                }
            });
            this.priceChart.data.datasets[3].data = emaArray;
        }

        this.priceChart.update();
    }

    // Update accuracy chart with metrics
    updateAccuracyChart(metrics) {
        if (!this.accuracyChart) {
            this.initAccuracyChart();
        }

        const { mae, rmse, r2, directionAccuracy } = metrics;
        
        this.accuracyChart.data.datasets[0].data = [
            mae || 0,
            rmse || 0,
            r2 || 0,
            directionAccuracy || 0
        ];

        this.accuracyChart.update();
    }

    // Create candlestick chart for stock data
    createCandlestickChart(stockData) {
        const ctx = document.getElementById('priceChart');
        if (!ctx) return;

        this.destroyChart('price');

        // Transform data for candlestick format
        const candlestickData = stockData.map(item => ({
            x: item.date,
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close
        }));

        this.priceChart = new Chart(ctx, {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: '股價',
                    data: candlestickData,
                    color: {
                        up: this.chartColors.secondary,
                        down: this.chartColors.danger,
                        unchanged: this.chartColors.gray
                    },
                    borderColor: {
                        up: this.chartColors.secondary,
                        down: this.chartColors.danger,
                        unchanged: this.chartColors.gray
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'K線圖',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '股價 (USD)'
                        }
                    }
                }
            }
        });
    }

    // Create volume chart
    createVolumeChart(stockData) {
        const ctx = document.getElementById('volumeChart');
        if (!ctx) return;

        const labels = stockData.map(item => this.formatDateForChart(item.date));
        const volumes = stockData.map(item => item.volume);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '成交量',
                    data: volumes,
                    backgroundColor: this.chartColors.gray + '60',
                    borderColor: this.chartColors.gray,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '成交量',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        title: {
                            display: true,
                            text: '成交量'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value / 1000000).toFixed(1) + 'M';
                            }
                        }
                    }
                }
            }
        });
    }

    // Format date for chart display
    formatDateForChart(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('zh-TW', {
            month: 'short',
            day: 'numeric'
        });
    }

    // Destroy specific chart
    destroyChart(type) {
        if (type === 'price' && this.priceChart) {
            this.priceChart.destroy();
            this.priceChart = null;
        } else if (type === 'accuracy' && this.accuracyChart) {
            this.accuracyChart.destroy();
            this.accuracyChart = null;
        }
    }

    // Destroy all charts
    destroyAllCharts() {
        this.destroyChart('price');
        this.destroyChart('accuracy');
    }

    // Export chart as image
    exportChart(type, filename) {
        let chart;
        if (type === 'price') {
            chart = this.priceChart;
        } else if (type === 'accuracy') {
            chart = this.accuracyChart;
        }
        
        if (!chart) return;

        const url = chart.toBase64Image();
        const link = document.createElement('a');
        link.download = filename || `${type}_chart.png`;
        link.href = url;
        link.click();
    }

    // Update chart themes
    updateTheme(theme = 'light') {
        const isDark = theme === 'dark';
        const textColor = isDark ? '#F3F4F6' : '#374151';
        const gridColor = isDark ? '#374151' : '#E5E7EB';

        Chart.defaults.color = textColor;
        
        if (this.priceChart) {
            this.priceChart.options.scales.x.grid.color = gridColor;
            this.priceChart.options.scales.y.grid.color = gridColor;
            this.priceChart.update();
        }
        
        if (this.accuracyChart) {
            this.accuracyChart.options.scales.x.grid.color = gridColor;
            this.accuracyChart.options.scales.y.grid.color = gridColor;
            this.accuracyChart.update();
        }
    }
}

    // Initialize global chart manager
    if (!window.chartManager) {
        window.chartManager = new ChartManager();
    }
})();