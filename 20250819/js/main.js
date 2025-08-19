// Main Application Controller
(function() {
    'use strict';
    
    class StockPredictionApp {
    constructor() {
        this.currentStock = null;
        this.currentModel = 'sma';
        this.predictionDays = 7;
        this.isAnalyzing = false;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateStatus('系統已就緒，請選擇股票開始分析', 'info');
        
        // Initialize charts
        chartManager.initPriceChart();
        chartManager.initAccuracyChart();
    }

    bindEvents() {
        // Stock selection
        const stockSelect = document.getElementById('stockSelect');
        if (stockSelect) {
            stockSelect.addEventListener('change', (e) => {
                this.currentStock = e.target.value;
                if (this.currentStock) {
                    this.loadStockData();
                }
            });
        }

        // Model selection
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.currentModel = e.target.value;
            });
        }

        // Prediction days
        const predictionDays = document.getElementById('predictionDays');
        if (predictionDays) {
            predictionDays.addEventListener('change', (e) => {
                this.predictionDays = parseInt(e.target.value);
            });
        }

        // Analyze button
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => {
                this.runAnalysis();
            });
        }

        // Clear button
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearData();
            });
        }
    }

    async loadStockData() {
        if (!this.currentStock) return;

        try {
            this.updateStatus(`正在載入 ${this.currentStock} 股票資料...`, 'info');

            // Try to get real stock data first
            let stockData;
            try {
                stockData = await stockDataManager.getRealStockData(this.currentStock, 365);
                this.updateStatus(`成功載入 ${this.currentStock} 真實股價資料`, 'success');
            } catch (error) {
                stockData = stockDataManager.getStockData(this.currentStock);
                this.updateStatus(`載入模擬資料 ${this.currentStock}`, 'warning');
            }

            if (!stockData || stockData.length === 0) {
                throw new Error('股票資料不存在');
            }

            // Display recent data (last 30 days)
            const recentData = stockDataManager.getRecentData(this.currentStock, 30);
            this.updateStockTable(recentData);

            // Update price chart with historical data
            const smaData = stockDataManager.calculateSMA(this.currentStock, 20);
            const emaData = stockDataManager.calculateEMA(this.currentStock, 20);
            chartManager.updatePriceChart(recentData, [], smaData, emaData);

            // Display stock statistics
            this.displayStockStats();

            // Load sentiment and financial data
            this.loadSentimentData();
            this.loadFinancialData();

            this.updateStatus(`${this.currentStock} 股票資料載入完成`, 'success');

        } catch (error) {
            console.error('載入股票資料錯誤:', error);
            this.updateStatus(`載入股票資料失敗: ${error.message}`, 'error');
        }
    }

    async runAnalysis() {
        if (!this.currentStock) {
            this.updateStatus('請先選擇股票', 'warning');
            return;
        }

        if (this.isAnalyzing) {
            this.updateStatus('分析正在進行中，請稍候...', 'warning');
            return;
        }

        try {
            this.isAnalyzing = true;
            this.setAnalyzeButtonLoading(true);

            this.updateStatus(`正在使用 ${this.getModelName()} 進行股價預測...`, 'info');

            // Get stock data
            const stockData = stockDataManager.getStockData(this.currentStock);
            
            // Run AI prediction
            const result = await aiModelManager.predict(
                this.currentModel, 
                stockData, 
                this.predictionDays
            );

            // Update charts with predictions
            const recentData = stockDataManager.getRecentData(this.currentStock, 30);
            const smaData = stockDataManager.calculateSMA(this.currentStock, 20);
            const emaData = stockDataManager.calculateEMA(this.currentStock, 20);
            
            chartManager.updatePriceChart(recentData, result.predictions, smaData, emaData);
            chartManager.updateAccuracyChart(result.metrics);

            // Update metrics display
            this.updateMetricsDisplay(result.metrics);

            // Display prediction results
            this.displayPredictionResults(result.predictions);

            this.updateStatus(`預測完成！使用 ${this.getModelName()} 預測未來 ${this.predictionDays} 天`, 'success');

        } catch (error) {
            console.error('分析錯誤:', error);
            this.updateStatus(`分析失敗: ${error.message}`, 'error');
        } finally {
            this.isAnalyzing = false;
            this.setAnalyzeButtonLoading(false);
        }
    }

    updateStockTable(stockData) {
        const tableBody = document.getElementById('stockTableBody');
        if (!tableBody) return;

        tableBody.innerHTML = '';

        // Show last 10 records
        const displayData = stockData.slice(-10);
        
        displayData.forEach(item => {
            const row = document.createElement('tr');
            row.className = 'hover:bg-gray-50 transition-colors duration-150';
            
            const change = displayData.indexOf(item) > 0 ? 
                item.close - displayData[displayData.indexOf(item) - 1].close : 0;
            const changeClass = change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-600';

            row.innerHTML = `
                <td class="px-4 py-3 text-sm text-gray-900">${this.formatDate(item.date)}</td>
                <td class="px-4 py-3 text-sm text-gray-900">$${item.open.toFixed(2)}</td>
                <td class="px-4 py-3 text-sm text-green-600">$${item.high.toFixed(2)}</td>
                <td class="px-4 py-3 text-sm text-red-600">$${item.low.toFixed(2)}</td>
                <td class="px-4 py-3 text-sm font-medium ${changeClass}">$${item.close.toFixed(2)}</td>
                <td class="px-4 py-3 text-sm text-gray-600">${this.formatVolume(item.volume)}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    updateMetricsDisplay(metrics) {
        const elements = {
            maeValue: document.getElementById('maeValue'),
            rmseValue: document.getElementById('rmseValue'),
            r2Value: document.getElementById('r2Value'),
            directionAccuracy: document.getElementById('directionAccuracy')
        };

        Object.keys(elements).forEach(key => {
            if (elements[key]) {
                elements[key].classList.add('metric-update');
                setTimeout(() => {
                    elements[key].classList.remove('metric-update');
                }, 500);
            }
        });

        if (elements.maeValue) {
            elements.maeValue.textContent = metrics.mae.toFixed(4);
        }
        
        if (elements.rmseValue) {
            elements.rmseValue.textContent = metrics.rmse.toFixed(4);
        }
        
        if (elements.r2Value) {
            elements.r2Value.textContent = `${(metrics.r2 * 100).toFixed(2)}%`;
        }
        
        if (elements.directionAccuracy) {
            elements.directionAccuracy.textContent = `${(metrics.directionAccuracy * 100).toFixed(2)}%`;
        }
    }

    displayPredictionResults(predictions) {
        console.log('預測結果:', predictions);
        
        // Could add a separate table or section for predictions
        // For now, they are shown in the chart
    }

    displayStockStats() {
        if (!this.currentStock) return;

        const stats = stockDataManager.getStockStats(this.currentStock);
        if (!stats) return;

        // Update status with current stock info
        const changeIndicator = stats.change > 0 ? '↗' : stats.change < 0 ? '↘' : '→';
        const changeColor = stats.change > 0 ? 'text-green-600' : stats.change < 0 ? 'text-red-600' : 'text-gray-600';
        
        this.updateStatus(
            `${this.currentStock} 當前價格: $${stats.currentPrice} ${changeIndicator} ${stats.changePercent.toFixed(2)}%`, 
            'info'
        );
    }

    clearData() {
        // Reset selections
        this.currentStock = null;
        document.getElementById('stockSelect').value = '';
        
        // Clear table
        const tableBody = document.getElementById('stockTableBody');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="6" class="px-4 py-8 text-center text-gray-500">請選擇股票以查看資料</td></tr>';
        }

        // Clear metrics
        const metricElements = ['maeValue', 'rmseValue', 'r2Value', 'directionAccuracy'];
        metricElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.textContent = '--';
        });

        // Clear charts
        chartManager.destroyAllCharts();
        chartManager.initPriceChart();
        chartManager.initAccuracyChart();

        this.updateStatus('資料已清除', 'info');
    }

    // Load sentiment analysis data
    async loadSentimentData() {
        if (!this.currentStock || !window.sentimentAnalyzer) return;

        try {
            const sentiment = await window.sentimentAnalyzer.getStockSentiment(this.currentStock);
            this.updateSentimentDisplay(sentiment);
        } catch (error) {
            console.warn('Sentiment analysis failed:', error.message);
            this.clearSentimentDisplay();
        }
    }

    // Load financial data
    async loadFinancialData() {
        if (!this.currentStock || !window.financialAnalyzer) return;

        try {
            const financialData = await window.financialAnalyzer.getFinancialData(this.currentStock);
            this.updateFinancialDisplay(financialData);
        } catch (error) {
            console.warn('Financial analysis failed:', error.message);
            this.clearFinancialDisplay();
        }
    }

    // Update sentiment display
    updateSentimentDisplay(sentiment) {
        const elements = {
            overallSentiment: document.getElementById('overallSentiment'),
            newsCount: document.getElementById('newsCount'),
            socialVolume: document.getElementById('socialVolume'),
            recommendation: document.getElementById('recommendation')
        };

        if (elements.overallSentiment) {
            elements.overallSentiment.textContent = sentiment.sentiment_label;
            elements.overallSentiment.className = `text-2xl font-bold ${this.getSentimentColor(sentiment.overall_sentiment)}`;
        }

        if (elements.newsCount) {
            elements.newsCount.textContent = sentiment.news_count.toString();
        }

        if (elements.socialVolume) {
            elements.socialVolume.textContent = sentiment.social_volume.toString();
        }

        if (elements.recommendation && window.sentimentAnalyzer) {
            const signals = window.sentimentAnalyzer.generateTradingSignals(sentiment);
            if (signals.length > 0) {
                elements.recommendation.textContent = signals[0].signal;
                elements.recommendation.className = `text-2xl font-bold ${this.getRecommendationColor(signals[0].signal)}`;
            }
        }
    }

    // Update financial display
    updateFinancialDisplay(financialData) {
        const elements = {
            peRatio: document.getElementById('peRatio'),
            roeValue: document.getElementById('roeValue'),
            debtRatio: document.getElementById('debtRatio'),
            revenueGrowth: document.getElementById('revenueGrowth')
        };

        if (elements.peRatio) {
            elements.peRatio.textContent = financialData.pe_ratio.toFixed(1);
        }

        if (elements.roeValue) {
            elements.roeValue.textContent = `${(financialData.roe * 100).toFixed(1)}%`;
        }

        if (elements.debtRatio) {
            elements.debtRatio.textContent = financialData.debt_to_equity.toFixed(2);
        }

        if (elements.revenueGrowth) {
            elements.revenueGrowth.textContent = `${(financialData.revenue_growth * 100).toFixed(1)}%`;
            elements.revenueGrowth.className = `text-2xl font-bold ${financialData.revenue_growth >= 0 ? 'text-green-800' : 'text-red-800'}`;
        }
    }

    // Clear sentiment display
    clearSentimentDisplay() {
        const sentimentElements = ['overallSentiment', 'newsCount', 'socialVolume', 'recommendation'];
        sentimentElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.textContent = '--';
        });
    }

    // Clear financial display
    clearFinancialDisplay() {
        const financialElements = ['peRatio', 'roeValue', 'debtRatio', 'revenueGrowth'];
        financialElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.textContent = '--';
        });
    }

    // Get color class for sentiment
    getSentimentColor(sentiment) {
        if (sentiment > 0.1) return 'text-green-800';
        if (sentiment < -0.1) return 'text-red-800';
        return 'text-yellow-800';
    }

    // Get color class for recommendation
    getRecommendationColor(signal) {
        switch (signal) {
            case 'BUY': return 'text-green-800';
            case 'SELL': return 'text-red-800';
            case 'HOLD': return 'text-yellow-800';
            default: return 'text-gray-800';
        }
    }

    updateStatus(message, type = 'info') {
        const statusMessage = document.getElementById('statusMessage');
        const statusText = document.getElementById('statusText');
        
        if (!statusMessage || !statusText) return;

        statusText.textContent = message;
        statusMessage.className = 'mt-6 status-enter';

        // Remove previous type classes
        statusMessage.classList.remove('message-success', 'message-error', 'message-warning');
        
        // Add new type class
        if (type === 'success') {
            statusMessage.classList.add('message-success');
        } else if (type === 'error') {
            statusMessage.classList.add('message-error');
        } else if (type === 'warning') {
            statusMessage.classList.add('message-warning');
        } else {
            // Default info styling
            statusMessage.className = 'mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4 status-enter';
        }

        statusMessage.classList.remove('hidden');

        // Auto-hide after 5 seconds for success messages
        if (type === 'success') {
            setTimeout(() => {
                statusMessage.classList.add('hidden');
            }, 5000);
        }
    }

    setAnalyzeButtonLoading(loading) {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (!analyzeBtn) return;

        if (loading) {
            analyzeBtn.classList.add('loading');
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>分析中...';
        } else {
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-play mr-2"></i>開始分析預測';
        }
    }

    getModelName() {
        const modelNames = {
            sma: '簡單移動平均 (SMA)',
            ema: '指數移動平均 (EMA)',
            linear: '線性回歸',
            polynomial: '多項式回歸',
            xgboost: 'XGBoost 梯度提升',
            transformer: 'Transformer 注意力模型',
            lstm: 'LSTM 長短期記憶網路',
            ensemble: '集成學習模型',
            sentiment: '情緒分析增強模型'
        };
        return modelNames[this.currentModel] || this.currentModel;
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('zh-TW', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });
    }

    formatVolume(volume) {
        if (volume >= 1000000) {
            return (volume / 1000000).toFixed(1) + 'M';
        } else if (volume >= 1000) {
            return (volume / 1000).toFixed(1) + 'K';
        }
        return volume.toString();
    }

    // Export functionality
    exportData() {
        if (!this.currentStock) {
            this.updateStatus('請先選擇股票', 'warning');
            return;
        }

        const csvData = stockDataManager.exportToCSV(this.currentStock);
        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${this.currentStock}_stock_data.csv`;
        a.click();
        window.URL.revokeObjectURL(url);

        this.updateStatus('資料匯出成功', 'success');
    }

    // Error handling
    handleError(error, context = '') {
        console.error(`Error in ${context}:`, error);
        this.updateStatus(`${context}發生錯誤: ${error.message}`, 'error');
    }
}

// Application initialization
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the main application
    if (!window.app) {
        window.app = new StockPredictionApp();
    }
    
    console.log('AI股票預測分析系統已啟動');
    console.log('Available stocks:', stockDataManager.getAvailableStocks());
    console.log('Available models:', aiModelManager.getAvailableModels());
});

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    if (window.app) {
        window.app.updateStatus('系統發生未預期的錯誤', 'error');
    }
});

// Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        if (window.app) {
            window.app.updateStatus('系統發生未預期的錯誤', 'error');
        }
    });
})();