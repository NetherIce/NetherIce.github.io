// Stock Data Management System
(function() {
    'use strict';
    
    class StockDataManager {
    constructor() {
        this.stockData = {};
        this.currentStock = null;
        this.init();
    }

    init() {
        // Initialize with sample data (will be replaced with real data when available)
        this.generateSampleData();
    }

    // Get real stock data using the data provider
    async getRealStockData(symbol, days = 365) {
        if (window.stockDataProvider) {
            try {
                const realData = await window.stockDataProvider.getStockData(symbol, days);
                this.stockData[symbol] = realData;
                return realData;
            } catch (error) {
                console.warn('Failed to get real data, using fallback:', error.message);
                return this.getStockData(symbol);
            }
        }
        return this.getStockData(symbol);
    }

    // Generate realistic sample stock data
    generateSampleData() {
        const stocks = {
            'AAPL': {
                name: '蘋果公司',
                symbol: 'AAPL',
                basePrice: 150,
                volatility: 0.02
            },
            'TSLA': {
                name: '特斯拉',
                symbol: 'TSLA',
                basePrice: 200,
                volatility: 0.04
            },
            'GOOGL': {
                name: 'Google',
                symbol: 'GOOGL',
                basePrice: 120,
                volatility: 0.025
            },
            'MSFT': {
                name: '微軟',
                symbol: 'MSFT',
                basePrice: 300,
                volatility: 0.02
            },
            'NVDA': {
                name: 'NVIDIA',
                symbol: 'NVDA',
                basePrice: 400,
                volatility: 0.035
            }
        };

        Object.keys(stocks).forEach(symbol => {
            this.stockData[symbol] = this.generateHistoricalData(stocks[symbol], 365);
        });
    }

    // Generate historical stock data using random walk with drift
    generateHistoricalData(stock, days) {
        const data = [];
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        
        let currentPrice = stock.basePrice;
        const drift = 0.0005; // Small positive drift
        
        for (let i = 0; i < days; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            
            // Generate price movement using geometric Brownian motion
            const randomShock = (Math.random() - 0.5) * 2; // -1 to 1
            const priceChange = currentPrice * (drift + stock.volatility * randomShock);
            currentPrice += priceChange;
            
            // Ensure price doesn't go negative
            currentPrice = Math.max(currentPrice, stock.basePrice * 0.5);
            
            // Generate OHLC data
            const open = i === 0 ? currentPrice : data[i-1].close;
            const variation = currentPrice * stock.volatility * 0.5;
            
            const high = currentPrice + Math.random() * variation;
            const low = currentPrice - Math.random() * variation;
            const close = currentPrice;
            
            // Generate volume (random but realistic)
            const volume = Math.floor(Math.random() * 50000000) + 10000000;
            
            data.push({
                date: this.formatDate(date),
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(Math.max(open, high, close).toFixed(2)),
                low: parseFloat(Math.min(open, low, close).toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: volume
            });
        }
        
        return data;
    }

    // Format date to YYYY-MM-DD
    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    // Get stock data by symbol
    getStockData(symbol) {
        return this.stockData[symbol] || [];
    }

    // Get recent stock data (last N days)
    getRecentData(symbol, days = 30) {
        const data = this.getStockData(symbol);
        return data.slice(-days);
    }

    // Get stock price for a specific date
    getStockPrice(symbol, date) {
        const data = this.getStockData(symbol);
        const record = data.find(item => item.date === date);
        return record ? record.close : null;
    }

    // Calculate simple moving average
    calculateSMA(symbol, period = 20) {
        const data = this.getStockData(symbol);
        const smaData = [];
        
        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const average = slice.reduce((sum, item) => sum + item.close, 0) / period;
            smaData.push({
                date: data[i].date,
                value: parseFloat(average.toFixed(2))
            });
        }
        
        return smaData;
    }

    // Calculate exponential moving average
    calculateEMA(symbol, period = 20) {
        const data = this.getStockData(symbol);
        if (data.length < period) return [];
        
        const emaData = [];
        const multiplier = 2 / (period + 1);
        
        // First EMA is SMA
        const firstSMA = data.slice(0, period)
            .reduce((sum, item) => sum + item.close, 0) / period;
        
        emaData.push({
            date: data[period - 1].date,
            value: parseFloat(firstSMA.toFixed(2))
        });
        
        // Calculate subsequent EMAs
        for (let i = period; i < data.length; i++) {
            const currentPrice = data[i].close;
            const previousEMA = emaData[emaData.length - 1].value;
            const ema = (currentPrice * multiplier) + (previousEMA * (1 - multiplier));
            
            emaData.push({
                date: data[i].date,
                value: parseFloat(ema.toFixed(2))
            });
        }
        
        return emaData;
    }

    // Get stock statistics
    getStockStats(symbol) {
        const data = this.getStockData(symbol);
        if (data.length === 0) return null;
        
        const prices = data.map(item => item.close);
        const volumes = data.map(item => item.volume);
        
        const currentPrice = prices[prices.length - 1];
        const previousPrice = prices[prices.length - 2];
        const change = currentPrice - previousPrice;
        const changePercent = (change / previousPrice) * 100;
        
        const high52Week = Math.max(...prices.slice(-252)); // 252 trading days in a year
        const low52Week = Math.min(...prices.slice(-252));
        
        const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
        
        return {
            currentPrice: parseFloat(currentPrice.toFixed(2)),
            change: parseFloat(change.toFixed(2)),
            changePercent: parseFloat(changePercent.toFixed(2)),
            high52Week: parseFloat(high52Week.toFixed(2)),
            low52Week: parseFloat(low52Week.toFixed(2)),
            avgVolume: Math.floor(avgVolume),
            totalRecords: data.length
        };
    }

    // Add new stock data point
    addDataPoint(symbol, dataPoint) {
        if (!this.stockData[symbol]) {
            this.stockData[symbol] = [];
        }
        this.stockData[symbol].push(dataPoint);
    }

    // Update stock data for a specific date
    updateDataPoint(symbol, date, updates) {
        const data = this.stockData[symbol];
        if (!data) return false;
        
        const index = data.findIndex(item => item.date === date);
        if (index !== -1) {
            this.stockData[symbol][index] = { ...data[index], ...updates };
            return true;
        }
        return false;
    }

    // Get available stock symbols
    getAvailableStocks() {
        return Object.keys(this.stockData);
    }

    // Calculate price volatility
    calculateVolatility(symbol, period = 30) {
        const data = this.getRecentData(symbol, period);
        if (data.length < 2) return 0;
        
        const returns = [];
        for (let i = 1; i < data.length; i++) {
            const returnRate = Math.log(data[i].close / data[i-1].close);
            returns.push(returnRate);
        }
        
        const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
        const volatility = Math.sqrt(variance) * Math.sqrt(252); // Annualized
        
        return parseFloat((volatility * 100).toFixed(2));
    }

    // Export data to CSV format
    exportToCSV(symbol) {
        const data = this.getStockData(symbol);
        if (data.length === 0) return '';
        
        const headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'];
        const csvRows = [headers.join(',')];
        
        data.forEach(row => {
            const values = [row.date, row.open, row.high, row.low, row.close, row.volume];
            csvRows.push(values.join(','));
        });
        
        return csvRows.join('\n');
    }
}

    // Initialize global stock data manager
    if (!window.stockDataManager) {
        window.stockDataManager = new StockDataManager();
    }
})();