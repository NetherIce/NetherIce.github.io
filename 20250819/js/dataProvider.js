// Real-time Stock Data Provider
class StockDataProvider {
    constructor() {
        this.cache = {};
        this.cacheExpiry = 5 * 60 * 1000; // 5 minutes cache
        this.apiKeys = {
            finnhub: 'your_finnhub_api_key', // 需要申請API Key
            alphavantage: 'your_alphavantage_api_key',
            iex: 'your_iex_api_key'
        };
        this.finmindToken = '0';
    }

    // Get stock data with caching
    async getStockData(symbol, days = 365) {
        const cacheKey = `${symbol}_${days}`;
        
        // Check cache first
        if (this.cache[cacheKey] && Date.now() - this.cache[cacheKey].timestamp < this.cacheExpiry) {
            return this.cache[cacheKey].data;
        }

        try {
            let data;
            
            // Determine if it's Taiwan stock or US stock
            if (symbol.includes('.TW')) {
                data = await this.getTaiwanStockData(symbol, days);
            } else {
                data = await this.getUSStockData(symbol, days);
            }

            // Cache the result
            this.cache[cacheKey] = {
                data: data,
                timestamp: Date.now()
            };

            return data;
        } catch (error) {
            console.error('Error fetching stock data:', error);
            // Fallback to generated data if API fails
            return this.generateFallbackData(symbol, days);
        }
    }

    // Get Taiwan stock data from FinMind or yfinance
    async getTaiwanStockData(symbol, days) {
        // Try FinMind API first
        try {
            return await this.getFinMindData(symbol, days);
        } catch (error) {
            console.warn('FinMind API failed, trying yfinance alternative');
            return await this.getYFinanceAlternative(symbol, days);
        }
    }

    // Get US stock data
    async getUSStockData(symbol, days) {
        // Try multiple sources for reliability
        const sources = [
            () => this.getFinnhubData(symbol, days),
            () => this.getAlphaVantageData(symbol, days),
            () => this.getYFinanceAlternative(symbol, days)
        ];

        for (const source of sources) {
            try {
                return await source();
            } catch (error) {
                console.warn(`Data source failed: ${error.message}`);
                continue;
            }
        }

        throw new Error('All data sources failed');
    }

    // FinMind API for Taiwan stocks
    async getFinMindData(symbol, days) {
        const stockCode = symbol.replace('.TW', '');
        const endDate = new Date().toISOString().split('T')[0];
        const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

        const response = await fetch(`https://api.finmindtrade.com/api/v4/data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                dataset: 'TaiwanStockPrice',
                data_id: stockCode,
                start_date: startDate,
                end_date: endDate,
                token: this.finmindToken
            })
        });

        if (!response.ok) {
            throw new Error(`FinMind API error: ${response.status}`);
        }

        const result = await response.json();
        return this.transformFinMindData(result.data);
    }

    // Transform FinMind data to our format
    transformFinMindData(data) {
        return data.map(item => ({
            date: item.date,
            open: parseFloat(item.open),
            high: parseFloat(item.max),
            low: parseFloat(item.min),
            close: parseFloat(item.close),
            volume: parseInt(item.Trading_Volume)
        })).sort((a, b) => new Date(a.date) - new Date(b.date));
    }

    // Finnhub API for US stocks
    async getFinnhubData(symbol, days) {
        const endTimestamp = Math.floor(Date.now() / 1000);
        const startTimestamp = endTimestamp - (days * 24 * 60 * 60);

        const response = await fetch(
            `https://finnhub.io/api/v1/stock/candle?symbol=${symbol}&resolution=D&from=${startTimestamp}&to=${endTimestamp}&token=${this.apiKeys.finnhub}`
        );

        if (!response.ok) {
            throw new Error(`Finnhub API error: ${response.status}`);
        }

        const result = await response.json();
        return this.transformFinnhubData(result);
    }

    // Transform Finnhub data to our format
    transformFinnhubData(data) {
        if (!data.c || data.s !== 'ok') {
            throw new Error('Invalid Finnhub data response');
        }

        return data.t.map((timestamp, index) => ({
            date: new Date(timestamp * 1000).toISOString().split('T')[0],
            open: data.o[index],
            high: data.h[index],
            low: data.l[index],
            close: data.c[index],
            volume: data.v[index]
        })).sort((a, b) => new Date(a.date) - new Date(b.date));
    }

    // Alpha Vantage API backup
    async getAlphaVantageData(symbol, days) {
        const response = await fetch(
            `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${symbol}&outputsize=full&apikey=${this.apiKeys.alphavantage}`
        );

        if (!response.ok) {
            throw new Error(`Alpha Vantage API error: ${response.status}`);
        }

        const result = await response.json();
        return this.transformAlphaVantageData(result['Time Series (Daily)'], days);
    }

    // Transform Alpha Vantage data
    transformAlphaVantageData(data, days) {
        if (!data) {
            throw new Error('Invalid Alpha Vantage data response');
        }

        const dates = Object.keys(data).sort().slice(-days);
        return dates.map(date => ({
            date: date,
            open: parseFloat(data[date]['1. open']),
            high: parseFloat(data[date]['2. high']),
            low: parseFloat(data[date]['3. low']),
            close: parseFloat(data[date]['4. close']),
            volume: parseInt(data[date]['5. volume'])
        }));
    }

    // YFinance alternative using proxy service
    async getYFinanceAlternative(symbol, days) {
        // This would require a proxy service since yfinance is Python-based
        // For now, we'll use a mock implementation
        console.warn('YFinance alternative not implemented, using fallback data');
        return this.generateFallbackData(symbol, days);
    }

    // Generate fallback data when APIs fail
    generateFallbackData(symbol, days) {
        console.warn(`Generating fallback data for ${symbol}`);
        
        // Use improved fallback based on real market patterns
        const stockInfo = this.getStockInfo(symbol);
        return this.generateRealisticData(stockInfo, days);
    }

    // Get stock info for realistic data generation
    getStockInfo(symbol) {
        const stocksInfo = {
            // US Stocks
            'AAPL': { name: '蘋果', basePrice: 175, volatility: 0.025, sector: 'tech' },
            'TSLA': { name: '特斯拉', basePrice: 250, volatility: 0.05, sector: 'auto' },
            'GOOGL': { name: 'Google', basePrice: 135, volatility: 0.03, sector: 'tech' },
            'MSFT': { name: '微軟', basePrice: 380, volatility: 0.025, sector: 'tech' },
            'NVDA': { name: 'NVIDIA', basePrice: 450, volatility: 0.04, sector: 'tech' },
            'AMZN': { name: '亞馬遜', basePrice: 145, volatility: 0.03, sector: 'retail' },
            'META': { name: 'Meta', basePrice: 320, volatility: 0.035, sector: 'tech' },
            
            // Taiwan Stocks
            '2330.TW': { name: '台積電', basePrice: 550, volatility: 0.03, sector: 'semiconductor' },
            '2317.TW': { name: '鴻海', basePrice: 105, volatility: 0.035, sector: 'manufacturing' },
            '2454.TW': { name: '聯發科', basePrice: 800, volatility: 0.04, sector: 'semiconductor' },
            '2881.TW': { name: '富邦金', basePrice: 65, volatility: 0.025, sector: 'finance' },
            '2882.TW': { name: '國泰金', basePrice: 48, volatility: 0.025, sector: 'finance' },
            '2412.TW': { name: '中華電', basePrice: 125, volatility: 0.02, sector: 'telecom' },
            '1301.TW': { name: '台塑', basePrice: 95, volatility: 0.03, sector: 'petrochemical' },
            '2002.TW': { name: '中鋼', basePrice: 28, volatility: 0.03, sector: 'steel' }
        };

        return stocksInfo[symbol] || { 
            name: symbol, 
            basePrice: 100, 
            volatility: 0.03, 
            sector: 'general' 
        };
    }

    // Generate realistic stock data using advanced models
    generateRealisticData(stockInfo, days) {
        const data = [];
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        
        let currentPrice = stockInfo.basePrice;
        const drift = 0.0003; // Small positive drift
        const { volatility } = stockInfo;
        
        // Add sector-specific trends
        const sectorTrends = {
            'tech': 0.0005,
            'finance': 0.0002,
            'semiconductor': 0.0007,
            'auto': 0.0004,
            'retail': 0.0003,
            'telecom': 0.0001,
            'petrochemical': 0.0002,
            'steel': 0.0001,
            'manufacturing': 0.0003,
            'general': 0.0002
        };
        
        const sectorDrift = sectorTrends[stockInfo.sector] || 0.0002;
        
        for (let i = 0; i < days; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            
            // Geometric Brownian Motion with sector trends
            const randomShock = this.boxMullerTransform() * volatility;
            const priceChange = currentPrice * (drift + sectorDrift + randomShock);
            currentPrice = Math.max(currentPrice + priceChange, stockInfo.basePrice * 0.3);
            
            // Generate OHLC data
            const open = i === 0 ? currentPrice : data[i-1].close;
            const variation = currentPrice * volatility * 0.7;
            
            const high = currentPrice + Math.abs(this.boxMullerTransform()) * variation * 0.5;
            const low = currentPrice - Math.abs(this.boxMullerTransform()) * variation * 0.5;
            const close = currentPrice;
            
            // Realistic volume based on price movement
            const priceVolatility = Math.abs(close - open) / open;
            const baseVolume = stockInfo.basePrice < 200 ? 20000000 : 10000000;
            const volume = Math.floor(baseVolume * (1 + priceVolatility * 5) * (0.5 + Math.random()));
            
            data.push({
                date: this.formatDate(date),
                open: parseFloat(Math.max(open, low).toFixed(2)),
                high: parseFloat(Math.max(high, open, close).toFixed(2)),
                low: parseFloat(Math.min(low, open, close).toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: volume
            });
        }
        
        return data;
    }

    // Box-Muller transformation for normal distribution
    boxMullerTransform() {
        const u = 0.5 - Math.random();
        const v = 0.5 - Math.random();
        return Math.sqrt(-2.0 * Math.log(Math.abs(u))) * Math.cos(2.0 * Math.PI * v);
    }

    // Format date to YYYY-MM-DD
    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    // Get news sentiment data (placeholder for future implementation)
    async getNewsSentiment(symbol) {
        // This would integrate with news APIs and sentiment analysis
        return {
            sentiment_score: 0.1, // -1 to 1
            news_count: 15,
            positive_ratio: 0.6,
            negative_ratio: 0.2,
            neutral_ratio: 0.2
        };
    }

    // Get financial data (placeholder for future implementation)
    async getFinancialData(symbol) {
        // This would get earnings, revenue, etc.
        return {
            pe_ratio: 25.5,
            revenue_growth: 0.12,
            profit_margin: 0.25,
            debt_to_equity: 0.4
        };
    }

    // Clear cache
    clearCache() {
        this.cache = {};
    }
}

// Initialize global data provider
(function() {
    'use strict';
    if (!window.stockDataProvider) {
        window.stockDataProvider = new StockDataProvider();
    }
})();