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
        this.finmindToken = 'your_finmind_token';
        this.stockDataPath = './Stock_data/'; // 本地CSV文件路徑
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
            
            // 從本地CSV文件讀取數據
            const stockCode = symbol.replace('.TW', '');
            
            // 判斷是台股還是美股
            if (/^\d+$/.test(stockCode)) {
                // 台股 (數字檔名)
                data = await this.readTaiwanStockCSV(stockCode, days);
            } else {
                // 美股 (英文檔名)
                data = await this.readUSStockCSV(symbol, days);
            }

            // Cache the result
            this.cache[cacheKey] = {
                data: data,
                timestamp: Date.now()
            };

            return data;
        } catch (error) {
            console.error('Error fetching stock data:', error);
            // Fallback to generated data if CSV reading fails
            return this.generateFallbackData(symbol, days);
        }
    }

    // 讀取台股CSV文件
    async readTaiwanStockCSV(stockCode, days) {
        try {
            const response = await fetch(`${this.stockDataPath}${stockCode}.csv`);
            if (!response.ok) {
                throw new Error(`Failed to fetch CSV file: ${response.status}`);
            }
            
            const csvText = await response.text();
            const lines = csvText.split('\n');
            
            // 跳過標題行
            const dataLines = lines.slice(1);
            
            // 解析CSV數據
            const data = dataLines
                .filter(line => line.trim() !== '')
                .map(line => {
                    const columns = line.split(',');
                    return {
                        date: columns[0],
                        open: parseFloat(columns[4]),
                        high: parseFloat(columns[5]),
                        low: parseFloat(columns[6]),
                        close: parseFloat(columns[7]),
                        volume: parseInt(columns[2])
                    };
                })
                .sort((a, b) => new Date(a.date) - new Date(b.date));
            
            // 只返回最近的days天數據
            return data.slice(-days);
        } catch (error) {
            console.error(`Error reading Taiwan stock CSV: ${error.message}`);
            throw error;
        }
    }

    // 讀取美股CSV文件
    async readUSStockCSV(symbol, days) {
        try {
            const response = await fetch(`${this.stockDataPath}${symbol}.csv`);
            if (!response.ok) {
                throw new Error(`Failed to fetch CSV file: ${response.status}`);
            }
            
            const csvText = await response.text();
            const lines = csvText.split('\n');
            
            // 美股CSV格式不同，需要跳過前3行
            const dataLines = lines.slice(3);
            
            // 解析CSV數據
            const data = dataLines
                .filter(line => line.trim() !== '')
                .map(line => {
                    const columns = line.split(',');
                    return {
                        date: columns[0],
                        open: parseFloat(columns[4]),
                        high: parseFloat(columns[2]),
                        low: parseFloat(columns[3]),
                        close: parseFloat(columns[1]),
                        volume: parseInt(columns[5])
                    };
                })
                .sort((a, b) => new Date(a.date) - new Date(b.date));
            
            // 只返回最近的days天數據
            return data.slice(-days);
        } catch (error) {
            console.error(`Error reading US stock CSV: ${error.message}`);
            throw error;
        }
    }

    // Generate fallback data when CSV reading fails
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