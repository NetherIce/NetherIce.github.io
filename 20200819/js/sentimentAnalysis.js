// News Sentiment Analysis Integration
(function() {
    'use strict';

    class SentimentAnalyzer {
        constructor() {
            this.apiKeys = {
                newsapi: 'your_newsapi_key',
                finnhub: 'your_finnhub_key',
                alpaca: 'your_alpaca_key'
            };
            this.cache = {};
            this.cacheExpiry = 60 * 60 * 1000; // 1 hour cache
        }

        // Get news sentiment for a stock
        async getStockSentiment(symbol) {
            const cacheKey = `sentiment_${symbol}`;
            
            // Check cache
            if (this.cache[cacheKey] && Date.now() - this.cache[cacheKey].timestamp < this.cacheExpiry) {
                return this.cache[cacheKey].data;
            }

            try {
                const [newsData, socialData] = await Promise.all([
                    this.getNewsData(symbol),
                    this.getSocialMediaData(symbol)
                ]);

                const sentiment = this.analyzeCombinedSentiment(newsData, socialData);
                
                // Cache result
                this.cache[cacheKey] = {
                    data: sentiment,
                    timestamp: Date.now()
                };

                return sentiment;
            } catch (error) {
                console.warn('Sentiment analysis failed, using neutral:', error.message);
                return this.getNeutralSentiment();
            }
        }

        // Get news data from various sources
        async getNewsData(symbol) {
            const companyName = this.getCompanyName(symbol);
            const newsData = [];

            // Try multiple news sources
            try {
                // NewsAPI
                const newsApiData = await this.fetchNewsAPI(companyName);
                newsData.push(...newsApiData);
            } catch (error) {
                console.warn('NewsAPI failed:', error.message);
            }

            try {
                // Finnhub News
                const finnhubData = await this.fetchFinnhubNews(symbol);
                newsData.push(...finnhubData);
            } catch (error) {
                console.warn('Finnhub News failed:', error.message);
            }

            return newsData;
        }

        // Fetch news from NewsAPI
        async fetchNewsAPI(companyName) {
            const response = await fetch(
                `https://newsapi.org/v2/everything?q=${encodeURIComponent(companyName)}&language=en&sortBy=publishedAt&pageSize=20&apiKey=${this.apiKeys.newsapi}`
            );

            if (!response.ok) {
                throw new Error(`NewsAPI error: ${response.status}`);
            }

            const data = await response.json();
            return data.articles.map(article => ({
                title: article.title,
                description: article.description,
                content: article.content,
                publishedAt: article.publishedAt,
                source: 'NewsAPI',
                url: article.url
            }));
        }

        // Fetch news from Finnhub
        async fetchFinnhubNews(symbol) {
            const today = new Date().toISOString().split('T')[0];
            const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

            const response = await fetch(
                `https://finnhub.io/api/v1/company-news?symbol=${symbol}&from=${weekAgo}&to=${today}&token=${this.apiKeys.finnhub}`
            );

            if (!response.ok) {
                throw new Error(`Finnhub News error: ${response.status}`);
            }

            const data = await response.json();
            return data.map(article => ({
                title: article.headline,
                description: article.summary,
                content: article.summary,
                publishedAt: new Date(article.datetime * 1000).toISOString(),
                source: 'Finnhub',
                url: article.url
            }));
        }

        // Get social media sentiment (placeholder)
        async getSocialMediaData(symbol) {
            // This would integrate with Twitter API, Reddit API, etc.
            // For now, return mock data
            return {
                twitter_sentiment: Math.random() * 2 - 1, // -1 to 1
                reddit_sentiment: Math.random() * 2 - 1,
                social_volume: Math.floor(Math.random() * 1000) + 100,
                social_mentions: Math.floor(Math.random() * 500) + 50
            };
        }

        // Analyze sentiment from text using simple keyword-based approach
        analyzeSentiment(text) {
            if (!text) return 0;

            const positiveWords = [
                'buy', 'bull', 'bullish', 'growth', 'gain', 'profit', 'up', 'rise', 'surge',
                'strong', 'positive', 'good', 'great', 'excellent', 'outstanding', 'beat',
                'exceed', 'outperform', 'rally', 'momentum', 'boost', 'upgrade', 'recommend'
            ];

            const negativeWords = [
                'sell', 'bear', 'bearish', 'loss', 'down', 'fall', 'drop', 'decline',
                'weak', 'negative', 'bad', 'poor', 'terrible', 'disappointing', 'miss',
                'underperform', 'crash', 'plunge', 'downgrade', 'warning', 'concern', 'risk'
            ];

            const words = text.toLowerCase().split(/\s+/);
            let score = 0;

            words.forEach(word => {
                if (positiveWords.includes(word)) score += 1;
                if (negativeWords.includes(word)) score -= 1;
            });

            // Normalize score
            return Math.max(-1, Math.min(1, score / Math.max(words.length / 10, 1)));
        }

        // Combine multiple sentiment sources
        analyzeCombinedSentiment(newsData, socialData) {
            let totalSentiment = 0;
            let sentimentCount = 0;
            let newsDetails = [];

            // Analyze news sentiment
            newsData.forEach(article => {
                const titleSentiment = this.analyzeSentiment(article.title);
                const descSentiment = this.analyzeSentiment(article.description);
                const avgSentiment = (titleSentiment + descSentiment) / 2;

                totalSentiment += avgSentiment;
                sentimentCount++;

                newsDetails.push({
                    title: article.title,
                    sentiment: avgSentiment,
                    publishedAt: article.publishedAt,
                    source: article.source,
                    url: article.url
                });
            });

            // Add social media sentiment
            if (socialData.twitter_sentiment !== undefined) {
                totalSentiment += socialData.twitter_sentiment * 0.3;
                sentimentCount += 0.3;
            }
            if (socialData.reddit_sentiment !== undefined) {
                totalSentiment += socialData.reddit_sentiment * 0.2;
                sentimentCount += 0.2;
            }

            const overallSentiment = sentimentCount > 0 ? totalSentiment / sentimentCount : 0;

            // Categorize sentiment
            let sentimentLabel = 'Neutral';
            if (overallSentiment > 0.1) sentimentLabel = 'Positive';
            else if (overallSentiment < -0.1) sentimentLabel = 'Negative';

            return {
                overall_sentiment: parseFloat(overallSentiment.toFixed(3)),
                sentiment_label: sentimentLabel,
                confidence: Math.min(sentimentCount / 10, 1), // Higher confidence with more data
                news_count: newsData.length,
                social_volume: socialData.social_volume || 0,
                news_details: newsDetails.slice(0, 10), // Top 10 news items
                analysis_timestamp: new Date().toISOString(),
                sources: {
                    news_sentiment: newsData.length > 0 ? totalSentiment / newsData.length : 0,
                    social_sentiment: (socialData.twitter_sentiment + socialData.reddit_sentiment) / 2
                }
            };
        }

        // Get neutral sentiment as fallback
        getNeutralSentiment() {
            return {
                overall_sentiment: 0,
                sentiment_label: 'Neutral',
                confidence: 0.5,
                news_count: 0,
                social_volume: 0,
                news_details: [],
                analysis_timestamp: new Date().toISOString(),
                sources: {
                    news_sentiment: 0,
                    social_sentiment: 0
                }
            };
        }

        // Get company name from stock symbol
        getCompanyName(symbol) {
            const companies = {
                'AAPL': 'Apple Inc',
                'TSLA': 'Tesla Inc',
                'GOOGL': 'Google Alphabet',
                'MSFT': 'Microsoft Corporation',
                'NVDA': 'NVIDIA Corporation',
                'AMZN': 'Amazon.com Inc',
                'META': 'Meta Platforms Inc',
                '2330.TW': '台積電 TSMC',
                '2317.TW': '鴻海精密 Foxconn',
                '2454.TW': '聯發科 MediaTek',
                '2881.TW': '富邦金控',
                '2882.TW': '國泰金控',
                '2412.TW': '中華電信',
                '1301.TW': '台塑企業',
                '2002.TW': '中國鋼鐵'
            };

            return companies[symbol] || symbol;
        }

        // Generate sentiment-based trading signals
        generateTradingSignals(sentiment) {
            const signals = [];

            if (sentiment.overall_sentiment > 0.3) {
                signals.push({
                    signal: 'BUY',
                    strength: 'Strong',
                    reason: 'Very positive sentiment detected',
                    confidence: sentiment.confidence
                });
            } else if (sentiment.overall_sentiment > 0.1) {
                signals.push({
                    signal: 'BUY',
                    strength: 'Moderate',
                    reason: 'Positive sentiment detected',
                    confidence: sentiment.confidence
                });
            } else if (sentiment.overall_sentiment < -0.3) {
                signals.push({
                    signal: 'SELL',
                    strength: 'Strong',
                    reason: 'Very negative sentiment detected',
                    confidence: sentiment.confidence
                });
            } else if (sentiment.overall_sentiment < -0.1) {
                signals.push({
                    signal: 'SELL',
                    strength: 'Moderate',
                    reason: 'Negative sentiment detected',
                    confidence: sentiment.confidence
                });
            } else {
                signals.push({
                    signal: 'HOLD',
                    strength: 'Neutral',
                    reason: 'Neutral sentiment, no clear direction',
                    confidence: sentiment.confidence
                });
            }

            return signals;
        }

        // Clear sentiment cache
        clearCache() {
            this.cache = {};
        }
    }

    // Financial Data Analyzer
    class FinancialDataAnalyzer {
        constructor() {
            this.cache = {};
            this.cacheExpiry = 24 * 60 * 60 * 1000; // 24 hours cache
        }

        // Get fundamental financial data
        async getFinancialData(symbol) {
            const cacheKey = `financial_${symbol}`;
            
            if (this.cache[cacheKey] && Date.now() - this.cache[cacheKey].timestamp < this.cacheExpiry) {
                return this.cache[cacheKey].data;
            }

            try {
                const financialData = await this.fetchFinancialData(symbol);
                
                this.cache[cacheKey] = {
                    data: financialData,
                    timestamp: Date.now()
                };

                return financialData;
            } catch (error) {
                console.warn('Financial data fetch failed:', error.message);
                return this.getMockFinancialData(symbol);
            }
        }

        // Fetch real financial data (placeholder)
        async fetchFinancialData(symbol) {
            // This would integrate with financial APIs like Alpha Vantage, IEX, etc.
            throw new Error('Real financial API not configured');
        }

        // Generate mock financial data
        getMockFinancialData(symbol) {
            return {
                pe_ratio: 15 + Math.random() * 30,
                pb_ratio: 1 + Math.random() * 5,
                debt_to_equity: Math.random() * 2,
                roe: 0.05 + Math.random() * 0.25,
                revenue_growth: -0.1 + Math.random() * 0.4,
                profit_margin: 0.05 + Math.random() * 0.25,
                current_ratio: 1 + Math.random() * 2,
                quick_ratio: 0.5 + Math.random() * 1.5,
                earnings_growth: -0.2 + Math.random() * 0.5,
                dividend_yield: Math.random() * 0.06,
                market_cap: 1e9 + Math.random() * 1e12,
                analyst_rating: Math.random() * 5,
                price_target: 100 + Math.random() * 200
            };
        }

        // Analyze financial health
        analyzeFinancialHealth(financialData) {
            let score = 0;
            let factors = [];

            // P/E Ratio analysis
            if (financialData.pe_ratio < 15) {
                score += 2;
                factors.push('Low P/E ratio indicates undervaluation');
            } else if (financialData.pe_ratio > 30) {
                score -= 1;
                factors.push('High P/E ratio may indicate overvaluation');
            }

            // Debt analysis
            if (financialData.debt_to_equity < 0.5) {
                score += 1;
                factors.push('Low debt-to-equity ratio shows financial stability');
            } else if (financialData.debt_to_equity > 1.5) {
                score -= 2;
                factors.push('High debt levels pose financial risk');
            }

            // Profitability
            if (financialData.roe > 0.15) {
                score += 2;
                factors.push('Strong return on equity');
            } else if (financialData.roe < 0.05) {
                score -= 1;
                factors.push('Low return on equity');
            }

            // Growth
            if (financialData.revenue_growth > 0.1) {
                score += 2;
                factors.push('Strong revenue growth');
            } else if (financialData.revenue_growth < 0) {
                score -= 2;
                factors.push('Negative revenue growth is concerning');
            }

            return {
                score: score,
                rating: this.getHealthRating(score),
                factors: factors,
                recommendation: this.getRecommendation(score)
            };
        }

        getHealthRating(score) {
            if (score >= 5) return 'Excellent';
            if (score >= 3) return 'Good';
            if (score >= 0) return 'Fair';
            if (score >= -3) return 'Poor';
            return 'Very Poor';
        }

        getRecommendation(score) {
            if (score >= 4) return 'Strong Buy';
            if (score >= 2) return 'Buy';
            if (score >= -1) return 'Hold';
            if (score >= -3) return 'Sell';
            return 'Strong Sell';
        }
    }

    // Initialize global instances
    if (!window.sentimentAnalyzer) {
        window.sentimentAnalyzer = new SentimentAnalyzer();
    }
    
    if (!window.financialAnalyzer) {
        window.financialAnalyzer = new FinancialDataAnalyzer();
    }

    console.log('Sentiment analysis and financial data modules loaded');

})();