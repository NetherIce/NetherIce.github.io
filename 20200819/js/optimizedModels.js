// Optimized AI Models with Enhanced Accuracy and Direction Prediction
(function() {
    'use strict';

    // Enhanced Simple Moving Average Model
    class OptimizedSMAModel {
        constructor(period = 20) {
            this.period = period;
            this.trainedData = [];
            this.adaptivePeriod = period;
        }

        async train(stockData) {
            this.trainedData = stockData;
            // Optimize period based on volatility
            this.adaptivePeriod = this.optimizePeriod(stockData);
            return true;
        }

        optimizePeriod(data) {
            // Calculate optimal period based on volatility
            const volatility = this.calculateVolatility(data.slice(-60));
            if (volatility > 0.04) return Math.max(10, this.period - 5); // High volatility: shorter period
            if (volatility < 0.02) return Math.min(50, this.period + 10); // Low volatility: longer period
            return this.period;
        }

        calculateVolatility(data) {
            if (data.length < 2) return 0.03;
            const returns = [];
            for (let i = 1; i < data.length; i++) {
                returns.push(Math.log(data[i].close / data[i-1].close));
            }
            const mean = returns.reduce((a, b) => a + b) / returns.length;
            const variance = returns.reduce((acc, r) => acc + Math.pow(r - mean, 2), 0) / returns.length;
            return Math.sqrt(variance * 252); // Annualized
        }

        predict(days) {
            if (this.trainedData.length < this.adaptivePeriod) {
                return [];
            }

            const predictions = [];
            const recent = this.trainedData.slice(-this.adaptivePeriod);
            let currentSMA = recent.reduce((sum, item) => sum + item.close, 0) / this.adaptivePeriod;
            
            // Calculate trend and momentum
            const trend = this.calculateTrend(recent);
            const momentum = this.calculateMomentum(recent);
            
            const lastDate = new Date(this.trainedData[this.trainedData.length - 1].date);
            let lastPrice = this.trainedData[this.trainedData.length - 1].close;
            
            for (let i = 1; i <= days; i++) {
                const predictionDate = new Date(lastDate);
                predictionDate.setDate(predictionDate.getDate() + i);
                
                // Enhanced prediction with trend and momentum
                const trendAdjustment = trend * i * 0.5;
                const momentumAdjustment = momentum * Math.exp(-i * 0.1);
                const volatilityAdjustment = (Math.random() - 0.5) * lastPrice * 0.01;
                
                const predictedPrice = currentSMA + trendAdjustment + momentumAdjustment + volatilityAdjustment;
                
                predictions.push({
                    date: this.formatDate(predictionDate),
                    price: parseFloat(Math.max(predictedPrice, lastPrice * 0.5).toFixed(2)),
                    confidence: Math.max(0.85 - (i * 0.03), 0.4)
                });
                
                // Update SMA for next prediction (simplified)
                currentSMA = predictedPrice;
                lastPrice = predictedPrice;
            }
            
            return predictions;
        }

        calculateTrend(data) {
            if (data.length < 3) return 0;
            const prices = data.map(d => d.close);
            const n = prices.length;
            const x = Array.from({length: n}, (_, i) => i);
            
            // Linear regression for trend
            const sumX = x.reduce((a, b) => a + b);
            const sumY = prices.reduce((a, b) => a + b);
            const sumXY = x.reduce((acc, xi, i) => acc + xi * prices[i], 0);
            const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
            
            return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        }

        calculateMomentum(data) {
            if (data.length < 5) return 0;
            const recent = data.slice(-5);
            const older = data.slice(-10, -5);
            
            const recentAvg = recent.reduce((sum, d) => sum + d.close, 0) / recent.length;
            const olderAvg = older.reduce((sum, d) => sum + d.close, 0) / older.length;
            
            return (recentAvg - olderAvg) / olderAvg;
        }

        formatDate(date) {
            return date.toISOString().split('T')[0];
        }
    }

    // Enhanced Exponential Moving Average Model
    class OptimizedEMAModel {
        constructor(period = 20) {
            this.period = period;
            this.alpha = 2 / (period + 1);
            this.trainedData = [];
            this.emaHistory = [];
        }

        async train(stockData) {
            this.trainedData = stockData;
            this.calculateEMAHistory();
            return true;
        }

        calculateEMAHistory() {
            if (this.trainedData.length < this.period) return;
            
            this.emaHistory = [];
            let ema = this.trainedData.slice(0, this.period)
                .reduce((sum, item) => sum + item.close, 0) / this.period;
            
            this.emaHistory.push(ema);
            
            for (let i = this.period; i < this.trainedData.length; i++) {
                ema = (this.trainedData[i].close * this.alpha) + (ema * (1 - this.alpha));
                this.emaHistory.push(ema);
            }
        }

        predict(days) {
            if (this.emaHistory.length === 0) return [];

            const predictions = [];
            let currentEMA = this.emaHistory[this.emaHistory.length - 1];
            
            // Calculate EMA velocity and acceleration
            const velocity = this.calculateEMAVelocity();
            const acceleration = this.calculateEMAAcceleration();
            
            const lastDate = new Date(this.trainedData[this.trainedData.length - 1].date);
            
            for (let i = 1; i <= days; i++) {
                const predictionDate = new Date(lastDate);
                predictionDate.setDate(predictionDate.getDate() + i);
                
                // Enhanced prediction with velocity and acceleration
                const velocityEffect = velocity * i;
                const accelerationEffect = acceleration * i * i * 0.5;
                const dampening = Math.exp(-i * 0.05); // Reduce confidence over time
                
                const predictedPrice = currentEMA + (velocityEffect + accelerationEffect) * dampening;
                const noise = (Math.random() - 0.5) * predictedPrice * 0.008;
                
                const finalPrice = Math.max(predictedPrice + noise, currentEMA * 0.7);
                
                predictions.push({
                    date: this.formatDate(predictionDate),
                    price: parseFloat(finalPrice.toFixed(2)),
                    confidence: Math.max(0.88 - (i * 0.025), 0.45)
                });
                
                // Update EMA for next prediction
                currentEMA = (finalPrice * this.alpha) + (currentEMA * (1 - this.alpha));
            }
            
            return predictions;
        }

        calculateEMAVelocity() {
            if (this.emaHistory.length < 5) return 0;
            const recent = this.emaHistory.slice(-5);
            return (recent[recent.length - 1] - recent[0]) / recent.length;
        }

        calculateEMAAcceleration() {
            if (this.emaHistory.length < 10) return 0;
            const velocities = [];
            for (let i = 5; i < this.emaHistory.length; i += 5) {
                const segment = this.emaHistory.slice(i-5, i);
                const velocity = (segment[segment.length - 1] - segment[0]) / segment.length;
                velocities.push(velocity);
            }
            if (velocities.length < 2) return 0;
            return velocities[velocities.length - 1] - velocities[velocities.length - 2];
        }

        formatDate(date) {
            return date.toISOString().split('T')[0];
        }
    }

    // Enhanced Linear Regression Model
    class OptimizedLinearRegressionModel {
        constructor() {
            this.coefficients = [];
            this.trainedData = [];
            this.features = [];
            this.rsquared = 0;
        }

        async train(stockData) {
            this.trainedData = stockData;
            if (stockData.length < 30) return false;
            
            // Use more data for better training
            const trainingData = stockData.slice(-Math.min(120, stockData.length));
            this.features = this.extractFeatures(trainingData);
            
            // Multiple linear regression with regularization
            this.coefficients = this.fitMultipleRegression(this.features);
            this.rsquared = this.calculateRSquared();
            
            return true;
        }

        extractFeatures(data) {
            const features = [];
            const targets = [];
            
            for (let i = 10; i < data.length - 1; i++) {
                const feature = [];
                
                // Time-based features
                feature.push(i); // Time trend
                feature.push(Math.sin(2 * Math.PI * i / 252)); // Annual cycle
                feature.push(Math.sin(2 * Math.PI * i / 21)); // Monthly cycle
                
                // Price-based features
                feature.push(data[i].close);
                feature.push((data[i].close - data[i-5].close) / data[i-5].close); // 5-day return
                feature.push((data[i].close - data[i-10].close) / data[i-10].close); // 10-day return
                
                // Technical indicators
                const sma5 = data.slice(i-4, i+1).reduce((sum, d) => sum + d.close, 0) / 5;
                const sma10 = data.slice(i-9, i+1).reduce((sum, d) => sum + d.close, 0) / 10;
                feature.push(data[i].close / sma5); // Price to SMA5 ratio
                feature.push(data[i].close / sma10); // Price to SMA10 ratio
                feature.push(sma5 / sma10); // SMA5 to SMA10 ratio
                
                // Volatility features
                const returns = [];
                for (let j = i-9; j <= i; j++) {
                    if (j > 0) returns.push(Math.log(data[j].close / data[j-1].close));
                }
                const volatility = Math.sqrt(returns.reduce((acc, r) => acc + r*r, 0) / returns.length);
                feature.push(volatility);
                
                features.push(feature);
                targets.push(data[i + 1].close);
            }
            
            return { features, targets };
        }

        fitMultipleRegression(data) {
            const { features, targets } = data;
            const n = features.length;
            const p = features[0].length;
            
            // Add intercept term
            const X = features.map(f => [1, ...f]);
            const y = targets;
            
            // Normal equation: β = (X'X)^(-1)X'y
            const XtX = this.matrixMultiply(this.transpose(X), X);
            const XtXInv = this.matrixInverse(XtX);
            const Xty = this.matrixVectorMultiply(this.transpose(X), y);
            
            return this.matrixVectorMultiply(XtXInv, Xty);
        }

        predict(days) {
            if (this.coefficients.length === 0) return [];

            const predictions = [];
            const lastDataIndex = this.trainedData.length - 1;
            const lastDate = new Date(this.trainedData[lastDataIndex].date);
            
            for (let i = 1; i <= days; i++) {
                const predictionDate = new Date(lastDate);
                predictionDate.setDate(predictionDate.getDate() + i);
                
                // Extract features for prediction
                const futureIndex = lastDataIndex + i;
                const features = this.extractPredictionFeatures(futureIndex, i);
                
                // Make prediction
                let prediction = this.coefficients[0]; // Intercept
                for (let j = 0; j < features.length; j++) {
                    prediction += this.coefficients[j + 1] * features[j];
                }
                
                // Add confidence-based noise
                const confidence = Math.max(0.75 - i * 0.02, 0.3);
                const noise = (Math.random() - 0.5) * prediction * (1 - confidence) * 0.1;
                prediction += noise;
                
                // Ensure reasonable price bounds
                const lastPrice = predictions.length > 0 ? predictions[predictions.length - 1].price : this.trainedData[lastDataIndex].close;
                prediction = Math.max(prediction, lastPrice * 0.5);
                prediction = Math.min(prediction, lastPrice * 2);
                
                predictions.push({
                    date: this.formatDate(predictionDate),
                    price: parseFloat(prediction.toFixed(2)),
                    confidence: confidence
                });
            }
            
            return predictions;
        }

        extractPredictionFeatures(futureIndex, daysAhead) {
            const lastData = this.trainedData.slice(-10);
            const lastPrice = lastData[lastData.length - 1].close;
            
            const features = [];
            
            // Time-based features
            features.push(futureIndex);
            features.push(Math.sin(2 * Math.PI * futureIndex / 252));
            features.push(Math.sin(2 * Math.PI * futureIndex / 21));
            
            // Use last known price as baseline
            features.push(lastPrice);
            
            // Approximate returns (decay with time)
            const decayFactor = Math.exp(-daysAhead * 0.1);
            const return5d = (lastData[lastData.length - 1].close - lastData[lastData.length - 5].close) / lastData[lastData.length - 5].close;
            const return10d = (lastData[lastData.length - 1].close - lastData[0].close) / lastData[0].close;
            features.push(return5d * decayFactor);
            features.push(return10d * decayFactor);
            
            // Technical indicators
            const sma5 = lastData.slice(-5).reduce((sum, d) => sum + d.close, 0) / 5;
            const sma10 = lastData.reduce((sum, d) => sum + d.close, 0) / 10;
            features.push(lastPrice / sma5);
            features.push(lastPrice / sma10);
            features.push(sma5 / sma10);
            
            // Volatility
            const returns = [];
            for (let i = 1; i < lastData.length; i++) {
                returns.push(Math.log(lastData[i].close / lastData[i-1].close));
            }
            const volatility = Math.sqrt(returns.reduce((acc, r) => acc + r*r, 0) / returns.length);
            features.push(volatility);
            
            return features;
        }

        // Matrix operations
        transpose(matrix) {
            return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
        }

        matrixMultiply(a, b) {
            const result = [];
            for (let i = 0; i < a.length; i++) {
                result[i] = [];
                for (let j = 0; j < b[0].length; j++) {
                    let sum = 0;
                    for (let k = 0; k < b.length; k++) {
                        sum += a[i][k] * b[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }

        matrixVectorMultiply(matrix, vector) {
            return matrix.map(row => row.reduce((sum, val, i) => sum + val * vector[i], 0));
        }

        matrixInverse(matrix) {
            // Simplified inverse for small matrices (Gauss-Jordan elimination)
            const n = matrix.length;
            const augmented = matrix.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]);
            
            // Forward elimination
            for (let i = 0; i < n; i++) {
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                        maxRow = k;
                    }
                }
                [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
                
                for (let k = i + 1; k < n; k++) {
                    const factor = augmented[k][i] / augmented[i][i];
                    for (let j = i; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
            
            // Back substitution
            for (let i = n - 1; i >= 0; i--) {
                for (let k = i - 1; k >= 0; k--) {
                    const factor = augmented[k][i] / augmented[i][i];
                    for (let j = i; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
            
            // Normalize
            for (let i = 0; i < n; i++) {
                const divisor = augmented[i][i];
                for (let j = 0; j < 2 * n; j++) {
                    augmented[i][j] /= divisor;
                }
            }
            
            return augmented.map(row => row.slice(n));
        }

        calculateRSquared() {
            if (this.features.features.length === 0) return 0;
            
            const predictions = this.features.features.map(f => {
                let pred = this.coefficients[0];
                for (let i = 0; i < f.length; i++) {
                    pred += this.coefficients[i + 1] * f[i];
                }
                return pred;
            });
            
            const actual = this.features.targets;
            const mean = actual.reduce((a, b) => a + b) / actual.length;
            
            const totalSumSquares = actual.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
            const residualSumSquares = actual.reduce((acc, val, i) => acc + Math.pow(val - predictions[i], 2), 0);
            
            return totalSumSquares === 0 ? 0 : 1 - (residualSumSquares / totalSumSquares);
        }

        formatDate(date) {
            return date.toISOString().split('T')[0];
        }
    }

    // Enhanced Polynomial Regression Model
    class OptimizedPolynomialRegressionModel {
        constructor(degree = 3) {
            this.degree = degree;
            this.coefficients = [];
            this.trainedData = [];
            this.scaleParams = { mean: 0, std: 1 };
        }

        async train(stockData) {
            this.trainedData = stockData;
            if (stockData.length < this.degree * 10) return false;
            
            const trainingData = stockData.slice(-Math.min(180, stockData.length));
            const { x, y } = this.prepareData(trainingData);
            
            // Normalize features
            this.scaleParams = this.calculateScaleParams(x);
            const xScaled = this.scaleFeatures(x);
            
            this.coefficients = this.fitPolynomial(xScaled, y);
            return true;
        }

        prepareData(data) {
            const x = [];
            const y = [];
            
            for (let i = 0; i < data.length - 1; i++) {
                x.push(i);
                y.push(data[i + 1].close);
            }
            
            return { x, y };
        }

        calculateScaleParams(x) {
            const mean = x.reduce((a, b) => a + b) / x.length;
            const variance = x.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / x.length;
            const std = Math.sqrt(variance);
            return { mean, std: std || 1 };
        }

        scaleFeatures(x) {
            return x.map(val => (val - this.scaleParams.mean) / this.scaleParams.std);
        }

        fitPolynomial(x, y) {
            const n = x.length;
            const degree = this.degree;
            
            // Create design matrix
            const X = x.map(xi => {
                const row = [];
                for (let d = 0; d <= degree; d++) {
                    row.push(Math.pow(xi, d));
                }
                return row;
            });
            
            // Normal equation with regularization
            const XtX = this.matrixMultiply(this.transpose(X), X);
            
            // Add regularization to diagonal
            const lambda = 0.01;
            for (let i = 0; i < XtX.length; i++) {
                XtX[i][i] += lambda;
            }
            
            const XtXInv = this.matrixInverse(XtX);
            const Xty = this.matrixVectorMultiply(this.transpose(X), y);
            
            return this.matrixVectorMultiply(XtXInv, Xty);
        }

        predict(days) {
            if (this.coefficients.length === 0) return [];

            const predictions = [];
            const lastDate = new Date(this.trainedData[this.trainedData.length - 1].date);
            const startX = this.trainedData.length - 1;
            
            for (let i = 1; i <= days; i++) {
                const predictionDate = new Date(lastDate);
                predictionDate.setDate(predictionDate.getDate() + i);
                
                const x = startX + i;
                const xScaled = (x - this.scaleParams.mean) / this.scaleParams.std;
                
                // Polynomial prediction
                let prediction = 0;
                for (let d = 0; d <= this.degree; d++) {
                    prediction += this.coefficients[d] * Math.pow(xScaled, d);
                }
                
                // Add trend damping for longer predictions
                const dampening = Math.exp(-i * 0.02);
                const baseline = this.trainedData[this.trainedData.length - 1].close;
                prediction = baseline + (prediction - baseline) * dampening;
                
                // Add realistic noise
                const noise = (Math.random() - 0.5) * prediction * 0.015;
                prediction += noise;
                
                // Bound the prediction
                prediction = Math.max(prediction, baseline * 0.5);
                
                predictions.push({
                    date: this.formatDate(predictionDate),
                    price: parseFloat(prediction.toFixed(2)),
                    confidence: Math.max(0.8 - i * 0.025, 0.35)
                });
            }
            
            return predictions;
        }

        // Matrix utility methods (similar to linear regression)
        transpose(matrix) {
            return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
        }

        matrixMultiply(a, b) {
            const result = [];
            for (let i = 0; i < a.length; i++) {
                result[i] = [];
                for (let j = 0; j < b[0].length; j++) {
                    let sum = 0;
                    for (let k = 0; k < b.length; k++) {
                        sum += a[i][k] * b[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }

        matrixVectorMultiply(matrix, vector) {
            return matrix.map(row => row.reduce((sum, val, i) => sum + val * vector[i], 0));
        }

        matrixInverse(matrix) {
            // Same implementation as linear regression
            const n = matrix.length;
            const augmented = matrix.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]);
            
            for (let i = 0; i < n; i++) {
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                        maxRow = k;
                    }
                }
                [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
                
                for (let k = i + 1; k < n; k++) {
                    const factor = augmented[k][i] / augmented[i][i];
                    for (let j = i; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
            
            for (let i = n - 1; i >= 0; i--) {
                for (let k = i - 1; k >= 0; k--) {
                    const factor = augmented[k][i] / augmented[i][i];
                    for (let j = i; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
            
            for (let i = 0; i < n; i++) {
                const divisor = augmented[i][i];
                for (let j = 0; j < 2 * n; j++) {
                    augmented[i][j] /= divisor;
                }
            }
            
            return augmented.map(row => row.slice(n));
        }

        formatDate(date) {
            return date.toISOString().split('T')[0];
        }
    }

    // Replace models in existing aiModelManager
    if (window.aiModelManager) {
        // Replace traditional models with optimized versions
        window.aiModelManager.models.sma = new OptimizedSMAModel();
        window.aiModelManager.models.ema = new OptimizedEMAModel();
        window.aiModelManager.models.linear = new OptimizedLinearRegressionModel();
        window.aiModelManager.models.polynomial = new OptimizedPolynomialRegressionModel();
        
        console.log('Optimized traditional models loaded successfully');
        
        // Update model descriptions
        const originalGetModelDescription = window.aiModelManager.getModelDescription;
        window.aiModelManager.getModelDescription = function(modelType) {
            const descriptions = {
                sma: '優化簡單移動平均：自適應週期，融入趨勢和動量分析',
                ema: '優化指數移動平均：加入速度和加速度預測，提升方向準確率',
                linear: '多元線性回歸：整合技術指標和時間序列特徵的線性模型',
                polynomial: '正則化多項式回歸：非線性關係建模，加入趨勢阻尼機制',
                xgboost: 'XGBoost 梯度提升：集成決策樹，處理複雜非線性關係',
                transformer: 'Transformer 注意力模型：捕捉長期依賴和市場模式',
                lstm: 'LSTM 長短期記憶網路：專門處理時序資料的深度學習模型',
                ensemble: '集成學習模型：多模型加權融合，提升預測穩定性',
                sentiment: '情緒增強模型：整合新聞情緒和基本面分析的綜合模型'
            };
            
            return descriptions[modelType] || originalGetModelDescription.call(this, modelType);
        };
    }

})();