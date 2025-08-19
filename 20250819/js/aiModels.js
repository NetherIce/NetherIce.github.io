// AI Prediction Models for Stock Price Forecasting
(function() {
    'use strict';
    
    class AIModelManager {
    constructor() {
        this.models = {
            sma: new SimpleMovingAverageModel(),
            ema: new ExponentialMovingAverageModel(),
            linear: new LinearRegressionModel(),
            polynomial: new PolynomialRegressionModel()
        };
        this.currentModel = null;
        this.predictions = [];
        this.metrics = {};
    }

    // Get available models
    getAvailableModels() {
        return Object.keys(this.models);
    }

    // Train and predict using specified model
    async predict(modelType, stockData, predictionDays = 7) {
        if (!this.models[modelType]) {
            throw new Error(`Model type ${modelType} not supported`);
        }

        this.currentModel = this.models[modelType];
        
        // Train model
        await this.currentModel.train(stockData);
        
        // Make predictions
        this.predictions = this.currentModel.predict(predictionDays);
        
        // Calculate metrics
        this.metrics = this.calculateMetrics(stockData, this.predictions);
        
        return {
            predictions: this.predictions,
            metrics: this.metrics,
            model: modelType
        };
    }

    // Calculate model performance metrics
    calculateMetrics(actualData, predictions) {
        if (actualData.length === 0 || predictions.length === 0) {
            return { mae: 0, rmse: 0, r2: 0, directionAccuracy: 0 };
        }

        // Use last N actual values to compare with predictions
        const compareLength = Math.min(actualData.length, predictions.length);
        const actual = actualData.slice(-compareLength).map(item => item.close);
        const predicted = predictions.slice(0, compareLength).map(item => item.price);

        // Mean Absolute Error (MAE)
        const mae = this.calculateMAE(actual, predicted);
        
        // Root Mean Square Error (RMSE)
        const rmse = this.calculateRMSE(actual, predicted);
        
        // R-squared (R²)
        const r2 = this.calculateR2(actual, predicted);
        
        // Direction Accuracy (percentage of correct trend predictions)
        const directionAccuracy = this.calculateDirectionAccuracy(actual, predicted);

        return {
            mae: parseFloat(mae.toFixed(4)),
            rmse: parseFloat(rmse.toFixed(4)),
            r2: parseFloat(r2.toFixed(4)),
            directionAccuracy: parseFloat(directionAccuracy.toFixed(4))
        };
    }

    // Calculate Mean Absolute Error
    calculateMAE(actual, predicted) {
        if (actual.length !== predicted.length || actual.length === 0) return 0;
        
        const sum = actual.reduce((acc, val, i) => {
            return acc + Math.abs(val - predicted[i]);
        }, 0);
        
        return sum / actual.length;
    }

    // Calculate Root Mean Square Error
    calculateRMSE(actual, predicted) {
        if (actual.length !== predicted.length || actual.length === 0) return 0;
        
        const sumSquares = actual.reduce((acc, val, i) => {
            return acc + Math.pow(val - predicted[i], 2);
        }, 0);
        
        return Math.sqrt(sumSquares / actual.length);
    }

    // Calculate R-squared
    calculateR2(actual, predicted) {
        if (actual.length !== predicted.length || actual.length === 0) return 0;
        
        const actualMean = actual.reduce((sum, val) => sum + val, 0) / actual.length;
        
        const totalSumSquares = actual.reduce((acc, val) => {
            return acc + Math.pow(val - actualMean, 2);
        }, 0);
        
        const residualSumSquares = actual.reduce((acc, val, i) => {
            return acc + Math.pow(val - predicted[i], 2);
        }, 0);
        
        return totalSumSquares === 0 ? 0 : 1 - (residualSumSquares / totalSumSquares);
    }

    // Calculate Direction Accuracy
    calculateDirectionAccuracy(actual, predicted) {
        if (actual.length < 2 || predicted.length < 2) return 0;
        
        let correctDirections = 0;
        let totalDirections = 0;
        
        for (let i = 1; i < Math.min(actual.length, predicted.length); i++) {
            const actualDirection = actual[i] > actual[i-1];
            const predictedDirection = predicted[i] > predicted[i-1];
            
            if (actualDirection === predictedDirection) {
                correctDirections++;
            }
            totalDirections++;
        }
        
        return totalDirections === 0 ? 0 : correctDirections / totalDirections;
    }

    // Get model description
    getModelDescription(modelType) {
        const descriptions = {
            sma: '簡單移動平均：計算最近N天的平均價格',
            ema: '指數移動平均：給予近期價格更高權重',
            linear: '線性回歸：使用線性函數擬合價格趨勢',
            polynomial: '多項式回歸：使用多項式函數捕捉非線性趨勢'
        };
        
        return descriptions[modelType] || '未知模型';
    }
}

// Simple Moving Average Model
class SimpleMovingAverageModel {
    constructor(period = 20) {
        this.period = period;
        this.trainedData = [];
    }

    async train(stockData) {
        this.trainedData = stockData;
        return true;
    }

    predict(days) {
        if (this.trainedData.length < this.period) {
            return [];
        }

        const predictions = [];
        const recent = this.trainedData.slice(-this.period);
        let currentAverage = recent.reduce((sum, item) => sum + item.close, 0) / this.period;
        
        const lastDate = new Date(this.trainedData[this.trainedData.length - 1].date);
        
        for (let i = 1; i <= days; i++) {
            const predictionDate = new Date(lastDate);
            predictionDate.setDate(predictionDate.getDate() + i);
            
            // Add some random variation
            const variation = (Math.random() - 0.5) * currentAverage * 0.02;
            const predictedPrice = currentAverage + variation;
            
            predictions.push({
                date: this.formatDate(predictionDate),
                price: parseFloat(predictedPrice.toFixed(2)),
                confidence: 0.8 - (i * 0.05) // Decreasing confidence over time
            });
            
            // Update average for next prediction
            currentAverage = predictedPrice;
        }
        
        return predictions;
    }

    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
}

// Exponential Moving Average Model
class ExponentialMovingAverageModel {
    constructor(period = 20) {
        this.period = period;
        this.trainedData = [];
        this.alpha = 2 / (period + 1);
    }

    async train(stockData) {
        this.trainedData = stockData;
        return true;
    }

    predict(days) {
        if (this.trainedData.length < this.period) {
            return [];
        }

        const predictions = [];
        
        // Calculate initial EMA
        let ema = this.trainedData.slice(0, this.period)
            .reduce((sum, item) => sum + item.close, 0) / this.period;
        
        // Update EMA with recent data
        for (let i = this.period; i < this.trainedData.length; i++) {
            ema = (this.trainedData[i].close * this.alpha) + (ema * (1 - this.alpha));
        }
        
        const lastDate = new Date(this.trainedData[this.trainedData.length - 1].date);
        
        for (let i = 1; i <= days; i++) {
            const predictionDate = new Date(lastDate);
            predictionDate.setDate(predictionDate.getDate() + i);
            
            // Add trend and random variation
            const trend = (ema - this.trainedData[this.trainedData.length - 10].close) / 10;
            const variation = (Math.random() - 0.5) * ema * 0.015;
            const predictedPrice = ema + trend + variation;
            
            predictions.push({
                date: this.formatDate(predictionDate),
                price: parseFloat(predictedPrice.toFixed(2)),
                confidence: 0.85 - (i * 0.04)
            });
            
            // Update EMA for next prediction
            ema = (predictedPrice * this.alpha) + (ema * (1 - this.alpha));
        }
        
        return predictions;
    }

    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
}

// Linear Regression Model
class LinearRegressionModel {
    constructor() {
        this.slope = 0;
        this.intercept = 0;
        this.trainedData = [];
    }

    async train(stockData) {
        this.trainedData = stockData;
        
        if (stockData.length < 2) return false;
        
        // Use last 60 days or all available data
        const trainingData = stockData.slice(-60);
        const n = trainingData.length;
        
        // Prepare data for linear regression
        const x = Array.from({length: n}, (_, i) => i);
        const y = trainingData.map(item => item.close);
        
        // Calculate slope and intercept
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
        const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
        
        this.slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        this.intercept = (sumY - this.slope * sumX) / n;
        
        return true;
    }

    predict(days) {
        if (this.trainedData.length === 0) return [];
        
        const predictions = [];
        const lastDate = new Date(this.trainedData[this.trainedData.length - 1].date);
        const startX = this.trainedData.length;
        
        for (let i = 1; i <= days; i++) {
            const predictionDate = new Date(lastDate);
            predictionDate.setDate(predictionDate.getDate() + i);
            
            // Linear prediction with some noise
            const basePrice = this.slope * (startX + i - 1) + this.intercept;
            const noise = (Math.random() - 0.5) * basePrice * 0.01;
            const predictedPrice = Math.max(basePrice + noise, 0.01);
            
            predictions.push({
                date: this.formatDate(predictionDate),
                price: parseFloat(predictedPrice.toFixed(2)),
                confidence: Math.max(0.8 - (i * 0.03), 0.2)
            });
        }
        
        return predictions;
    }

    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
}

// Polynomial Regression Model
class PolynomialRegressionModel {
    constructor(degree = 2) {
        this.degree = degree;
        this.coefficients = [];
        this.trainedData = [];
    }

    async train(stockData) {
        this.trainedData = stockData;
        
        if (stockData.length < this.degree + 1) return false;
        
        // Use last 90 days or all available data
        const trainingData = stockData.slice(-90);
        const n = trainingData.length;
        
        // Prepare data
        const x = Array.from({length: n}, (_, i) => i);
        const y = trainingData.map(item => item.close);
        
        // Simplified polynomial regression (2nd degree)
        this.coefficients = this.fitPolynomial(x, y, this.degree);
        
        return true;
    }

    // Simplified polynomial fitting using normal equation
    fitPolynomial(x, y, degree) {
        const n = x.length;
        
        // For simplicity, implement 2nd degree polynomial
        if (degree === 2) {
            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = y.reduce((a, b) => a + b, 0);
            const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
            const sumX3 = x.reduce((acc, xi) => acc + xi * xi * xi, 0);
            const sumX4 = x.reduce((acc, xi) => acc + Math.pow(xi, 4), 0);
            const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
            const sumX2Y = x.reduce((acc, xi, i) => acc + xi * xi * y[i], 0);
            
            // Solve 3x3 system for ax² + bx + c
            const a = ((n * sumX2Y - sumX2 * sumY) * (n * sumX2 - sumX * sumX) - 
                      (n * sumXY - sumX * sumY) * (n * sumX3 - sumX * sumX2)) /
                     ((n * sumX4 - sumX2 * sumX2) * (n * sumX2 - sumX * sumX) - 
                      (n * sumX3 - sumX * sumX2) * (n * sumX3 - sumX * sumX2));
            
            const b = ((n * sumXY - sumX * sumY) - a * (n * sumX3 - sumX * sumX2)) / 
                     (n * sumX2 - sumX * sumX);
            
            const c = (sumY - b * sumX - a * sumX2) / n;
            
            return [c, b, a]; // coefficients for c + bx + ax²
        }
        
        // Fallback to linear
        return [0, 1, 0];
    }

    predict(days) {
        if (this.trainedData.length === 0 || this.coefficients.length === 0) return [];
        
        const predictions = [];
        const lastDate = new Date(this.trainedData[this.trainedData.length - 1].date);
        const startX = this.trainedData.length;
        
        for (let i = 1; i <= days; i++) {
            const predictionDate = new Date(lastDate);
            predictionDate.setDate(predictionDate.getDate() + i);
            
            const x = startX + i - 1;
            
            // Polynomial prediction: a₀ + a₁x + a₂x²
            let predictedPrice = this.coefficients[0] + 
                               this.coefficients[1] * x + 
                               this.coefficients[2] * x * x;
            
            // Add noise and ensure positive price
            const noise = (Math.random() - 0.5) * predictedPrice * 0.012;
            predictedPrice = Math.max(predictedPrice + noise, 0.01);
            
            predictions.push({
                date: this.formatDate(predictionDate),
                price: parseFloat(predictedPrice.toFixed(2)),
                confidence: Math.max(0.75 - (i * 0.04), 0.25)
            });
        }
        
        return predictions;
    }

    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
}

    // Initialize global AI model manager
    if (!window.aiModelManager) {
        window.aiModelManager = new AIModelManager();
    }
})();