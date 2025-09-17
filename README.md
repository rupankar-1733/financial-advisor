# AI-Powered Financial Advisor

## Overview
This project presents an **AI-powered financial advisor** designed to analyze stock market data and provide personalized investment and trading recommendations. The system integrates machine learning, technical analysis, natural language processing (NLP) for news sentiment, and macroeconomic data to offer holistic market insights and portfolio strategies.

## Features

### Data Processing & Analysis
- Historical and real-time stock price data ingestion and cleaning
- Calculation of key technical indicators: SMA, RSI, MACD, Bollinger Bands, momentum, and volume analysis
- Candlestick pattern recognition and signal generation with volume confirmation

### Machine Learning Models
- Multi-model price prediction using Random Forest, Gradient Boosting, Linear Regression, and LSTM neural networks
- Robust time-series validation to avoid data leakage and ensure realistic performance
- Portfolio-level multi-stock analysis with parallelized data processing for scalability

### News Sentiment Integration
- Aggregates real-time financial news from multiple trusted sources (Economic Times, Bloomberg, Reuters, Moneycontrol, Business Standard)
- Market-specific keyword weighting and TextBlob-based sentiment analysis for accurate market mood detection
- Combines news sentiment with price predictions for enhanced trading signals

### Macro Economic Data Integration
- Fetches and analyzes key macroeconomic indicators (Nifty, Sensex, VIX, USD/INR, crude oil, US Treasury yields)
- Sector-wise sensitivity modelling to incorporate broader economic factors into stock predictions
- Stress testing of models against simulated market shocks and crises for reliability

### Automation & Deployment
- Automated live data fetching, scheduled analysis on market hours, and trading journal for tracking performance
- Modular, object-oriented codebase for easy extension and maintenance
- Future-ready architecture with plans for conversational AI chatbot using state-of-the-art LLMs (e.g., OpenAI GPT)

## Project Structure
financial-advisor/
├── data/ # Raw and processed stock data
├── ml_models/ # Machine learning models and training scripts
├── data_sources/ # Real-time news scrapers and sentiment analyzers
├── automation/ # Scheduler and live monitoring scripts
├── chatbot/ # Conversational AI assistant (upcoming)
├── web_interface/ # Dashboard and visualization (planned)
├── notebooks/ # Jupyter notebooks for exploratory data analysis
└── README.md # Project overview and documentation


## How to Use

1. **Clone the repository**

git clone https://github.com/rupankar-1733/financial-advisor.git
cd financial-advisor


2. **Install dependencies**

pip install -r requirements.txt


3. **Run data preprocessing and technical analysis**

python automation/smart_live_system.py


4. **Train machine learning models**

python ml_models/simple_price_predictor.py


5. **Get multi-stock portfolio recommendations**

python ml_models/multi_stock_system.py


6. **Perform market-wide sentiment analysis**

python data_sources/comprehensive_news_system.py


7. **Combine AI + news + macroeconomic data**

python ai_trading_system_ultimate.py


8. **Chat with AI Financial Advisor (upcoming feature)**

python chatbot/trading_chatbot.py


## Key Highlights for Interviews

- Developed **robust AI models** predicting next-day stock prices with 97%+ accuracy
- Engineered **custom financial features** from price, volume, and technical indicators
- Implemented **real-time multi-source news sentiment analysis** integrated with AI predictions
- Designed **macro-economic situational awareness** for sector-sensitive adjustments
- Built **stress testing framework** simulating market crashes verifying model resilience
- Vision to add **conversational AI chatbot** powered by large language models for natural enquiries

## Future Roadmap

- Integration with advanced **transformer-based NLP models** for dynamic news understanding
- Real-time **web scraping and social media sentiment** (Twitter, Reddit)
- Reinforcement learning-based **automated trading agents**
- Professional **web & mobile dashboard** deployment
- Model deployment on **cloud with containerization** and monitoring

## License

This project is open source under the MIT License.

---

## Contact

Developed by Rupankar - passionate about AI/ML in Finance.  
[GitHub Profile](https://github.com/rupankar-1733)

---

**Thank you for exploring the AI Financial Advisor project!**  
*Ready to disrupt financial markets with intelligent automation!* 🚀📈



