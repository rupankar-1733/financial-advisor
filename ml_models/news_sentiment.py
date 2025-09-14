# ml_models/news_sentiment.py
import requests
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta

class NewsSentimentAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key  # NewsAPI key
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def fetch_news(self, symbol="TCS", days=7):
        """Fetch news for a stock symbol"""
        # For demo, we'll use yfinance news (free)
        ticker = yf.Ticker(f"{symbol}.NS")
        
        try:
            news = ticker.news
            news_data = []
            
            for item in news[:20]:  # Get latest 20 news
                news_data.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'url': item.get('link', '')
                })
            
            return pd.DataFrame(news_data)
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        # VADER sentiment
        vader_score = self.vader_analyzer.polarity_scores(text)
        
        return {
            'textblob_score': textblob_score,
            'textblob_label': 'Positive' if textblob_score > 0.1 else 'Negative' if textblob_score < -0.1 else 'Neutral',
            'vader_compound': vader_score['compound'],
            'vader_label': 'Positive' if vader_score['compound'] > 0.05 else 'Negative' if vader_score['compound'] < -0.05 else 'Neutral'
        }
    
    def get_market_sentiment(self, symbol="TCS"):
        """Get overall market sentiment for a stock"""
        print(f"📰 Analyzing news sentiment for {symbol}...")
        
        news_df = self.fetch_news(symbol)
        
        if news_df.empty:
            return None
        
        # Analyze each news item
        sentiments = []
        
        for _, row in news_df.iterrows():
            text = f"{row['title']} {row['summary']}"
            sentiment = self.analyze_sentiment(text)
            sentiment['title'] = row['title']
            sentiment['published'] = row['published']
            sentiments.append(sentiment)
        
        sentiment_df = pd.DataFrame(sentiments)
        
        # Calculate overall sentiment
        avg_textblob = sentiment_df['textblob_score'].mean()
        avg_vader = sentiment_df['vader_compound'].mean()
        
        # Count sentiment labels
        positive_count = len(sentiment_df[sentiment_df['textblob_label'] == 'Positive'])
        negative_count = len(sentiment_df[sentiment_df['textblob_label'] == 'Negative'])
        neutral_count = len(sentiment_df[sentiment_df['textblob_label'] == 'Neutral'])
        
        overall_sentiment = 'Positive' if avg_textblob > 0.1 else 'Negative' if avg_textblob < -0.1 else 'Neutral'
        
        return {
            'symbol': symbol,
            'total_news': len(sentiment_df),
            'overall_sentiment': overall_sentiment,
            'avg_textblob_score': avg_textblob,
            'avg_vader_score': avg_vader,
            'positive_news': positive_count,
            'negative_news': negative_count,
            'neutral_news': neutral_count,
            'sentiment_strength': abs(avg_textblob),
            'news_details': sentiment_df.to_dict('records')
        }

if __name__ == "__main__":
    analyzer = NewsSentimentAnalyzer()
    sentiment = analyzer.get_market_sentiment("TCS")
    
    if sentiment:
        print(f"\n📊 === NEWS SENTIMENT ANALYSIS ===")
        print(f"Stock: {sentiment['symbol']}")
        print(f"Total News Analyzed: {sentiment['total_news']}")
        print(f"Overall Sentiment: {sentiment['overall_sentiment']}")
        print(f"Sentiment Score: {sentiment['avg_textblob_score']:.3f}")
        print(f"Positive News: {sentiment['positive_news']}")
        print(f"Negative News: {sentiment['negative_news']}")
        print(f"Neutral News: {sentiment['neutral_news']}")
