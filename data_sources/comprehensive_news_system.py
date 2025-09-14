# data_sources/comprehensive_news_system.py - REAL Market Intelligence
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
import feedparser
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

class ComprehensiveMarketIntelligence:
    def __init__(self):
        # Multiple news sources for comprehensive coverage
        self.news_sources = {
            'economic_times': {
                'url': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
                'weight': 1.0,
                'category': 'indian_business'
            },
            'business_standard': {
                'url': 'https://www.business-standard.com/rss/markets-106.rss',
                'weight': 0.8,
                'category': 'indian_markets'
            },
            'moneycontrol': {
                'url': 'https://www.moneycontrol.com/rss/businessnews.xml',
                'weight': 0.9,
                'category': 'indian_finance'
            },
            'reuters_business': {
                'url': 'https://feeds.reuters.com/reuters/businessNews',
                'weight': 1.2,
                'category': 'global_business'
            },
            'bloomberg': {
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'weight': 1.5,
                'category': 'global_markets'
            }
        }
        
        # Market sentiment keywords and their impact weights
        self.sentiment_keywords = {
            # Positive market drivers
            'bullish_keywords': {
                'rate cut': 2.0, 'stimulus': 2.5, 'GDP growth': 1.8, 'profit': 1.5,
                'rally': 1.8, 'bull market': 2.0, 'economic recovery': 2.2,
                'foreign investment': 1.7, 'merger': 1.5, 'acquisition': 1.5,
                'dividend': 1.2, 'earnings beat': 2.0, 'upgrade': 1.8, 'buyback': 1.6
            },
            # Negative market drivers  
            'bearish_keywords': {
                'recession': -2.5, 'inflation': -1.8, 'rate hike': -2.0, 'war': -2.2,
                'crash': -2.8, 'bear market': -2.0, 'slowdown': -1.5, 'bankruptcy': -2.0,
                'fraud': -2.2, 'scandal': -1.8, 'downgrade': -1.5, 'sell-off': -1.8,
                'correction': -1.5, 'volatility': -1.0, 'uncertainty': -1.2
            },
            # Global/Political events
            'geopolitical_keywords': {
                'election': 1.5, 'policy': 1.2, 'trade war': -1.8, 'sanctions': -1.5,
                'brexit': -1.0, 'china': 1.0, 'usa': 1.2, 'fed': 2.0, 'rbi': 1.8,
                'trump': 1.5, 'biden': 1.0, 'modi': 1.2, 'oil': 1.5, 'dollar': 1.3
            }
        }
        
        self.all_news = []
        self.market_sentiment_score = 0
    
    def fetch_rss_feed(self, source_name, source_info):
        """Fetch news from RSS feeds"""
        try:
            print(f"📰 Fetching {source_name}...")
            feed = feedparser.parse(source_info['url'])
            
            news_items = []
            for entry in feed.entries[:10]:  # Get latest 10 items
                # Get publication date
                pub_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                
                # Only get news from last 24 hours
                if (datetime.now() - pub_date).days <= 1:
                    news_item = {
                        'source': source_name,
                        'category': source_info['category'],
                        'weight': source_info['weight'],
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': pub_date,
                        'text': f"{entry.get('title', '')} {entry.get('summary', '')}"
                    }
                    news_items.append(news_item)
            
            print(f"✅ {source_name}: {len(news_items)} recent articles")
            return news_items
            
        except Exception as e:
            print(f"❌ Error fetching {source_name}: {e}")
            return []
    
    def fetch_all_news_parallel(self):
        """Fetch news from all sources in parallel"""
        print("🌍 === COMPREHENSIVE NEWS AGGREGATION ===")
        
        all_news = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(self.fetch_rss_feed, name, info): name 
                for name, info in self.news_sources.items()
            }
            
            for future in future_to_source:
                try:
                    news_items = future.result(timeout=15)  # 15 second timeout
                    all_news.extend(news_items)
                except Exception as e:
                    source_name = future_to_source[future]
                    print(f"❌ {source_name}: {e}")
        
        self.all_news = all_news
        print(f"📰 Total news collected: {len(all_news)} articles")
        return len(all_news) > 0
    
    def analyze_keyword_sentiment(self, text):
        """Analyze sentiment based on market-specific keywords"""
        text_lower = text.lower()
        sentiment_score = 0
        keyword_matches = []
        
        # Check all keyword categories
        for category, keywords in self.sentiment_keywords.items():
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    sentiment_score += weight
                    keyword_matches.append({
                        'keyword': keyword,
                        'weight': weight,
                        'category': category
                    })
        
        return sentiment_score, keyword_matches
    
    def analyze_comprehensive_sentiment(self):
        """Comprehensive sentiment analysis combining multiple methods"""
        print("\n🧠 === COMPREHENSIVE SENTIMENT ANALYSIS ===")
        
        if not self.all_news:
            return {'overall_sentiment': 0, 'analysis': 'No news data'}
        
        total_sentiment = 0
        total_weight = 0
        category_sentiments = {}
        keyword_impact = {}
        
        for news_item in self.all_news:
            # TextBlob sentiment
            blob_sentiment = TextBlob(news_item['text']).sentiment.polarity
            
            # Keyword-based sentiment
            keyword_sentiment, keywords = self.analyze_keyword_sentiment(news_item['text'])
            
            # Combined sentiment
            combined_sentiment = (blob_sentiment + keyword_sentiment) * news_item['weight']
            
            total_sentiment += combined_sentiment
            total_weight += news_item['weight']
            
            # Category-wise sentiment
            category = news_item['category']
            if category not in category_sentiments:
                category_sentiments[category] = {'sentiment': 0, 'count': 0}
            category_sentiments[category]['sentiment'] += combined_sentiment
            category_sentiments[category]['count'] += 1
            
            # Track keyword impacts
            for kw in keywords:
                if kw['keyword'] not in keyword_impact:
                    keyword_impact[kw['keyword']] = {'count': 0, 'total_impact': 0}
                keyword_impact[kw['keyword']]['count'] += 1
                keyword_impact[kw['keyword']]['total_impact'] += kw['weight']
        
        # Overall sentiment
        overall_sentiment = total_sentiment / total_weight if total_weight > 0 else 0
        
        # Normalize category sentiments
        for category in category_sentiments:
            if category_sentiments[category]['count'] > 0:
                category_sentiments[category]['avg_sentiment'] = (
                    category_sentiments[category]['sentiment'] / 
                    category_sentiments[category]['count']
                )
        
        self.market_sentiment_score = overall_sentiment
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_strength': abs(overall_sentiment),
            'market_direction': 'BULLISH' if overall_sentiment > 0.5 else 'BEARISH' if overall_sentiment < -0.5 else 'NEUTRAL',
            'category_sentiments': category_sentiments,
            'top_keywords': sorted(keyword_impact.items(), key=lambda x: abs(x[1]['total_impact']), reverse=True)[:10],
            'news_count': len(self.all_news),
            'confidence': min(len(self.all_news) / 50, 1.0)  # Confidence based on news volume
        }
    
    def get_macro_economic_indicators(self):
        """Fetch key macro indicators that move markets"""
        print("\n📊 === MACRO ECONOMIC INDICATORS ===")
        
        try:
            # Fetch key indices and indicators
            indicators = {
                '^NSEI': 'Nifty 50',
                '^BSESN': 'Sensex',
                '^VIX': 'VIX (Fear Index)',
                'USDINR=X': 'USD/INR',
                '^TNX': 'US 10Y Treasury',
                'CL=F': 'Crude Oil'
            }
            
            macro_data = {}
            
            for symbol, name in indicators.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='5d')
                    
                    if not data.empty:
                        current = data['Close'].iloc[-1]
                        previous = data['Close'].iloc[-2] if len(data) > 1 else current
                        change_pct = ((current - previous) / previous) * 100
                        
                        macro_data[name] = {
                            'current': current,
                            'change_pct': change_pct,
                            'symbol': symbol
                        }
                        
                        print(f"   📈 {name}: {current:.2f} ({change_pct:+.2f}%)")
                        
                except:
                    continue
            
            return macro_data
            
        except Exception as e:
            print(f"❌ Error fetching macro data: {e}")
            return {}
    
    def generate_market_intelligence_report(self):
        """Generate comprehensive market intelligence report"""
        print("=" * 90)
        print("🌍 COMPREHENSIVE MARKET INTELLIGENCE REPORT")
        print(f"📅 Generated: {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}")
        print("=" * 90)
        
        # Fetch and analyze news
        if not self.fetch_all_news_parallel():
            print("❌ Failed to fetch news data")
            return None
        
        # Sentiment analysis
        sentiment_analysis = self.analyze_comprehensive_sentiment()
        
        # Macro indicators
        macro_data = self.get_macro_economic_indicators()
        
        # Display results
        print(f"\n📰 === NEWS SENTIMENT ANALYSIS ===")
        print(f"Overall Market Sentiment: {sentiment_analysis['overall_sentiment']:.3f}")
        print(f"Market Direction: {sentiment_analysis['market_direction']}")
        print(f"Sentiment Strength: {sentiment_analysis['sentiment_strength']:.3f}")
        print(f"Analysis Confidence: {sentiment_analysis['confidence']:.1%}")
        print(f"News Articles Analyzed: {sentiment_analysis['news_count']}")
        
        # Top keywords driving sentiment
        print(f"\n🔥 TOP MARKET DRIVERS:")
        for keyword, data in sentiment_analysis['top_keywords'][:5]:
            impact = "🟢 BULLISH" if data['total_impact'] > 0 else "🔴 BEARISH"
            print(f"   {impact} '{keyword}': {data['count']} mentions, Impact: {data['total_impact']:+.1f}")
        
        # Category breakdown
        print(f"\n📊 SENTIMENT BY CATEGORY:")
        for category, data in sentiment_analysis['category_sentiments'].items():
            if 'avg_sentiment' in data:
                sentiment = data['avg_sentiment']
                direction = "🟢 Positive" if sentiment > 0.2 else "🔴 Negative" if sentiment < -0.2 else "⚪ Neutral"
                print(f"   {direction} {category.title()}: {sentiment:+.3f} ({data['count']} articles)")
        
        # Market recommendation
        print(f"\n🎯 === MARKET INTELLIGENCE SUMMARY ===")
        
        overall_score = sentiment_analysis['overall_sentiment']
        
        if overall_score > 1.0:
            recommendation = "🚀 STRONGLY BULLISH - Major positive catalysts detected"
        elif overall_score > 0.3:
            recommendation = "📈 BULLISH - Positive market sentiment"
        elif overall_score > -0.3:
            recommendation = "⚪ NEUTRAL - Mixed signals, be cautious"
        elif overall_score > -1.0:
            recommendation = "📉 BEARISH - Negative market sentiment"
        else:
            recommendation = "🚨 STRONGLY BEARISH - Major negative catalysts"
        
        print(f"Market Outlook: {recommendation}")
        print(f"Confidence Level: {sentiment_analysis['confidence']:.1%}")
        
        return {
            'sentiment': sentiment_analysis,
            'macro_data': macro_data,
            'recommendation': recommendation,
            'overall_score': overall_score
        }

if __name__ == "__main__":
    # Initialize comprehensive intelligence system
    intel_system = ComprehensiveMarketIntelligence()
    
    # Generate full market intelligence report
    report = intel_system.generate_market_intelligence_report()
    
    if report:
        print("\n" + "="*90)
        print("✅ COMPREHENSIVE MARKET INTELLIGENCE COMPLETE!")
        print("🌍 Your AI now sees the entire market ecosystem!")
    else:
        print("❌ Market intelligence gathering failed")

