# data_sources/comprehensive_intelligence_system.py - Ultimate Market Intelligence
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
import feedparser
import time
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import re

class UltimateMarketIntelligence:
    def __init__(self):
        print("🌍 Initializing Ultimate Market Intelligence System...")
        
        # Comprehensive news sources (expanded)
        self.news_sources = {
            'economic_times': {
                'url': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
                'weight': 1.0,
                'region': 'India'
            },
            'reuters_global': {
                'url': 'https://feeds.reuters.com/reuters/businessNews',
                'weight': 1.5,
                'region': 'Global'
            },
            'bloomberg_markets': {
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'weight': 1.8,
                'region': 'Global'
            },
            'cnn_business': {
                'url': 'http://rss.cnn.com/rss/money_news_international.rss',
                'weight': 1.2,
                'region': 'Global'
            },
            'financial_express': {
                'url': 'https://www.financialexpress.com/market/rss',
                'weight': 1.0,
                'region': 'India'
            }
        }
        
        # Global macro indicators that affect Indian markets
        self.macro_indicators = {
            'DXY': 'US Dollar Index',
            '^VIX': 'Fear & Greed Index',
            '^TNX': 'US 10Y Treasury',
            'CL=F': 'Crude Oil WTI',
            'GC=F': 'Gold Futures',
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum',
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            'USDINR=X': 'USD/INR',
            '^NSEI': 'Nifty 50',
            'EURUSD=X': 'EUR/USD'
        }
        
        # Market-moving keywords (expanded and weighted)
        self.advanced_keywords = {
            # Central Bank Actions (High Impact)
            'fed_keywords': {
                'fed rate': 3.0, 'rate hike': 2.8, 'rate cut': 3.2, 'powell': 2.5,
                'fomc': 2.8, 'tapering': 2.0, 'qe': 2.2, 'monetary policy': 2.0
            },
            
            # Geopolitical Events (High Impact)
            'geopolitical_keywords': {
                'war': -3.5, 'ukraine': -2.0, 'russia': -2.2, 'china tension': -2.5,
                'trade war': -2.8, 'sanctions': -2.0, 'nuclear': -3.0, 'conflict': -2.5,
                'peace': 2.0, 'resolution': 1.8, 'ceasefire': 2.2
            },
            
            # Economic Indicators (Medium-High Impact)
            'economic_keywords': {
                'inflation': -2.5, 'deflation': -2.0, 'gdp growth': 2.8, 'recession': -3.5,
                'unemployment': -2.0, 'job growth': 2.2, 'consumer confidence': 2.0,
                'manufacturing': 1.8, 'pmi': 1.5, 'retail sales': 1.8
            },
            
            # Corporate Actions (Medium Impact)
            'corporate_keywords': {
                'earnings beat': 2.5, 'earnings miss': -2.5, 'guidance': 1.8,
                'buyback': 2.0, 'dividend': 1.5, 'merger': 2.2, 'acquisition': 2.0,
                'ipo': 1.8, 'split': 1.2, 'restructuring': -1.5
            },
            
            # Technology & Innovation (Medium Impact for Indian IT)
            'tech_keywords': {
                'ai': 2.0, 'artificial intelligence': 2.2, 'automation': 1.8,
                'cloud': 1.5, 'digitization': 1.8, 'semiconductor': 2.5,
                'chip shortage': -2.0, 'tech regulation': -1.8
            },
            
            # Crypto & Alt Assets (New Addition)
            'crypto_keywords': {
                'bitcoin': 1.5, 'cryptocurrency': 1.2, 'blockchain': 1.8,
                'defi': 1.0, 'nft': 0.8, 'crypto crash': -2.0, 'crypto ban': -2.5
            },
            
            # Energy & Commodities
            'commodity_keywords': {
                'oil price': 2.0, 'crude': 1.8, 'opec': 2.2, 'energy crisis': -2.5,
                'renewable energy': 2.0, 'gold': 1.5, 'copper': 1.2, 'steel': 1.0
            }
        }
        
        # Social media indicators (simulated - would need real API access)
        self.social_sentiment_sources = {
            'reddit_investing': {'weight': 1.2, 'keywords': ['investing', 'stocks', 'market']},
            'twitter_finance': {'weight': 1.5, 'keywords': ['$NIFTY', '$SPY', 'stocks']},
            'discord_trading': {'weight': 0.8, 'keywords': ['trading', 'calls', 'puts']}
        }
        
        self.all_intelligence = {}
        self.market_regime = 'UNKNOWN'
        
    def fetch_global_macro_data(self):
        """Enhanced macro data with more indicators"""
        print("🌍 === FETCHING GLOBAL MACRO INTELLIGENCE ===")
        
        macro_data = {}
        
        for symbol, name in self.macro_indicators.items():
            try:
                print(f"   📊 Fetching {name}...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='5d')
                
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2] if len(data) > 1 else current
                    change_pct = ((current - previous) / previous) * 100
                    
                    # Calculate volatility
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    
                    macro_data[name] = {
                        'symbol': symbol,
                        'current': current,
                        'change_pct': change_pct,
                        'volatility': volatility,
                        'trend': 'UP' if change_pct > 0 else 'DOWN',
                        'significance': 'HIGH' if abs(change_pct) > 2 else 'MEDIUM' if abs(change_pct) > 1 else 'LOW'
                    }
                    
                    print(f"      ✅ {name}: {current:.2f} ({change_pct:+.2f}%)")
                    
            except Exception as e:
                print(f"      ❌ {name}: {e}")
                continue
        
        self.all_intelligence['macro_data'] = macro_data
        return macro_data
    
    def analyze_crypto_correlation(self):
        """Analyze crypto impact on traditional markets"""
        print("\n₿ === CRYPTO-STOCK CORRELATION ANALYSIS ===")
        
        try:
            # Get crypto data
            btc_data = yf.Ticker('BTC-USD').history(period='5d')
            eth_data = yf.Ticker('ETH-USD').history(period='5d')
            
            # Get stock index data
            nifty_data = yf.Ticker('^NSEI').history(period='5d')
            
            if len(btc_data) >= 2 and len(nifty_data) >= 2:
                btc_returns = btc_data['Close'].pct_change().dropna()
                nifty_returns = nifty_data['Close'].pct_change().dropna()
                
                # Calculate correlation
                if len(btc_returns) >= 2 and len(nifty_returns) >= 2:
                    correlation = np.corrcoef(btc_returns[-min(len(btc_returns), len(nifty_returns)):], 
                                           nifty_returns[-min(len(btc_returns), len(nifty_returns)):])[0,1]
                    
                    btc_change = ((btc_data['Close'].iloc[-1] - btc_data['Close'].iloc[-2]) / 
                                 btc_data['Close'].iloc[-2]) * 100
                    
                    crypto_analysis = {
                        'btc_change': btc_change,
                        'correlation_with_stocks': correlation,
                        'risk_assessment': 'HIGH_CORRELATION' if abs(correlation) > 0.7 else 
                                         'MODERATE_CORRELATION' if abs(correlation) > 0.3 else 'LOW_CORRELATION',
                        'impact_on_stocks': 'POSITIVE' if btc_change > 5 else 'NEGATIVE' if btc_change < -5 else 'NEUTRAL'
                    }
                    
                    print(f"   ₿ BTC Change: {btc_change:+.2f}%")
                    print(f"   🔗 BTC-Nifty Correlation: {correlation:.3f}")
                    print(f"   📊 Risk Level: {crypto_analysis['risk_assessment']}")
                    
                    self.all_intelligence['crypto_analysis'] = crypto_analysis
                    return crypto_analysis
                    
        except Exception as e:
            print(f"   ❌ Crypto analysis failed: {e}")
        
        return None
    
    def fetch_comprehensive_news(self):
        """Enhanced news fetching with better error handling"""
        print("\n📰 === COMPREHENSIVE GLOBAL NEWS ANALYSIS ===")
        
        all_news = []
        
        def fetch_single_source(source_name, source_info):
            try:
                print(f"   📡 Fetching {source_name} ({source_info['region']})...")
                feed = feedparser.parse(source_info['url'])
                
                news_items = []
                for entry in feed.entries[:10]:
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Only get news from last 24 hours
                    if (datetime.now() - pub_date).days <= 1:
                        news_items.append({
                            'source': source_name,
                            'region': source_info['region'],
                            'weight': source_info['weight'],
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'published': pub_date,
                            'text': f"{entry.get('title', '')} {entry.get('summary', '')}"
                        })
                
                print(f"      ✅ {source_name}: {len(news_items)} articles")
                return news_items
                
            except Exception as e:
                print(f"      ❌ {source_name}: {e}")
                return []
        
        # Parallel fetching
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(fetch_single_source, name, info): name 
                for name, info in self.news_sources.items()
            }
            
            for future in futures:
                try:
                    news_items = future.result(timeout=15)
                    all_news.extend(news_items)
                except Exception as e:
                    print(f"      ❌ Future failed: {e}")
        
        self.all_intelligence['news_articles'] = all_news
        print(f"   📊 Total articles: {len(all_news)}")
        return all_news
    
    def advanced_sentiment_analysis(self, news_articles):
        """Advanced multi-layered sentiment analysis"""
        print("\n🧠 === ADVANCED SENTIMENT ANALYSIS ===")
        
        if not news_articles:
            return {'overall_sentiment': 0, 'confidence': 0}
        
        total_sentiment = 0
        total_weight = 0
        keyword_impacts = {}
        regional_sentiments = {}
        
        for article in news_articles:
            text = article['text'].lower()
            article_sentiment = 0
            
            # Multi-layer keyword analysis
            for category, keywords in self.advanced_keywords.items():
                for keyword, impact in keywords.items():
                    if keyword in text:
                        article_sentiment += impact
                        
                        if keyword not in keyword_impacts:
                            keyword_impacts[keyword] = {'count': 0, 'total_impact': 0, 'category': category}
                        keyword_impacts[keyword]['count'] += 1
                        keyword_impacts[keyword]['total_impact'] += impact
            
            # TextBlob sentiment
            blob_sentiment = TextBlob(article['text']).sentiment.polarity * 2  # Scale up
            
            # Combined sentiment with article weight
            combined_sentiment = (article_sentiment + blob_sentiment) * article['weight']
            
            total_sentiment += combined_sentiment
            total_weight += article['weight']
            
            # Regional sentiment tracking
            region = article['region']
            if region not in regional_sentiments:
                regional_sentiments[region] = {'sentiment': 0, 'count': 0}
            regional_sentiments[region]['sentiment'] += combined_sentiment
            regional_sentiments[region]['count'] += 1
        
        # Calculate overall sentiment
        overall_sentiment = total_sentiment / total_weight if total_weight > 0 else 0
        
        # Determine market regime
        if overall_sentiment > 1.5:
            self.market_regime = 'STRONG_BULL'
        elif overall_sentiment > 0.5:
            self.market_regime = 'BULL'
        elif overall_sentiment > -0.5:
            self.market_regime = 'NEUTRAL'
        elif overall_sentiment > -1.5:
            self.market_regime = 'BEAR'
        else:
            self.market_regime = 'STRONG_BEAR'
        
        # Calculate confidence based on news volume and keyword diversity
        confidence = min(len(news_articles) / 50, 1.0) * min(len(keyword_impacts) / 20, 1.0)
        
        sentiment_analysis = {
            'overall_sentiment': overall_sentiment,
            'market_regime': self.market_regime,
            'confidence': confidence,
            'regional_sentiments': {r: s['sentiment']/s['count'] for r, s in regional_sentiments.items()},
            'top_keywords': sorted(keyword_impacts.items(), key=lambda x: abs(x[1]['total_impact']), reverse=True)[:10],
            'news_count': len(news_articles)
        }
        
        print(f"   📊 Overall Sentiment: {overall_sentiment:.3f}")
        print(f"   🎭 Market Regime: {self.market_regime}")
        print(f"   🎯 Confidence: {confidence:.1%}")
        
        self.all_intelligence['sentiment'] = sentiment_analysis
        return sentiment_analysis
    
    def calculate_support_resistance_zones(self, symbol, period='3mo'):
        """Calculate key support/resistance levels for ₹10k investment"""
        print(f"\n📊 === CALCULATING ZONES FOR {symbol} ===")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1d')
            
            if len(data) < 20:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate support/resistance using pivot points
            highs = data['High']
            lows = data['Low']
            closes = data['Close']
            
            # Recent pivot highs and lows
            resistance_levels = []
            support_levels = []
            
            # Use rolling windows to find local maxima/minima
            for window in [5, 10, 20]:
                rolling_max = highs.rolling(window, center=True).max()
                rolling_min = lows.rolling(window, center=True).min()
                
                # Find resistance (local maxima)
                for i in range(window, len(data) - window):
                    if highs.iloc[i] == rolling_max.iloc[i] and highs.iloc[i] > current_price:
                        resistance_levels.append(highs.iloc[i])
                
                # Find support (local minima)  
                for i in range(window, len(data) - window):
                    if lows.iloc[i] == rolling_min.iloc[i] and lows.iloc[i] < current_price:
                        support_levels.append(lows.iloc[i])
            
            # Clean and sort levels
            resistance_levels = sorted(list(set([r for r in resistance_levels if r > current_price])))[:3]
            support_levels = sorted(list(set([s for s in support_levels if s < current_price])), reverse=True)[:3]
            
            # For ₹10k investment - calculate position sizes
            shares_possible = 10000 // current_price
            
            zones = {
                'current_price': current_price,
                'investment_amount': 10000,
                'shares_possible': int(shares_possible),
                'support_zones': support_levels,
                'resistance_zones': resistance_levels,
                'entry_recommendation': self.get_entry_strategy(current_price, support_levels, resistance_levels),
                'risk_levels': {
                    'stop_loss': support_levels[0] if support_levels else current_price * 0.95,
                    'take_profit_1': resistance_levels[0] if resistance_levels else current_price * 1.05,
                    'take_profit_2': resistance_levels[1] if len(resistance_levels) > 1 else current_price * 1.10
                }
            }
            
            print(f"   💰 Current Price: ₹{current_price:.2f}")
            print(f"   🛒 Shares for ₹10k: {shares_possible}")
            print(f"   🔻 Support Zones: {[f'₹{s:.2f}' for s in support_levels[:3]]}")
            print(f"   🔺 Resistance Zones: {[f'₹{r:.2f}' for r in resistance_levels[:3]]}")
            
            return zones
            
        except Exception as e:
            print(f"   ❌ Zone calculation failed: {e}")
            return None
    
    def get_entry_strategy(self, current_price, support_levels, resistance_levels):
        """Get entry strategy for ₹10k investment"""
        
        if not support_levels or not resistance_levels:
            return "WAIT - Insufficient data for clear zones"
        
        nearest_support = support_levels[0]
        nearest_resistance = resistance_levels[0]
        
        support_distance = (current_price - nearest_support) / current_price * 100
        resistance_distance = (nearest_resistance - current_price) / current_price * 100
        
        if support_distance < 2:  # Very close to support
            return "BUY NOW - Near strong support"
        elif resistance_distance < 2:  # Very close to resistance
            return "WAIT - Near resistance, expect pullback"
        elif support_distance < 5:  # Reasonably close to support
            return "BUY ON DIP - Wait for closer to support"
        elif resistance_distance > 10:  # Good upside potential
            return "GRADUAL BUY - Good risk/reward ratio"
        else:
            return "NEUTRAL - No clear edge, wait for better setup"
    
    def generate_ultimate_intelligence_report(self):
        """Generate comprehensive market intelligence report"""
        print("🚀 === ULTIMATE MARKET INTELLIGENCE SYSTEM ===")
        print("=" * 80)
        
        # Fetch all intelligence
        news_articles = self.fetch_comprehensive_news()
        sentiment = self.advanced_sentiment_analysis(news_articles)
        macro_data = self.fetch_global_macro_data()
        crypto_analysis = self.analyze_crypto_correlation()
        
        # Calculate zones for key stocks
        key_stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'NIFTY50']
        stock_zones = {}
        
        for stock in key_stocks:
            zones = self.calculate_support_resistance_zones(stock)
            if zones:
                stock_zones[stock] = zones
        
        self.all_intelligence['stock_zones'] = stock_zones
        
        # Generate final report
        self.display_ultimate_report()
        
        return self.all_intelligence
    
    def display_ultimate_report(self):
        """Display comprehensive intelligence report"""
        print("\n" + "=" * 90)
        print("🌍 ULTIMATE MARKET INTELLIGENCE REPORT")
        print(f"📅 {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}")
        print("=" * 90)
        
        # Market Regime
        sentiment = self.all_intelligence.get('sentiment', {})
        print(f"🎭 Market Regime: {self.market_regime}")
        print(f"📊 Sentiment Score: {sentiment.get('overall_sentiment', 0):.3f}")
        print(f"🎯 Confidence: {sentiment.get('confidence', 0):.1%}")
        
        # Top Market Drivers
        print(f"\n🔥 TOP GLOBAL MARKET DRIVERS:")
        top_keywords = sentiment.get('top_keywords', [])[:5]
        for keyword, data in top_keywords:
            impact = "🟢 BULLISH" if data['total_impact'] > 0 else "🔴 BEARISH"
            print(f"   {impact} '{keyword}': {data['count']} mentions, Impact: {data['total_impact']:+.1f}")
        
        # Macro Overview
        macro_data = self.all_intelligence.get('macro_data', {})
        print(f"\n🌍 GLOBAL MACRO SNAPSHOT:")
        for name, data in list(macro_data.items())[:6]:
            trend_emoji = "📈" if data['trend'] == 'UP' else "📉"
            print(f"   {trend_emoji} {name}: {data['current']:.2f} ({data['change_pct']:+.2f}%)")
        
        # Crypto Impact
        crypto = self.all_intelligence.get('crypto_analysis', {})
        if crypto:
            print(f"\n₿ CRYPTO MARKET IMPACT:")
            print(f"   Bitcoin: {crypto['btc_change']:+.2f}%")
            print(f"   Stock Correlation: {crypto['correlation_with_stocks']:.3f}")
            print(f"   Risk Level: {crypto['risk_assessment']}")
        
        # Investment Zones for ₹10k
        stock_zones = self.all_intelligence.get('stock_zones', {})
        if stock_zones:
            print(f"\n💰 ₹10,000 INVESTMENT ZONES:")
            print("-" * 70)
            print(f"{'STOCK':<12} {'PRICE':<8} {'SHARES':<6} {'STRATEGY':<25}")
            print("-" * 70)
            
            for stock, zones in stock_zones.items():
                if stock != 'NIFTY50':  # Skip index
                    stock_name = stock.replace('.NS', '')
                    price = f"₹{zones['current_price']:.0f}"
                    shares = zones['shares_possible']
                    strategy = zones['entry_recommendation'][:24]  # Truncate
                    
                    print(f"{stock_name:<12} {price:<8} {shares:<6} {strategy:<25}")
        
        print("\n" + "=" * 90)
        print("✅ ULTIMATE INTELLIGENCE ANALYSIS COMPLETE!")
        print("🚀 Multi-dimensional market analysis with global perspective!")

if __name__ == "__main__":
    # Initialize and run ultimate intelligence system
    ultimate_intel = UltimateMarketIntelligence()
    
    # Generate comprehensive report
    intelligence = ultimate_intel.generate_ultimate_intelligence_report()
    
    # Save report
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"data/ultimate_intelligence_{timestamp}.json", "w") as f:
            json.dump(intelligence, f, indent=2, default=str)
        print(f"📊 Intelligence report saved!")
    except:
        pass
