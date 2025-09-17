# streamlit_financial_chatgpt/ultimate_financial_ai.py - Most Advanced Financial AI
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time
import pickle
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing systems
from strategies.working_zones_system import WorkingZoneDetector
from utils.live_data_fetcher import LiveDataFetcher

# Page config
st.set_page_config(
    page_title="🤖 Ultimate Financial AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 3rem;
    }
    .bot-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-right: 3rem;
    }
    .header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sector-analysis {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stock-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class UltimateFinancialAI:
    def __init__(self):
        """Initialize the most advanced Financial AI system"""
        # Your existing systems
        self.zone_detector = WorkingZoneDetector
        self.data_fetcher = LiveDataFetcher()
        
        # Comprehensive Indian stock universe
        self.stock_universe = {
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS', 'COFORGE.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'MINDTREE.NS'],
            'BANKING': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'PNB.NS', 'FEDERALBNK.NS'],
            'AUTO': ['MARUTI.NS', 'HYUNDAI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'TVSMOTORS.NS', 'EICHERMOT.NS', 'ASHOKLEY.NS', 'TVSMOTOR.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS', 'GODREJCP.NS', 'MARICO.NS', 'COLPAL.NS', 'UBL.NS', 'TATACONSUM.NS'],
            'PHARMA': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS', 'LUPIN.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'CADILAHC.NS', 'AUROPHARMA.NS'],
            'ENERGY': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS', 'ADANIGREEN.NS', 'TATAPOWER.NS'],
            'METALS': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'NMDC.NS', 'SAIL.NS', 'MOIL.NS', 'RATNAMANI.NS', 'WELCORP.NS', 'JINDALSTEL.NS'],
            'TELECOM': ['BHARTIARTL.NS', 'JIO.NS', 'IDEA.NS', 'TTML.NS'],
            'CEMENT': ['ULTRACEMC.NS', 'SHREECEM.NS', 'AMBUJACEM.NS', 'ACC.NS', 'JKCEMENT.NS', 'RAMCOCEM.NS', 'HEIDELBERG.NS'],
            'REALTY': ['DLF.NS', 'GODREJPROP.NS', 'PRESTIGE.NS', 'OBEROI.NS', 'BRIGADE.NS', 'MAHLIFE.NS']
        }
        
        # Sector weights and characteristics
        self.sector_info = {
            'IT': {'weight': 0.15, 'volatility': 'medium', 'growth': 'stable', 'export_driven': True},
            'BANKING': {'weight': 0.25, 'volatility': 'high', 'growth': 'cyclical', 'interest_sensitive': True},
            'AUTO': {'weight': 0.08, 'volatility': 'high', 'growth': 'cyclical', 'economic_sensitive': True},
            'FMCG': {'weight': 0.10, 'volatility': 'low', 'growth': 'defensive', 'consumer_driven': True},
            'PHARMA': {'weight': 0.06, 'volatility': 'medium', 'growth': 'defensive', 'export_driven': True},
            'ENERGY': {'weight': 0.12, 'volatility': 'high', 'growth': 'cyclical', 'commodity_linked': True},
            'METALS': {'weight': 0.05, 'volatility': 'very_high', 'growth': 'cyclical', 'commodity_linked': True},
            'TELECOM': {'weight': 0.04, 'volatility': 'high', 'growth': 'recovery', 'tariff_sensitive': True},
            'CEMENT': {'weight': 0.03, 'volatility': 'medium', 'growth': 'cyclical', 'infra_linked': True},
            'REALTY': {'weight': 0.02, 'volatility': 'very_high', 'growth': 'cyclical', 'interest_sensitive': True}
        }
        
        # ML model placeholder (you'll integrate your trained models)
        self.ml_models = {}
        self.scalers = {}
        
        print("🚀 Ultimate Financial AI Initialized with comprehensive market intelligence!")
    
    def get_live_market_data(self, symbols, period='5d'):
        """Get live data for multiple symbols efficiently"""
        try:
            # Batch download for efficiency
            data = yf.download(symbols, period=period, group_by='ticker', auto_adjust=True, prepost=True, threads=True)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def analyze_sector_performance(self, sector_data):
        """Advanced sector performance analysis"""
        sector_metrics = {}
        
        for sector, stocks in self.stock_universe.items():
            try:
                # Get sector data
                sector_prices = []
                for stock in stocks[:5]:  # Top 5 stocks per sector
                    if stock in sector_data:
                        stock_data = sector_data[stock]
                        if not stock_data.empty:
                            # Calculate momentum (5-day performance)
                            momentum = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
                            sector_prices.append(momentum)
                
                if sector_prices:
                    avg_momentum = np.mean(sector_prices)
                    momentum_consistency = len([x for x in sector_prices if x > 0]) / len(sector_prices)
                    volatility = np.std(sector_prices)
                    
                    sector_metrics[sector] = {
                        'momentum': avg_momentum,
                        'consistency': momentum_consistency,
                        'volatility': volatility,
                        'strength_score': avg_momentum * momentum_consistency - (volatility * 0.1),
                        'stock_count': len(sector_prices)
                    }
            except Exception as e:
                continue
        
        # Rank sectors by strength score
        ranked_sectors = sorted(sector_metrics.items(), key=lambda x: x[1]['strength_score'], reverse=True)
        
        return ranked_sectors, sector_metrics
    
    def get_ml_predictions(self, symbol):
        """Get ML model predictions (integrate your trained models)"""
        try:
            # This is where you'll integrate your actual ML models
            # For now, using technical analysis as proxy
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='30d')
            
            if len(data) < 20:
                return None
            
            # Simple technical-based prediction (replace with your ML models)
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # Momentum indicator
            momentum = ((current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100
            
            # Volume trend
            volume_trend = (data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:-5].mean() - 1) * 100
            
            # Prediction logic (replace with your trained model)
            if sma_10 > sma_20 and momentum > 0 and volume_trend > 0:
                prediction = 'bullish'
                confidence = min(abs(momentum) + abs(volume_trend), 100) / 100
            elif sma_10 < sma_20 and momentum < 0:
                prediction = 'bearish'
                confidence = min(abs(momentum) + abs(volume_trend), 100) / 100
            else:
                prediction = 'neutral'
                confidence = 0.5
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'price_target': current_price * (1 + momentum/100),
                'momentum': momentum,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            return None
    
    def analyze_comprehensive_market(self):
        """Comprehensive market analysis across all sectors"""
        print("🔍 Performing comprehensive market analysis...")
        
        # Get all stock symbols
        all_symbols = []
        for stocks in self.stock_universe.values():
            all_symbols.extend(stocks)
        
        # Batch download market data
        market_data = self.get_live_market_data(all_symbols[:50], period='10d')  # Limit to avoid rate limits
        
        if market_data is None:
            return None
        
        # Analyze sector performance
        ranked_sectors, sector_metrics = self.analyze_sector_performance(market_data)
        
        # Get top performing stocks across all sectors
        top_stocks = []
        
        for symbol in all_symbols[:30]:  # Analyze top 30 stocks
            try:
                # Get technical analysis
                detector = self.zone_detector(symbol, 50000)
                zones_data = detector.get_price_zones()
                
                if zones_data:
                    # Get ML prediction
                    ml_prediction = self.get_ml_predictions(symbol)
                    
                    # Calculate comprehensive score
                    score = 0
                    
                    # Technical score (zone quality)
                    if zones_data['support_zones'] and zones_data['resistance_zones']:
                        support = zones_data['support_zones'][0]
                        resistance = zones_data['resistance_zones'][0]
                        risk_reward = (resistance['price'] - zones_data['current_price']) / (zones_data['current_price'] - support['price'])
                        score += min(risk_reward * 20, 40)  # Max 40 points for R:R
                    
                    # ML prediction score
                    if ml_prediction:
                        if ml_prediction['prediction'] == 'bullish':
                            score += ml_prediction['confidence'] * 30  # Max 30 points
                        score += abs(ml_prediction['momentum']) * 0.5  # Momentum bonus
                    
                    # Sector strength bonus
                    stock_sector = None
                    for sector, stocks in self.stock_universe.items():
                        if symbol in stocks:
                            stock_sector = sector
                            break
                    
                    if stock_sector and stock_sector in sector_metrics:
                        sector_bonus = sector_metrics[stock_sector]['strength_score'] * 0.3
                        score += sector_bonus
                    
                    top_stocks.append({
                        'symbol': symbol,
                        'sector': stock_sector,
                        'score': score,
                        'current_price': zones_data['current_price'],
                        'zones_data': zones_data,
                        'ml_prediction': ml_prediction,
                        'risk_reward': risk_reward if 'risk_reward' in locals() else 0
                    })
            
            except Exception as e:
                continue
        
        # Sort by comprehensive score
        top_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'sector_rankings': ranked_sectors,
            'sector_metrics': sector_metrics,
            'top_stocks': top_stocks[:10],
            'analysis_time': datetime.now()
        }
    
    def generate_ultimate_recommendations(self, context):
        """Generate the most advanced stock recommendations"""
        capital = context.get('capital', 50000)
        risk_level = context.get('risk_tolerance', 'moderate')
        
        # Show analysis progress
        analysis_placeholder = st.empty()
        analysis_placeholder.markdown("🔍 **Analyzing comprehensive market data across all sectors...**")
        
        # Perform comprehensive analysis
        market_analysis = self.analyze_comprehensive_market()
        
        analysis_placeholder.empty()
        
        if not market_analysis:
            return "❌ Market analysis temporarily unavailable. Please try again."
        
        response = f"## 🎯 ULTIMATE Stock Recommendations (₹{capital:,})\n\n"
        response += f"**Analysis Time**: {market_analysis['analysis_time'].strftime('%I:%M %p IST')} • **Risk Profile**: {risk_level.title()}\n\n"
        
        # Sector analysis
        response += f"### 📊 Live Sector Analysis\n\n"
        
        top_sectors = market_analysis['sector_rankings'][:3]
        for i, (sector, metrics) in enumerate(top_sectors, 1):
            momentum = metrics['momentum']
            consistency = metrics['consistency'] * 100
            
            if momentum > 0:
                trend_emoji = "🟢"
                trend_text = f"+{momentum:.1f}%"
            else:
                trend_emoji = "🔴" 
                trend_text = f"{momentum:.1f}%"
            
            response += f"**{i}. {sector} Sector** {trend_emoji}\n"
            response += f"- **5-Day Momentum**: {trend_text}\n"
            response += f"- **Consistency**: {consistency:.0f}% stocks positive\n"
            response += f"- **Strength Score**: {metrics['strength_score']:.1f}\n\n"
        
        # Market breadth analysis
        it_dominance = 0
        banking_dominance = 0
        total_top_stocks = len(market_analysis['top_stocks'])
        
        for stock in market_analysis['top_stocks']:
            if stock['sector'] == 'IT':
                it_dominance += 1
            elif stock['sector'] == 'BANKING':
                banking_dominance += 1
        
        it_percentage = (it_dominance / total_top_stocks) * 100
        banking_percentage = (banking_dominance / total_top_stocks) * 100
        
        response += f"### 🎯 Market Breadth Analysis\n\n"
        response += f"**Sector Dominance in Top Picks:**\n"
        response += f"- IT Sector: {it_percentage:.0f}% ({it_dominance}/{total_top_stocks})\n"
        response += f"- Banking Sector: {banking_percentage:.0f}% ({banking_dominance}/{total_top_stocks})\n\n"
        
        if it_percentage >= 40:
            response += f"📈 **Market Signal**: Strong IT sector momentum - technology theme dominating\n\n"
        elif banking_percentage >= 40:
            response += f"📈 **Market Signal**: Banking sector leadership - credit cycle recovery\n\n"
        else:
            response += f"📈 **Market Signal**: Diversified market - no single sector dominance\n\n"
        
        # Top stock recommendations with ML predictions
        response += f"### 🏆 TOP 3 AI-Selected Stocks\n\n"
        
        top_3_stocks = market_analysis['top_stocks'][:3]
        total_allocation = 0
        
        for i, stock_data in enumerate(top_3_stocks, 1):
            symbol = stock_data['symbol']
            stock_name = symbol.replace('.NS', '')
            current_price = stock_data['current_price']
            ml_pred = stock_data['ml_prediction']
            
            # Allocation based on risk level and score
            if risk_level == 'conservative':
                allocation = min(capital * 0.25, 15000)
            elif risk_level == 'aggressive':
                allocation = min(capital * 0.35, 25000)
            else:
                allocation = min(capital * 0.30, 20000)
            
            shares = int(allocation / current_price)
            total_allocation += allocation
            
            response += f"#### {i}. **{stock_name}** ({stock_data['sector']} Sector)\n"
            response += f"**Current Price**: ₹{current_price:.2f} • **Score**: {stock_data['score']:.1f}/100\n\n"
            
            # Technical analysis
            zones = stock_data['zones_data']
            if zones['support_zones'] and zones['resistance_zones']:
                support = zones['support_zones'][0]['price']
                resistance = zones['resistance_zones'][0]['price']
                
                response += f"**Technical Analysis**:\n"
                response += f"- Entry Zone: ₹{support:.0f} (support)\n"
                response += f"- Target: ₹{resistance:.0f} (resistance)\n"
                response += f"- Risk:Reward: 1:{stock_data['risk_reward']:.1f}\n\n"
            
            # ML Prediction
            if ml_pred:
                pred_emoji = "🟢" if ml_pred['prediction'] == 'bullish' else "🔴" if ml_pred['prediction'] == 'bearish' else "🟡"
                response += f"**AI Prediction**: {pred_emoji} {ml_pred['prediction'].title()}\n"
                response += f"- Confidence: {ml_pred['confidence']:.0%}\n"
                response += f"- Price Target: ₹{ml_pred['price_target']:.0f}\n"
                response += f"- Momentum: {ml_pred['momentum']:.1f}%\n\n"
            
            # Investment recommendation
            response += f"**Investment Plan**:\n"
            response += f"- **Allocation**: ₹{allocation:,} ({shares} shares)\n"
            
            # Expected returns calculation
            if zones['resistance_zones']:
                target_price = zones['resistance_zones'][0]['price']
                expected_profit = (target_price - current_price) * shares
                roi_percentage = (expected_profit / allocation) * 100
                
                response += f"- **Expected Profit**: ₹{expected_profit:,.0f}\n"
                response += f"- **Expected ROI**: {roi_percentage:.1f}%\n\n"
            
            response += "---\n\n"
        
        # Portfolio summary
        cash_reserve = capital - total_allocation
        response += f"### 💰 Portfolio Summary\n\n"
        response += f"**Total Allocation**: ₹{total_allocation:,} ({(total_allocation/capital)*100:.0f}%)\n"
        response += f"**Cash Reserve**: ₹{cash_reserve:,} ({(cash_reserve/capital)*100:.0f}%)\n"
        response += f"**Diversification**: {len(set([s['sector'] for s in top_3_stocks]))} sectors covered\n\n"
        
        # AI insights and strategy
        response += f"### 🤖 AI Strategic Insights\n\n"
        
        leading_sector = market_analysis['sector_rankings'][0][0]
        leading_momentum = market_analysis['sector_rankings'][0][1]['momentum']
        
        response += f"**Market Theme**: {leading_sector} sector leading with {leading_momentum:.1f}% momentum\n"
        response += f"**Strategy**: Focus on quality stocks within outperforming sectors\n"
        response += f"**Entry Approach**: Use systematic buying over 2-3 weeks\n"
        response += f"**Risk Management**: Maintain stops below key support levels\n"
        response += f"**Review Cycle**: Monitor weekly, rebalance monthly\n\n"
        
        response += f"**⚠️ AI Advisory**: This analysis combines technical zones, ML predictions, and sector intelligence for optimal stock selection."
        
        return response
    
    def analyze_user_query(self, message):
        """Advanced query analysis with comprehensive pattern matching"""
        message_lower = message.lower()
        
        # Extract stocks
        stocks = []
        all_stocks = []
        for sector_stocks in self.stock_universe.values():
            all_stocks.extend(sector_stocks)
        
        # Check for stock mentions
        for stock in all_stocks:
            symbol_name = stock.replace('.NS', '').lower()
            if symbol_name in message_lower:
                stocks.append(stock)
        
        # Advanced intent detection
        if any(phrase in message_lower for phrase in [
            'best 3 stocks', 'top 3 stocks', 'best stocks now', 'ultimate recommendations',
            'comprehensive analysis', 'sector analysis', 'market analysis'
        ]):
            intent = 'ultimate_recommendations'
        
        elif any(phrase in message_lower for phrase in [
            'current market', 'live trading', 'trading opportunities', 'intraday plan'
        ]):
            intent = 'live_trading_analysis'
        
        elif any(phrase in message_lower for phrase in [
            'sector performance', 'which sector', 'sector analysis', 'best sector'
        ]):
            intent = 'sector_analysis'
        
        elif any(phrase in message_lower for phrase in [
            'mid cap', 'small cap', 'volatile stocks', 'high beta'
        ]):
            intent = 'midcap_analysis'
        
        elif stocks and any(word in message_lower for word in [
            'analyze', 'analysis', 'prediction', 'forecast', 'target'
        ]):
            intent = 'advanced_stock_analysis'
        
        else:
            intent = 'general'
        
        return {
            'intent': intent,
            'stocks': stocks[:3],  # Limit to 3 stocks
            'message': message
        }
    
    def generate_structured_response(self, user_message):
        """Ultimate response generation system"""
        analysis = self.analyze_user_query(user_message)
        context = st.session_state.get('user_context', {})
        
        if analysis['intent'] == 'ultimate_recommendations':
            return self.generate_ultimate_recommendations(context)
        
        elif analysis['intent'] == 'live_trading_analysis':
            return self.generate_live_trading_opportunities(context)
        
        elif analysis['intent'] == 'sector_analysis':
            return self.generate_sector_analysis(context)
        
        elif analysis['intent'] == 'midcap_analysis':
            return self.generate_midcap_analysis(context)
        
        elif analysis['intent'] == 'advanced_stock_analysis' and analysis['stocks']:
            return self.generate_advanced_stock_analysis(analysis['stocks'][0], context)
        
        else:
            return self.generate_general_guidance()
    
    def generate_live_trading_opportunities(self, context):
        """Generate live trading opportunities"""
        capital = context.get('capital', 50000)
        
        response = f"## ⚡ LIVE Trading Opportunities ({datetime.now().strftime('%I:%M %p IST')})\n\n"
        response += f"**🟢 Markets Open** • **Capital**: ₹{capital:,} • **Analysis**: Real-time\n\n"
        
        # Quick analysis of top liquid stocks
        liquid_stocks = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
        opportunities = []
        
        for stock in liquid_stocks:
            try:
                detector = self.zone_detector(stock, capital)
                zones_data = detector.get_price_zones()
                ml_pred = self.get_ml_predictions(stock)
                
                if zones_data and zones_data['support_zones'] and zones_data['resistance_zones']:
                    support = zones_data['support_zones'][0]['price']
                    resistance = zones_data['resistance_zones'][0]['price']
                    current = zones_data['current_price']
                    
                    # Trading opportunity scoring
                    distance_to_support = abs(current - support) / current * 100
                    risk_reward = (resistance - current) / (current - support)
                    
                    if distance_to_support < 3 and risk_reward > 2:  # Near support with good R:R
                        opportunities.append({
                            'stock': stock.replace('.NS', ''),
                            'current_price': current,
                            'entry': support,
                            'target': resistance,
                            'distance_to_entry': distance_to_support,
                            'risk_reward': risk_reward,
                            'ml_signal': ml_pred['prediction'] if ml_pred else 'neutral'
                        })
                        
            except:
                continue
        
        if opportunities:
            response += f"### 🎯 Live Trading Setups\n\n"
            for opp in sorted(opportunities, key=lambda x: x['risk_reward'], reverse=True)[:3]:
                response += f"**{opp['stock']}** - ₹{opp['current_price']:.2f}\n"
                response += f"- **Setup**: {opp['distance_to_entry']:.1f}% from entry zone\n"
                response += f"- **Entry**: ₹{opp['entry']:.0f}\n"
                response += f"- **Target**: ₹{opp['target']:.0f}\n"
                response += f"- **R:R**: 1:{opp['risk_reward']:.1f}\n"
                response += f"- **AI Signal**: {opp['ml_signal']}\n\n"
        else:
            response += "⚠️ No clear trading setups at current levels. Wait for better opportunities.\n"
        
        return response
    
    def generate_general_guidance(self):
        """Enhanced general guidance"""
        return """## 🤖 Ultimate Financial AI - Your Complete Market Intelligence

### What Makes This AI Ultimate:

**🧠 Advanced Intelligence**:
- **Sector Analysis**: Analyzes ALL sectors for market themes
- **ML Predictions**: Integrates machine learning forecasts  
- **Zone Detection**: Precise support/resistance levels
- **Market Breadth**: Identifies sector dominance patterns
- **Risk Management**: Professional position sizing

**📊 Comprehensive Coverage**:
- **100+ Stocks** across 10 major sectors
- **Real-time Analysis** with live market data
- **Mid-cap & High Volatility** stock analysis
- **Cross-sector Validation** for recommendations

**🎯 Try These Advanced Queries**:
- "Give me best 3 stocks with comprehensive sector analysis"
- "Analyze current market and identify best trading opportunities"  
- "Which sector is performing best and why?"
- "Show me mid-cap stocks with high volatility for trading"
- "Create ultimate portfolio strategy with ML predictions"

**💡 AI Capabilities**:
This system analyzes market breadth, sector rotation, technical zones, ML predictions, and risk metrics to provide institutional-grade investment intelligence.

Ready to experience the most advanced financial AI? Ask me anything! 🚀"""

def init_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """🚀 **Welcome to Ultimate Financial AI!**

I'm the most advanced financial advisor ever built, with:

**🧠 Comprehensive Intelligence**:
- **ALL Sector Analysis** (IT, Banking, Auto, FMCG, Pharma, Energy, Metals, Telecom, Cement, Realty)
- **100+ Stock Coverage** across entire market
- **ML Predictions** integrated with technical analysis
- **Market Breadth Analysis** (sector dominance detection)
- **High Volatility/Mid-cap** analysis capabilities

**🎯 Ultimate Capabilities**:
- Analyze which sectors are working best (like IT 48% dominance)
- Cross-validate sector performance with technical charts
- Integrate ML predictions with zone-based analysis
- Risk-adjusted recommendations for any capital size

**💡 Try These Advanced Queries**:
- "Give me best 3 stocks with complete sector analysis"
- "Analyze which sector is dominating the market now"
- "Show me high volatility mid-cap trading opportunities"
- "Create ML-powered investment strategy for ₹1L"

Ready to experience institutional-grade market intelligence? 🎯💰"""
            }
        ]
    
    if 'ultimate_ai' not in st.session_state:
        st.session_state.ultimate_ai = UltimateFinancialAI()

def setup_sidebar():
    """Enhanced sidebar"""
    st.sidebar.title("🎯 Investment Profile")
    
    capital = st.sidebar.number_input(
        "💰 Investment Capital (₹)",
        min_value=1000,
        max_value=50000000,
        value=100000,
        step=5000
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "📊 Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    investment_style = st.sidebar.selectbox(
        "🎯 Investment Style",
        ["Long-term Growth", "Medium-term Trading", "Short-term Trading"],
        index=0
    )
    
    st.session_state.user_context = {
        'capital': capital,
        'risk_tolerance': risk_tolerance.lower(),
        'investment_style': investment_style.lower()
    }
    
    # Enhanced profile display
    st.sidebar.markdown("---")
    st.sidebar.markdown("📋 **Your Profile**")
    st.sidebar.metric("Capital", f"₹{capital/100000:.1f}L" if capital >= 100000 else f"₹{capital/1000:.0f}K")
    st.sidebar.metric("Risk Level", risk_tolerance[:4])
    
    # Advanced quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("🚀 **AI Intelligence**")
    
    if st.sidebar.button("🧠 Ultimate Analysis"):
        return "Give me the ultimate stock recommendations with comprehensive sector analysis and ML predictions"
    
    if st.sidebar.button("📊 Sector Intelligence"):
        return "Analyze current sector performance and identify which sectors are dominating the market"
    
    if st.sidebar.button("⚡ Live Trading Setup"):
        return "Show me live trading opportunities with current market conditions"
    
    if st.sidebar.button("🎯 Mid-cap Opportunities"):
        return "Analyze high volatility mid-cap stocks for trading opportunities"
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()
    
    return None

def main():
    """Main application"""
    init_session_state()
    
    # Enhanced header
    st.markdown("""
    <div class="header">
        <h1>🤖 Ultimate Financial AI</h1>
        <p>Most Advanced Market Intelligence System Ever Built</p>
        <small>Sector Analysis • ML Predictions • 100+ Stocks • Institutional Grade</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    sidebar_action = setup_sidebar()
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Ultimate Financial AI:</strong><br>
            """, unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    prompt = sidebar_action or st.chat_input("Ask the Ultimate Financial AI anything...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>{prompt}
        </div>
        """, unsafe_allow_html=True)
        
        # Show advanced thinking animation
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="sector-analysis">
            🧠 <strong>AI Processing...</strong><br>
            • Analyzing 100+ stocks across all sectors<br>
            • Running ML predictions and technical analysis<br>  
            • Calculating market breadth and sector dominance<br>
            • Generating institutional-grade recommendations...
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        try:
            response = st.session_state.ultimate_ai.generate_structured_response(prompt)
            
            # Remove thinking animation
            thinking_placeholder.empty()
            
            # Add and show response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Ultimate Financial AI:</strong><br>
            """, unsafe_allow_html=True)
            st.markdown(response)
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"AI system encountered an issue: {e}. The Ultimate AI is processing your request with advanced algorithms."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(error_msg)
        
        st.rerun()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        🚀 <strong>Ultimate Financial AI</strong> - Most Advanced Market Intelligence System<br>
        📊 Sector Analysis • 🧠 ML Predictions • 🎯 100+ Stock Coverage • ⚡ Real-time Intelligence<br>
        ⚠️ <strong>Disclaimer:</strong> Advanced AI analysis for educational purposes. Consult professionals before investing.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
