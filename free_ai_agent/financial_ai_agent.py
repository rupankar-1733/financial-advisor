# free_ai_agent/financial_ai_agent.py - Your FREE AI Financial Advisor
import os
import sys
import json
import re
from datetime import datetime
import requests

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing trading systems
from strategies.working_zones_system import WorkingZoneDetector
from data_sources.comprehensive_intelligence_system import UltimateMarketIntelligence
from utils.live_data_fetcher import LiveDataFetcher

class FreeFinancialAI:
    def __init__(self, use_local_llm=True):
        """Initialize your FREE Financial AI Agent"""
        self.use_local_llm = use_local_llm
        
        # Initialize your trading systems
        self.zone_detector = WorkingZoneDetector
        self.market_intel = UltimateMarketIntelligence()
        self.data_fetcher = LiveDataFetcher()
        
        # Your AI's financial knowledge base
        self.financial_knowledge = {
            'risk_management_rules': [
                "Never risk more than 2% of capital per trade",
                "Always set stop losses below support levels",
                "Position size based on distance to stop loss",
                "Diversify across maximum 5-6 stocks",
                "Keep 20% cash for opportunities"
            ],
            'technical_analysis_rules': [
                "Buy near support zones with volume confirmation",
                "Sell near resistance zones or take partial profits",
                "Follow trend direction - trend is your friend",
                "Wait for price to reach key zones before entering",
                "Use multiple timeframes for confirmation"
            ],
            'investment_principles': [
                "Invest in companies with strong fundamentals",
                "Dollar cost average in volatile markets",
                "Hold quality stocks for long term",
                "Rebalance portfolio quarterly",
                "Stay updated with company earnings and news"
            ],
            'indian_stock_insights': [
                "TCS: IT giant, stable growth, dividend paying",
                "RELIANCE: Oil to telecom, high beta stock",
                "INFY: Global IT services, US revenue exposure",
                "HDFCBANK: Leading private bank, NPA concerns watch",
                "ITC: FMCG + tobacco, regulatory risks"
            ]
        }
        
        # Conversation memory
        self.conversation_history = []
        
        print("🤖 FREE Financial AI Agent Initialized!")
        if use_local_llm:
            print("🔧 Using local LLM - completely FREE!")
        else:
            print("🌐 Using rule-based responses")
    
    def analyze_stock_with_tools(self, symbol, capital=50000):
        """Analyze stock using your advanced tools"""
        try:
            print(f"🔍 Analyzing {symbol} with ₹{capital:,} capital...")
            
            detector = self.zone_detector(symbol, capital)
            zones_data = detector.get_price_zones()
            
            if not zones_data:
                return f"Unable to analyze {symbol} - insufficient data"
            
            # Extract key analysis points
            analysis = {
                'symbol': symbol.replace('.NS', ''),
                'current_price': zones_data['current_price'],
                'max_shares': int(capital / zones_data['current_price']),
                'best_support': zones_data['support_zones'][0] if zones_data['support_zones'] else None,
                'best_resistance': zones_data['resistance_zones'][0] if zones_data['resistance_zones'] else None,
                'support_count': len(zones_data['support_zones']),
                'resistance_count': len(zones_data['resistance_zones'])
            }
            
            return analysis
            
        except Exception as e:
            return f"Analysis failed: {e}"
    
    def get_investment_recommendation(self, user_query, user_profile=None):
        """Generate investment recommendations based on query analysis"""
        
        # Extract key information from user query
        query_analysis = self.analyze_user_query(user_query)
        
        # Get stock analysis if specific stocks mentioned
        stock_data = {}
        if query_analysis['stocks_mentioned']:
            for stock in query_analysis['stocks_mentioned'][:2]:  # Limit to 2 stocks
                capital = user_profile.get('capital', 50000) if user_profile else 50000
                stock_data[stock] = self.analyze_stock_with_tools(stock, capital)
        
        # Generate personalized response
        response = self.generate_financial_response(query_analysis, stock_data, user_profile)
        
        return response
    
    def analyze_user_query(self, query):
        """Analyze user query to understand intent and extract information"""
        query_lower = query.lower()
        
        # Extract stocks mentioned
        stocks_mentioned = []
        stock_symbols = ['tcs', 'infy', 'reliance', 'hdfcbank', 'itc', 'hindunilvr', 'sbin']
        for symbol in stock_symbols:
            if symbol in query_lower:
                stocks_mentioned.append(f"{symbol.upper()}.NS")
        
        # Determine query type
        query_type = 'general'
        if any(word in query_lower for word in ['buy', 'invest', 'should i']):
            query_type = 'investment_advice'
        elif any(word in query_lower for word in ['trade', 'trading', 'short term']):
            query_type = 'trading_advice'
        elif any(word in query_lower for word in ['analyze', 'analysis', 'target', 'support']):
            query_type = 'technical_analysis'
        elif any(word in query_lower for word in ['risk', 'stop loss', 'position size']):
            query_type = 'risk_management'
        elif any(word in query_lower for word in ['market', 'sentiment', 'today']):
            query_type = 'market_overview'
        
        # Extract mentioned capital amount
        capital_mentioned = None
        capital_patterns = [r'₹(\d+(?:,\d+)*(?:k|l|cr)?)', r'(\d+)k', r'(\d+)l', r'(\d+)\s*lakh']
        for pattern in capital_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                if 'k' in query_lower:
                    capital_mentioned = int(float(amount_str)) * 1000
                elif 'l' in query_lower or 'lakh' in query_lower:
                    capital_mentioned = int(float(amount_str)) * 100000
                else:
                    capital_mentioned = int(float(amount_str))
                break
        
        return {
            'query_type': query_type,
            'stocks_mentioned': stocks_mentioned,
            'capital_mentioned': capital_mentioned,
            'original_query': query
        }
    
    def generate_financial_response(self, query_analysis, stock_data, user_profile):
        """Generate comprehensive financial response"""
        
        query_type = query_analysis['query_type']
        stocks = query_analysis['stocks_mentioned']
        capital = query_analysis['capital_mentioned'] or (user_profile.get('capital', 50000) if user_profile else 50000)
        
        # Start building response
        response = f"🤖 **AI Financial Advisor Analysis**\n\n"
        
        if query_type == 'investment_advice':
            response += self.generate_investment_advice(stocks, stock_data, capital, user_profile)
        elif query_type == 'trading_advice':
            response += self.generate_trading_advice(stocks, stock_data, capital)
        elif query_type == 'technical_analysis':
            response += self.generate_technical_analysis(stocks, stock_data)
        elif query_type == 'risk_management':
            response += self.generate_risk_management_advice(capital, user_profile)
        elif query_type == 'market_overview':
            response += self.generate_market_overview()
        else:
            response += self.generate_general_advice()
        
        # Add personalized footer
        if user_profile:
            risk_level = user_profile.get('risk_tolerance', 'moderate')
            response += f"\n\n💡 **Note**: Advice tailored for {risk_level} risk profile with ₹{capital:,} capital."
        
        response += "\n\n⚠️ **Disclaimer**: This is educational content only. Please do your own research before investing."
        
        return response
    
    def generate_investment_advice(self, stocks, stock_data, capital, user_profile):
        """Generate investment-specific advice"""
        advice = f"💰 **Investment Strategy (₹{capital:,} Capital)**\n\n"
        
        if stocks and stock_data:
            for stock in stocks:
                if stock in stock_data and isinstance(stock_data[stock], dict):
                    data = stock_data[stock]
                    stock_name = data['symbol']
                    current_price = data['current_price']
                    max_shares = data['max_shares']
                    
                    advice += f"📊 **{stock_name} Analysis:**\n"
                    advice += f"   💰 Current Price: ₹{current_price:.2f}\n"
                    advice += f"   📈 Max Shares for Capital: {max_shares}\n"
                    
                    # Position sizing recommendation
                    recommended_allocation = min(capital * 0.2, capital // len(stocks))  # Max 20% per stock
                    recommended_shares = int(recommended_allocation / current_price)
                    
                    advice += f"   🎯 Recommended Investment: ₹{recommended_allocation:,.0f} ({recommended_shares} shares)\n"
                    
                    if data['best_support']:
                        support_price = data['best_support']['price']
                        advice += f"   🔻 Key Support: ₹{support_price:.0f} (good entry zone)\n"
                    
                    if data['best_resistance']:
                        resistance_price = data['best_resistance']['price']
                        expected_return = ((resistance_price - current_price) / current_price) * 100
                        advice += f"   🔺 Target: ₹{resistance_price:.0f} ({expected_return:.1f}% upside)\n"
                    
                    advice += "\n"
        else:
            # General investment advice
            advice += "🎯 **General Investment Guidelines:**\n\n"
            for principle in self.financial_knowledge['investment_principles']:
                advice += f"   • {principle}\n"
            
            advice += f"\n💡 **For ₹{capital:,} capital, consider:**\n"
            advice += "   • Diversify across 3-5 quality stocks\n"
            advice += "   • Allocate max ₹10,000-15,000 per stock\n"
            advice += "   • Keep 20% cash for opportunities\n"
            advice += "   • Focus on fundamentally strong companies\n"
        
        return advice
    
    def generate_trading_advice(self, stocks, stock_data, capital):
        """Generate trading-specific advice"""
        advice = f"⚡ **Trading Strategy (₹{capital:,} Capital)**\n\n"
        
        if stocks and stock_data:
            for stock in stocks:
                if stock in stock_data and isinstance(stock_data[stock], dict):
                    data = stock_data[stock]
                    stock_name = data['symbol']
                    current_price = data['current_price']
                    
                    advice += f"📊 **{stock_name} Trading Setup:**\n"
                    
                    if data['best_support'] and data['best_resistance']:
                        support = data['best_support']['price']
                        resistance = data['best_resistance']['price']
                        
                        # Calculate risk-reward
                        stop_loss = support * 0.98  # 2% below support
                        risk_per_share = current_price - stop_loss
                        reward_per_share = resistance - current_price
                        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
                        
                        # Position sizing (2% portfolio risk)
                        risk_amount = capital * 0.02
                        position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                        
                        advice += f"   🎯 Entry: Wait for ₹{support:.0f} zone\n"
                        advice += f"   🔺 Target: ₹{resistance:.0f}\n"
                        advice += f"   🛑 Stop Loss: ₹{stop_loss:.0f}\n"
                        advice += f"   ⚖️ Risk:Reward = 1:{rr_ratio:.1f}\n"
                        advice += f"   📊 Position Size: {position_size} shares\n"
                        advice += f"   💰 Investment: ₹{position_size * support:,.0f}\n"
                        
                        if rr_ratio >= 3:
                            advice += f"   ✅ **Excellent trade setup!**\n"
                        elif rr_ratio >= 2:
                            advice += f"   🟡 **Good trade setup**\n"
                        else:
                            advice += f"   🔴 **Poor risk-reward - avoid**\n"
                    
                    advice += "\n"
        else:
            advice += "⚡ **Key Trading Principles:**\n\n"
            for rule in self.financial_knowledge['technical_analysis_rules']:
                advice += f"   • {rule}\n"
        
        return advice
    
    def generate_technical_analysis(self, stocks, stock_data):
        """Generate technical analysis"""
        analysis = "📊 **Technical Analysis**\n\n"
        
        if stocks and stock_data:
            for stock in stocks:
                if stock in stock_data and isinstance(stock_data[stock], dict):
                    data = stock_data[stock]
                    stock_name = data['symbol']
                    
                    analysis += f"🎯 **{stock_name} Technical View:**\n"
                    analysis += f"   📊 Support Levels: {data['support_count']} identified\n"
                    analysis += f"   📊 Resistance Levels: {data['resistance_count']} identified\n"
                    
                    if data['best_support']:
                        support = data['best_support']
                        analysis += f"   🔻 Strongest Support: ₹{support['price']:.0f} ({support['method']})\n"
                    
                    if data['best_resistance']:
                        resistance = data['best_resistance']
                        analysis += f"   🔺 Strongest Resistance: ₹{resistance['price']:.0f} ({resistance['method']})\n"
                    
                    analysis += "\n"
        
        return analysis
    
    def generate_risk_management_advice(self, capital, user_profile):
        """Generate risk management advice"""
        advice = f"🛡️ **Risk Management (₹{capital:,} Portfolio)**\n\n"
        
        max_risk_per_trade = capital * 0.02
        max_position_size = capital * 0.2
        emergency_fund = capital * 0.1
        
        advice += f"📋 **Your Risk Parameters:**\n"
        advice += f"   • Max risk per trade: ₹{max_risk_per_trade:,.0f} (2%)\n"
        advice += f"   • Max position size: ₹{max_position_size:,.0f} (20%)\n"
        advice += f"   • Emergency fund: ₹{emergency_fund:,.0f} (10%)\n\n"
        
        advice += "🎯 **Risk Management Rules:**\n"
        for rule in self.financial_knowledge['risk_management_rules']:
            advice += f"   • {rule}\n"
        
        return advice
    
    def generate_market_overview(self):
        """Generate market overview"""
        overview = "🌍 **Market Overview**\n\n"
        
        try:
            # Get basic market status
            status = self.data_fetcher.get_current_market_status()
            
            overview += f"📊 **Market Status**: {'🟢 OPEN' if status['is_open'] else '🔴 CLOSED'}\n"
            overview += f"🕒 **Current Time**: {status['current_time']}\n\n"
            
            overview += "💡 **Current Market Themes:**\n"
            overview += "   • Focus on quality stocks with strong earnings\n"
            overview += "   • Monitor global cues and FII/DII flows\n"
            overview += "   • Sector rotation from growth to value\n"
            overview += "   • Volatility expected around major events\n"
            
        except:
            overview += "📊 Market analysis temporarily unavailable\n"
        
        return overview
    
    def generate_general_advice(self):
        """Generate general financial advice"""
        advice = "💡 **General Financial Guidance**\n\n"
        advice += "I can help you with:\n"
        advice += "   📊 Stock analysis and recommendations\n"
        advice += "   💰 Investment strategies and portfolio allocation\n"
        advice += "   ⚡ Trading setups with entry/exit points\n"
        advice += "   🛡️ Risk management and position sizing\n"
        advice += "   📈 Technical analysis and support/resistance levels\n\n"
        advice += "💬 **Try asking:**\n"
        advice += "   • 'Analyze TCS for ₹50k investment'\n"
        advice += "   • 'Should I buy RELIANCE for trading?'\n"
        advice += "   • 'What's my risk for ₹1L portfolio?'\n"
        advice += "   • 'Give me 3 good stocks to invest'\n"
        
        return advice
    
    def chat(self, user_input, user_profile=None):
        """Main chat interface"""
        print(f"👤 User: {user_input}")
        
        # Generate response
        response = self.get_investment_recommendation(user_input, user_profile)
        
        # Store in conversation history
        self.conversation_history.append({
            'user': user_input,
            'ai': response,
            'timestamp': datetime.now()
        })
        
        print(f"\n🤖 AI Financial Advisor:\n{response}")
        
        return response

def start_free_financial_ai():
    """Start the FREE Financial AI Agent"""
    print("🚀 === FREE AI FINANCIAL ADVISOR ===")
    print("💡 Powered by your advanced trading algorithms!")
    print("=" * 60)
    
    # Initialize AI agent
    ai_agent = FreeFinancialAI()
    
    # Get user profile
    print("\n📋 Let's set up your investment profile:")
    try:
        capital = int(input("💰 Investment capital (₹): "))
    except:
        capital = 50000
    
    risk_tolerance = input("🎯 Risk tolerance (conservative/moderate/aggressive): ").lower() or 'moderate'
    
    user_profile = {
        'capital': capital,
        'risk_tolerance': risk_tolerance,
        'setup_time': datetime.now()
    }
    
    print(f"\n✅ Profile created! Capital: ₹{capital:,}, Risk: {risk_tolerance}")
    print("\n💬 Start asking questions! (type 'quit' to exit)")
    print("="*60)
    
    # Chat loop
    while True:
        try:
            user_input = input(f"\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using FREE AI Financial Advisor!")
                break
            
            if not user_input:
                continue
            
            # Get AI response
            response = ai_agent.chat(user_input, user_profile)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_free_financial_ai()
