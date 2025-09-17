# simple_financial_chatgpt/app.py - Your Financial ChatGPT (Simplified)
import streamlit as st
import sys
import os
from datetime import datetime
import time
import json

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your working systems
from strategies.working_zones_system import WorkingZoneDetector
from utils.live_data_fetcher import LiveDataFetcher

# Page config
st.set_page_config(
    page_title="🤖 AI Financial ChatGPT",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like design
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    .bot-message {
        background: #f8f9fa;
        margin-right: 2rem;
    }
    .header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class SimpleFinancialChatGPT:
    def __init__(self):
        """Initialize Simple Financial ChatGPT"""
        self.zone_detector = WorkingZoneDetector
        self.data_fetcher = LiveDataFetcher()
        
        # Financial knowledge base
        self.stock_info = {
            'TCS': 'Leading IT services company, stable dividend payer, US revenue exposure',
            'INFY': 'Global IT services, strong fundamentals, consistent performer',
            'RELIANCE': 'Oil-to-digital conglomerate, high beta, diversified business',
            'HDFCBANK': 'Leading private sector bank, interest rate sensitive',
            'ITC': 'FMCG giant with tobacco business, regulatory risks'
        }
        
        self.trading_rules = [
            "Never risk more than 2% of capital per trade",
            "Always use stop losses below support levels",
            "Buy near support zones, sell near resistance",
            "Position size based on risk tolerance",
            "Keep 20% cash for opportunities"
        ]
    
    def analyze_user_query(self, message):
        """Analyze what user is asking"""
        message_lower = message.lower()
        
        # Extract stocks mentioned
        stocks = []
        for symbol in ['tcs', 'infy', 'reliance', 'hdfcbank', 'itc']:
            if symbol in message_lower:
                stocks.append(f"{symbol.upper()}.NS")
        
        # Determine intent
        if any(word in message_lower for word in ['buy', 'invest', 'should i']):
            intent = 'investment'
        elif any(word in message_lower for word in ['trade', 'trading']):
            intent = 'trading'
        elif any(word in message_lower for word in ['analyze', 'analysis', 'target']):
            intent = 'analysis'
        elif any(word in message_lower for word in ['risk', 'stop loss']):
            intent = 'risk'
        else:
            intent = 'general'
        
        return {'intent': intent, 'stocks': stocks, 'message': message}
    
    def get_stock_analysis(self, stock, capital):
        """Get detailed stock analysis"""
        try:
            detector = self.zone_detector(stock, capital)
            zones_data = detector.get_price_zones()
            
            if zones_data:
                return {
                    'current_price': zones_data['current_price'],
                    'support_zones': zones_data['support_zones'][:2],
                    'resistance_zones': zones_data['resistance_zones'][:2],
                    'max_shares': int(capital / zones_data['current_price'])
                }
            return None
        except Exception as e:
            return f"Analysis failed: {e}"
    
    def generate_response(self, query_analysis, context):
        """Generate comprehensive response"""
        intent = query_analysis['intent']
        stocks = query_analysis['stocks']
        message = query_analysis['message']
        
        capital = context.get('capital', 50000)
        risk_level = context.get('risk_tolerance', 'moderate')
        
        response = ""
        
        if intent == 'investment':
            response = self.generate_investment_advice(stocks, capital, risk_level)
        elif intent == 'trading':
            response = self.generate_trading_advice(stocks, capital)
        elif intent == 'analysis':
            response = self.generate_technical_analysis(stocks, capital)
        elif intent == 'risk':
            response = self.generate_risk_advice(capital, risk_level)
        else:
            response = self.generate_general_help()
        
        return response
    
    def generate_investment_advice(self, stocks, capital, risk_level):
        """Generate investment recommendations"""
        response = f"## 💰 Investment Strategy (₹{capital:,})\n\n"
        
        if stocks:
            for stock in stocks[:2]:  # Limit to 2 stocks
                stock_name = stock.replace('.NS', '')
                response += f"### 📊 {stock_name} Analysis\n\n"
                
                # Get live analysis
                analysis = self.get_stock_analysis(stock, capital)
                
                if isinstance(analysis, dict):
                    current_price = analysis['current_price']
                    max_shares = analysis['max_shares']
                    
                    # Position sizing based on risk level
                    if risk_level == 'conservative':
                        allocation = min(capital * 0.15, 15000)
                    elif risk_level == 'aggressive':
                        allocation = min(capital * 0.25, 25000)
                    else:
                        allocation = min(capital * 0.2, 20000)
                    
                    shares = int(allocation / current_price)
                    
                    response += f"**Current Price:** ₹{current_price:.2f}\n\n"
                    response += f"**Recommendation:** Invest ₹{allocation:,} ({shares} shares)\n\n"
                    
                    if analysis['support_zones']:
                        support = analysis['support_zones'][0]['price']
                        response += f"**Entry Zone:** ₹{support:.0f} (support level)\n\n"
                    
                    if analysis['resistance_zones']:
                        target = analysis['resistance_zones'][0]['price']
                        upside = ((target - current_price) / current_price) * 100
                        response += f"**Target:** ₹{target:.0f} ({upside:.1f}% upside potential)\n\n"
                    
                    # Add stock context
                    if stock_name in self.stock_info:
                        response += f"**About {stock_name}:** {self.stock_info[stock_name]}\n\n"
                else:
                    response += f"Analysis temporarily unavailable for {stock_name}\n\n"
        else:
            # General investment advice
            response += f"### For {risk_level.title()} Investor with ₹{capital:,}\n\n"
            response += "**Recommended Allocation:**\n\n"
            
            if risk_level == 'conservative':
                response += "- 70% Large-cap stocks (TCS, HDFCBANK)\n"
                response += "- 20% Index funds\n"
                response += "- 10% Cash/Emergency fund\n\n"
            elif risk_level == 'aggressive':
                response += "- 50% Growth stocks (TCS, RELIANCE)\n"
                response += "- 30% Mid-cap stocks\n"
                response += "- 20% High-growth sectors\n\n"
            else:
                response += "- 60% Large-cap stocks (TCS, INFY)\n"
                response += "- 25% Mid-cap diversification\n"
                response += "- 15% Cash for opportunities\n\n"
            
            response += "**Top Stock Recommendations:**\n\n"
            response += "1. **TCS** - Stable IT giant, good dividends\n"
            response += "2. **RELIANCE** - Diversified business model\n"
            response += "3. **HDFCBANK** - Leading private bank\n\n"
        
        return response
    
    def generate_trading_advice(self, stocks, capital):
        """Generate trading recommendations"""
        response = f"## ⚡ Trading Strategy (₹{capital:,})\n\n"
        
        if stocks:
            for stock in stocks[:2]:
                stock_name = stock.replace('.NS', '')
                response += f"### 📊 {stock_name} Trading Setup\n\n"
                
                analysis = self.get_stock_analysis(stock, capital)
                
                if isinstance(analysis, dict):
                    current = analysis['current_price']
                    
                    if analysis['support_zones'] and analysis['resistance_zones']:
                        support = analysis['support_zones'][0]['price']
                        resistance = analysis['resistance_zones'][0]['price']
                        
                        # Calculate trade metrics
                        entry = support
                        target = resistance
                        stop_loss = support * 0.98
                        
                        risk_per_share = entry - stop_loss
                        reward_per_share = target - entry
                        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
                        
                        # Position sizing (2% portfolio risk)
                        risk_amount = capital * 0.02
                        position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                        investment = position_size * entry
                        
                        response += f"**Entry Price:** ₹{entry:.0f} (at support)\n"
                        response += f"**Target Price:** ₹{target:.0f}\n"
                        response += f"**Stop Loss:** ₹{stop_loss:.0f}\n"
                        response += f"**Risk:Reward Ratio:** 1:{rr_ratio:.1f}\n"
                        response += f"**Position Size:** {position_size} shares\n"
                        response += f"**Investment Amount:** ₹{investment:,.0f}\n\n"
                        
                        # Trade quality assessment
                        if rr_ratio >= 3:
                            response += "**Trade Quality:** 🟢 Excellent setup!\n\n"
                        elif rr_ratio >= 2:
                            response += "**Trade Quality:** 🟡 Good setup\n\n"
                        else:
                            response += "**Trade Quality:** 🔴 Poor risk/reward - avoid\n\n"
        else:
            response += "### Key Trading Principles\n\n"
            for i, rule in enumerate(self.trading_rules, 1):
                response += f"{i}. {rule}\n"
            response += "\n"
        
        return response
    
    def generate_technical_analysis(self, stocks, capital):
        """Generate technical analysis"""
        response = f"## 📊 Technical Analysis\n\n"
        
        if stocks:
            for stock in stocks:
                stock_name = stock.replace('.NS', '')
                analysis = self.get_stock_analysis(stock, capital)
                
                if isinstance(analysis, dict):
                    response += f"### {stock_name} Technical View\n\n"
                    response += f"**Current Price:** ₹{analysis['current_price']:.2f}\n\n"
                    
                    response += "**Support Levels:**\n"
                    for i, support in enumerate(analysis['support_zones'], 1):
                        response += f"{i}. ₹{support['price']:.0f} ({support['method']})\n"
                    response += "\n"
                    
                    response += "**Resistance Levels:**\n"
                    for i, resistance in enumerate(analysis['resistance_zones'], 1):
                        response += f"{i}. ₹{resistance['price']:.0f} ({resistance['method']})\n"
                    response += "\n"
        
        return response
    
    def generate_risk_advice(self, capital, risk_level):
        """Generate risk management advice"""
        response = f"## 🛡️ Risk Management (₹{capital:,})\n\n"
        
        max_risk_per_trade = capital * 0.02
        max_position = capital * 0.2
        emergency_fund = capital * 0.1
        
        response += "### Your Risk Parameters\n\n"
        response += f"**Max Risk per Trade:** ₹{max_risk_per_trade:,.0f} (2% of portfolio)\n"
        response += f"**Max Position Size:** ₹{max_position:,.0f} (20% of portfolio)\n"
        response += f"**Emergency Fund:** ₹{emergency_fund:,.0f} (10% of portfolio)\n\n"
        
        response += "### Risk Management Rules\n\n"
        for i, rule in enumerate(self.trading_rules, 1):
            response += f"{i}. {rule}\n"
        
        return response
    
    def generate_general_help(self):
        """Generate general help"""
        response = """## 🤖 Welcome to AI Financial ChatGPT!

### What I Can Help You With:

**📊 Stock Analysis**
- Real-time support and resistance levels
- Technical analysis with entry/exit points
- Risk-reward calculations

**💰 Investment Planning**
- Portfolio allocation strategies
- Capital deployment recommendations
- Risk-based position sizing

**⚡ Trading Setups**
- Precise entry and exit zones
- Stop loss calculations
- Position sizing based on risk

### Try Asking:
- "Analyze TCS for ₹50k investment"
- "Should I buy RELIANCE for trading?"
- "What's my risk management for ₹1L?"
- "Give me 3 good stocks to invest"

Ready to help you make better financial decisions! 💰"""
        
        return response

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatgpt' not in st.session_state:
        st.session_state.chatgpt = SimpleFinancialChatGPT()

# Sidebar setup
def setup_sidebar():
    st.sidebar.header("🎯 Your Profile")
    
    capital = st.sidebar.number_input(
        "💰 Investment Capital (₹)",
        min_value=1000,
        max_value=10000000,
        value=50000,
        step=1000
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "📊 Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    st.session_state.user_context = {
        'capital': capital,
        'risk_tolerance': risk_tolerance.lower()
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Capital:** ₹{capital:,}")
    st.sidebar.markdown(f"**Risk:** {risk_tolerance}")
    
    # Quick buttons
    st.sidebar.markdown("---")
    st.sidebar.markdown("⚡ **Quick Actions**")
    
    if st.sidebar.button("📊 Market Overview"):
        return "What's the current market sentiment?"
    
    if st.sidebar.button("🎯 Stock Picks"):
        return "Give me 3 good stocks to invest"
    
    if st.sidebar.button("🛡️ Risk Check"):
        return "Analyze my risk management"
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    return None

# Main app
def main():
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🤖 AI Financial ChatGPT</h1>
        <p>Your Personal Financial Advisor & Trading Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    sidebar_action = setup_sidebar()
    
    # Display chat messages
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
                <strong>🤖 AI Financial Advisor:</strong><br>
            """, unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle input
    prompt = sidebar_action or st.chat_input("Ask me anything about stocks, trading, or investments...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>{prompt}
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤖 Analyzing market data..."):
            query_analysis = st.session_state.chatgpt.analyze_user_query(prompt)
            context = st.session_state.user_context
            response = st.session_state.chatgpt.generate_response(query_analysis, context)
            
            # Add AI response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show AI response
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 AI Financial Advisor:</strong><br>
            """, unsafe_allow_html=True)
            st.markdown(response)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This AI provides educational information only. Always do your own research before investing.")

if __name__ == "__main__":
    main()
