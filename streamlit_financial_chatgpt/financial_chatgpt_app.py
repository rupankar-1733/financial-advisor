# streamlit_financial_chatgpt/financial_chatgpt_app.py - Your Financial ChatGPT
import streamlit as st
import requests
import json
import sys
import os
from datetime import datetime
import pandas as pd
import time

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your trading systems
from strategies.working_zones_system import WorkingZoneDetector
from data_sources.comprehensive_intelligence_system import UltimateMarketIntelligence
from utils.live_data_fetcher import LiveDataFetcher

# Configure page
st.set_page_config(
    page_title="💰 AI Financial ChatGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stChat > div {
        background-color: transparent;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 3rem;
        flex-direction: row-reverse;
    }
    .bot-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-right: 3rem;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    .user-avatar {
        background: rgba(255,255,255,0.2);
    }
    .bot-avatar {
        background: #28a745;
        color: white;
    }
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #e9ecef;
        margin-top: 2rem;
    }
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-title {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .thinking-animation {
        color: #6c757d;
        font-style: italic;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

class FinancialChatGPT:
    def __init__(self):
        """Initialize your Financial ChatGPT"""
        # Hugging Face API settings (FREE)
        self.hf_token = st.secrets.get("HF_TOKEN", "hf_demo")  # You'll get this free
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        
        # Your trading systems
        self.zone_detector = WorkingZoneDetector
        self.market_intel = UltimateMarketIntelligence()
        self.data_fetcher = LiveDataFetcher()
        
        # Financial knowledge base to prevent hallucination
        self.financial_rules = {
            'risk_management': [
                "Never risk more than 2% of portfolio per trade",
                "Always use stop losses based on support levels",
                "Position size = (Portfolio Risk ÷ Distance to Stop Loss)",
                "Maximum 20% allocation per stock",
                "Keep 10-20% cash for opportunities"
            ],
            'technical_analysis': [
                "Buy near support zones with volume confirmation",
                "Sell near resistance or take partial profits",
                "Wait for price to reach identified zones",
                "Use multiple timeframe analysis",
                "Follow the trend direction"
            ],
            'indian_stocks': {
                'TCS': 'IT services giant, stable dividend, US revenue exposure',
                'RELIANCE': 'Oil to digital conglomerate, high beta stock',
                'INFY': 'Global IT services, strong fundamentals',
                'HDFCBANK': 'Leading private bank, interest rate sensitive',
                'ITC': 'FMCG + tobacco, regulatory risks'
            }
        }
    
    def generate_response(self, user_message, context):
        """Generate AI response using financial knowledge + real data"""
        
        # Analyze user query
        analysis = self.analyze_query(user_message, context)
        
        # Get live data if needed
        live_data = self.get_live_data(analysis, context)
        
        # Generate structured response
        response = self.create_structured_response(analysis, live_data, context)
        
        return response
    
    def analyze_query(self, message, context):
        """Analyze user query and extract intent"""
        message_lower = message.lower()
        
        # Extract stocks mentioned
        stocks = []
        for symbol in ['tcs', 'infy', 'reliance', 'hdfcbank', 'itc']:
            if symbol in message_lower:
                stocks.append(f"{symbol.upper()}.NS")
        
        # Determine query type
        if any(word in message_lower for word in ['buy', 'invest', 'should i']):
            query_type = 'investment_advice'
        elif any(word in message_lower for word in ['trade', 'trading', 'quick']):
            query_type = 'trading_advice'
        elif any(word in message_lower for word in ['analyze', 'target', 'support']):
            query_type = 'technical_analysis'
        elif any(word in message_lower for word in ['risk', 'stop loss']):
            query_type = 'risk_management'
        elif any(word in message_lower for word in ['market', 'sentiment']):
            query_type = 'market_overview'
        else:
            query_type = 'general'
        
        return {
            'type': query_type,
            'stocks': stocks,
            'original_message': message
        }
    
    def get_live_data(self, analysis, context):
        """Get live market data for mentioned stocks"""
        live_data = {}
        
        if analysis['stocks'] and len(analysis['stocks']) <= 2:
            for stock in analysis['stocks']:
                try:
                    capital = context.get('capital', 50000)
                    detector = self.zone_detector(stock, capital)
                    zones_data = detector.get_price_zones()
                    
                    if zones_data:
                        live_data[stock] = {
                            'current_price': zones_data['current_price'],
                            'support_zones': zones_data['support_zones'][:3],
                            'resistance_zones': zones_data['resistance_zones'][:3],
                            'max_shares': int(capital / zones_data['current_price'])
                        }
                except Exception as e:
                    live_data[stock] = f"Analysis unavailable: {e}"
        
        return live_data
    
    def create_structured_response(self, analysis, live_data, context):
        """Create structured, non-hallucinated response"""
        
        query_type = analysis['type']
        stocks = analysis['stocks']
        message = analysis['original_message']
        capital = context.get('capital', 50000)
        risk_level = context.get('risk_tolerance', 'moderate')
        
        # Base response structure
        response = f"## 🤖 AI Financial Analysis\n\n"
        
        if query_type == 'investment_advice':
            response += self.generate_investment_response(stocks, live_data, capital, risk_level)
        elif query_type == 'trading_advice':
            response += self.generate_trading_response(stocks, live_data, capital)
        elif query_type == 'technical_analysis':
            response += self.generate_technical_response(stocks, live_data)
        elif query_type == 'risk_management':
            response += self.generate_risk_response(capital, risk_level)
        elif query_type == 'market_overview':
            response += self.generate_market_response()
        else:
            response += self.generate_general_response()
        
        # Add personalized footer
        response += f"\n\n---\n**Your Profile:** ₹{capital:,} capital • {risk_level.title()} risk tolerance"
        
        return response
    
    def generate_investment_response(self, stocks, live_data, capital, risk_level):
        """Generate investment-specific response"""
        response = f"### 💰 Investment Strategy (₹{capital:,})\n\n"
        
        if stocks and live_data:
            for stock in stocks:
                if stock in live_data and isinstance(live_data[stock], dict):
                    data = live_data[stock]
                    stock_name = stock.replace('.NS', '')
                    
                    response += f"**📊 {stock_name} Analysis:**\n"
                    response += f"- Current Price: ₹{data['current_price']:.2f}\n"
                    response += f"- Max Shares: {data['max_shares']}\n"
                    
                    # Position sizing based on risk level
                    if risk_level == 'conservative':
                        allocation_pct = 15
                    elif risk_level == 'aggressive':
                        allocation_pct = 25
                    else:
                        allocation_pct = 20
                    
                    allocation = capital * (allocation_pct / 100)
                    shares = int(allocation / data['current_price'])
                    
                    response += f"- Recommended Investment: ₹{allocation:,.0f} ({shares} shares)\n"
                    
                    if data['support_zones']:
                        best_support = data['support_zones'][0]['price']
                        response += f"- Entry Zone: ₹{best_support:.0f} (support level)\n"
                    
                    if data['resistance_zones']:
                        target = data['resistance_zones'][0]['price']
                        upside = ((target - data['current_price']) / data['current_price']) * 100
                        response += f"- Target: ₹{target:.0f} ({upside:.1f}% upside)\n"
                    
                    response += "\n"
        else:
            # General advice based on risk profile
            response += f"**For {risk_level.title()} investors with ₹{capital:,}:**\n\n"
            response += "**Allocation Strategy:**\n"
            response += f"- Large-cap stocks: {70 if risk_level == 'conservative' else 50}%\n"
            response += f"- Mid-cap stocks: {20 if risk_level == 'conservative' else 35}%\n"
            response += f"- Cash/Emergency fund: {10 if risk_level == 'conservative' else 15}%\n\n"
            
            response += "**Recommended stocks for current market:**\n"
            response += "- **TCS**: Stable IT giant, good for conservative investors\n"
            response += "- **RELIANCE**: Diversified conglomerate, moderate risk\n"
            response += "- **HDFCBANK**: Leading private bank, interest rate play\n"
        
        return response
    
    def generate_trading_response(self, stocks, live_data, capital):
        """Generate trading-specific response"""
        response = f"### ⚡ Trading Setup (₹{capital:,})\n\n"
        
        if stocks and live_data:
            for stock in stocks:
                if stock in live_data and isinstance(live_data[stock], dict):
                    data = live_data[stock]
                    stock_name = stock.replace('.NS', '')
                    
                    response += f"**📊 {stock_name} Trading Plan:**\n"
                    
                    if data['support_zones'] and data['resistance_zones']:
                        support = data['support_zones'][0]['price']
                        resistance = data['resistance_zones'][0]['price']
                        current = data['current_price']
                        
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
                        
                        response += f"- **Entry:** ₹{entry:.0f} (at support)\n"
                        response += f"- **Target:** ₹{target:.0f}\n"
                        response += f"- **Stop Loss:** ₹{stop_loss:.0f}\n"
                        response += f"- **Risk:Reward:** 1:{rr_ratio:.1f}\n"
                        response += f"- **Position Size:** {position_size} shares\n"
                        response += f"- **Investment:** ₹{position_size * entry:,.0f}\n"
                        
                        # Trade quality
                        if rr_ratio >= 3:
                            response += f"- **Quality:** 🟢 Excellent setup\n"
                        elif rr_ratio >= 2:
                            response += f"- **Quality:** 🟡 Good setup\n"
                        else:
                            response += f"- **Quality:** 🔴 Poor risk/reward\n"
                    
                    response += "\n"
        else:
            response += "**Key Trading Rules:**\n\n"
            for rule in self.financial_rules['technical_analysis']:
                response += f"- {rule}\n"
        
        return response
    
    def generate_technical_response(self, stocks, live_data):
        """Generate technical analysis response"""
        response = "### 📊 Technical Analysis\n\n"
        
        if stocks and live_data:
            for stock in stocks:
                if stock in live_data and isinstance(live_data[stock], dict):
                    data = live_data[stock]
                    stock_name = stock.replace('.NS', '')
                    
                    response += f"**{stock_name} Technical View:**\n"
                    response += f"- Current Price: ₹{data['current_price']:.2f}\n"
                    response += f"- Support Levels: {len(data['support_zones'])} identified\n"
                    response += f"- Resistance Levels: {len(data['resistance_zones'])} identified\n"
                    
                    if data['support_zones']:
                        support = data['support_zones'][0]
                        response += f"- Key Support: ₹{support['price']:.0f} ({support['method']})\n"
                    
                    if data['resistance_zones']:
                        resistance = data['resistance_zones'][0]
                        response += f"- Key Resistance: ₹{resistance['price']:.0f} ({resistance['method']})\n"
                    
                    response += "\n"
        
        return response
    
    def generate_risk_response(self, capital, risk_level):
        """Generate risk management response"""
        response = f"### 🛡️ Risk Management (₹{capital:,})\n\n"
        
        max_risk_per_trade = capital * 0.02
        max_position = capital * 0.2
        emergency_fund = capital * 0.1
        
        response += f"**Your Risk Parameters:**\n"
        response += f"- Max risk per trade: ₹{max_risk_per_trade:,.0f} (2%)\n"
        response += f"- Max position size: ₹{max_position:,.0f} (20%)\n"
        response += f"- Emergency fund: ₹{emergency_fund:,.0f} (10%)\n\n"
        
        response += "**Risk Management Rules:**\n"
        for rule in self.financial_rules['risk_management']:
            response += f"- {rule}\n"
        
        return response
    
    def generate_market_response(self):
        """Generate market overview response"""
        response = "### 🌍 Market Overview\n\n"
        
        try:
            status = self.data_fetcher.get_current_market_status()
            response += f"**Market Status:** {'🟢 Open' if status['is_open'] else '🔴 Closed'}\n"
            response += f"**Time:** {status['current_time']}\n\n"
        except:
            pass
        
        response += "**Current Market Themes:**\n"
        response += "- Quality over quantity in stock selection\n"
        response += "- Focus on earnings growth and fundamentals\n"
        response += "- Monitor global cues and FII flows\n"
        response += "- Volatility expected around major events\n"
        
        return response
    
    def generate_general_response(self):
        """Generate general help response"""
        response = "### 💡 How I Can Help You\n\n"
        response += "I'm your AI Financial Advisor. I can help with:\n\n"
        response += "**📊 Stock Analysis**\n- Technical analysis with support/resistance\n- Entry and exit recommendations\n\n"
        response += "**💰 Investment Planning**\n- Portfolio allocation strategies\n- Risk-based recommendations\n\n"
        response += "**⚡ Trading Setups**\n- Entry/exit points with risk management\n- Position sizing calculations\n\n"
        response += "**🛡️ Risk Management**\n- Position sizing rules\n- Stop loss strategies\n\n"
        response += "**Try asking:**\n"
        response += '- "Analyze TCS for ₹50k investment"\n'
        response += '- "Should I buy RELIANCE for trading?"\n'
        response += '- "What\'s my risk with ₹1L portfolio?"\n'
        
        return response

def initialize_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """👋 **Welcome to AI Financial ChatGPT!**

I'm your personal financial advisor powered by advanced market analysis tools. I can help you with:

- 📊 **Stock Analysis** with real-time support/resistance levels
- 💰 **Investment Strategies** tailored to your capital and risk profile  
- ⚡ **Trading Setups** with precise entry/exit points
- 🛡️ **Risk Management** with position sizing calculations

What would you like to know about investing or trading today?"""
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    if 'financial_gpt' not in st.session_state:
        st.session_state.financial_gpt = FinancialChatGPT()

def setup_sidebar():
    """Setup sidebar with user profile"""
    st.sidebar.title("🎯 Your Investment Profile")
    
    # User inputs
    capital = st.sidebar.number_input(
        "💰 Investment Capital (₹)",
        min_value=1000,
        max_value=50000000,
        value=50000,
        step=1000,
        help="Total amount you want to invest/trade with"
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "📊 Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="Your comfort level with investment risk"
    )
    
    time_horizon = st.sidebar.selectbox(
        "⏰ Investment Horizon",
        ["Short-term (< 6 months)", "Medium-term (6-24 months)", "Long-term (2+ years)"],
        index=1,
        help="How long you plan to hold investments"
    )
    
    # Store in session state
    st.session_state.user_context = {
        'capital': capital,
        'risk_tolerance': risk_tolerance.lower(),
        'time_horizon': time_horizon.lower()
    }
    
    # Display current profile
    st.sidebar.markdown("---")
    st.sidebar.markdown("📋 **Current Profile**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Capital", f"₹{capital/1000:.0f}K")
    with col2:
        st.metric("Risk", risk_tolerance[:4])
    
    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("⚡ **Quick Actions**")
    
    if st.sidebar.button("📊 Market Overview", key="market_btn"):
        return "What's the current market sentiment and overview?"
    
    if st.sidebar.button("🎯 Stock Recommendations", key="stocks_btn"):
        return f"Give me 3 good stocks for {risk_tolerance.lower()} investor with ₹{capital:,}"
    
    if st.sidebar.button("🛡️ Risk Analysis", key="risk_btn"):
        return f"Analyze my risk management for ₹{capital:,} portfolio"
    
    # Clear chat
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    return None

def main():
    """Main application"""
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="header-title">
        <h1>🤖 AI Financial ChatGPT</h1>
        <p>Your Personal Financial Advisor & Trading Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    sidebar_action = setup_sidebar()
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="avatar user-avatar">👤</div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="avatar bot-avatar">🤖</div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # Handle sidebar actions
    prompt = sidebar_action
    
    # Chat input
    if not prompt:
        prompt = st.chat_input("Ask me anything about stocks, trading, or investments...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message immediately
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="avatar user-avatar">👤</div>
            <div>{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show thinking animation
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="chat-message bot-message">
            <div class="avatar bot-avatar">🤖</div>
            <div class="thinking-animation">🤔 Analyzing market data and generating response...</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        try:
            context = st.session_state.get('user_context', {})
            response = st.session_state.financial_gpt.generate_response(prompt, context)
            
            # Remove thinking animation
            thinking_placeholder.empty()
            
            # Add AI response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show AI response
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="avatar bot-avatar">🤖</div>
                <div>{response}</div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"I encountered an issue: {e}. Please try rephrasing your question."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(error_msg)
        
        # Rerun to update chat
        time.sleep(0.1)  # Brief pause for better UX
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Important Disclaimer:</strong> This AI provides educational information only. 
        All trading and investing involves risk. Please do your own research and consult with qualified financial professionals before making investment decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
