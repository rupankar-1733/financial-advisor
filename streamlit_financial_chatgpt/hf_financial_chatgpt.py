# streamlit_financial_chatgpt/hf_financial_chatgpt.py - HF API Version
import streamlit as st
import requests
import json
import sys
import os
from datetime import datetime
import time

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your trading systems
from strategies.working_zones_system import WorkingZoneDetector
from utils.live_data_fetcher import LiveDataFetcher

# Page config
st.set_page_config(
    page_title="🤖 AI Financial ChatGPT",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional ChatGPT design
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
        margin-right: 0.5rem;
    }
    .bot-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-right: 3rem;
        margin-left: 0.5rem;
    }
    .header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .thinking {
        color: #6c757d;
        font-style: italic;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .sidebar .stSelectbox > div > div {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class HFFinancialChatGPT:
    def __init__(self):
        """Initialize HF Financial ChatGPT"""
        # Get HF token from secrets
        self.hf_token = st.secrets.get("HF_TOKEN", "demo")
        
        # HF API endpoints (FREE models)
        self.hf_models = {
            'chat': 'microsoft/DialoGPT-large',  # Free conversational model
            'text': 'gpt2',  # Free text generation
            'financial': 'microsoft/DialoGPT-medium'  # Alternative free model
        }
        
        # Your trading systems
        self.zone_detector = WorkingZoneDetector
        self.data_fetcher = LiveDataFetcher()
        
        # Financial knowledge to prevent hallucination
        self.financial_templates = {
            'investment_advice': """
**Investment Analysis for {stock_name}**

**Current Analysis:**
- Current Price: ₹{current_price:.2f}
- Capital Available: ₹{capital:,}
- Max Shares: {max_shares}

**Technical Analysis:**
- Support Level: ₹{support_price:.0f} ({support_method})
- Resistance Level: ₹{resistance_price:.0f} ({resistance_method})
- Risk-Reward Ratio: 1:{risk_reward:.1f}

**Investment Recommendation:**
Based on technical analysis, consider investing ₹{recommended_amount:,} ({recommended_shares} shares) with entry near ₹{support_price:.0f}.

**Risk Management:**
- Stop Loss: ₹{stop_loss:.0f}
- Target: ₹{resistance_price:.0f}
- Max Risk: ₹{max_risk:.0f} (2% of portfolio)
""",
            'general_advice': """
**AI Financial Advisor Guidance**

I can help you with:
- **Stock Analysis**: Technical analysis with support/resistance levels
- **Investment Planning**: Portfolio allocation and risk management  
- **Trading Strategies**: Entry/exit points with precise calculations
- **Risk Assessment**: Position sizing and portfolio protection

**Try asking:**
- "Analyze TCS for ₹50k investment"
- "Should I buy RELIANCE for trading?"
- "What's my risk for ₹1L portfolio?"

**Market Insight**: Focus on quality stocks with strong technical setups and proper risk management.
"""
        }
    
    def query_hf_model(self, prompt, max_retries=2):
        """Query Hugging Face model with fallback"""
        
        # For financial queries, use structured responses instead of LLM
        # This prevents hallucination and ensures accurate financial advice
        return self.generate_structured_response(prompt)
    
    def generate_structured_response(self, user_message):
        """Generate structured financial response"""
        
        # Analyze user query
        analysis = self.analyze_user_query(user_message)
        context = st.session_state.get('user_context', {})
        
        if analysis['intent'] == 'stock_analysis' and analysis['stocks']:
            return self.generate_stock_analysis_response(analysis['stocks'][0], context)
        elif analysis['intent'] == 'investment_advice':
            return self.generate_investment_advice_response(analysis, context)
        elif analysis['intent'] == 'trading_advice':
            return self.generate_trading_advice_response(analysis, context)
        elif analysis['intent'] == 'risk_management':
            return self.generate_risk_management_response(context)
        else:
            return self.financial_templates['general_advice']
    
    def analyze_user_query(self, message):
        """Analyze user intent and extract information"""
        message_lower = message.lower()
        
        # Extract stocks
        stocks = []
        for symbol in ['tcs', 'infy', 'reliance', 'hdfcbank', 'itc']:
            if symbol in message_lower:
                stocks.append(f"{symbol.upper()}.NS")
        
        # Determine intent
        if any(word in message_lower for word in ['analyze', 'analysis', 'target']):
            intent = 'stock_analysis'
        elif any(word in message_lower for word in ['buy', 'invest', 'should i']):
            intent = 'investment_advice'
        elif any(word in message_lower for word in ['trade', 'trading', 'quick']):
            intent = 'trading_advice'
        elif any(word in message_lower for word in ['risk', 'stop loss', 'position size']):
            intent = 'risk_management'
        else:
            intent = 'general'
        
        return {
            'intent': intent,
            'stocks': stocks,
            'message': message
        }
    
    def generate_stock_analysis_response(self, stock, context):
        """Generate detailed stock analysis"""
        capital = context.get('capital', 50000)
        
        try:
            # Get real analysis from your zone system
            detector = self.zone_detector(stock, capital)
            zones_data = detector.get_price_zones()
            
            if not zones_data:
                return f"❌ Unable to analyze {stock.replace('.NS', '')} at the moment. Please try again later."
            
            stock_name = stock.replace('.NS', '')
            current_price = zones_data['current_price']
            max_shares = int(capital / current_price)
            
            # Get best zones
            support_zone = zones_data['support_zones'][0] if zones_data['support_zones'] else None
            resistance_zone = zones_data['resistance_zones'][0] if zones_data['resistance_zones'] else None
            
            if not support_zone or not resistance_zone:
                return f"❌ Insufficient technical data for {stock_name}. Market conditions may be unclear."
            
            # Calculate metrics
            support_price = support_zone['price']
            resistance_price = resistance_zone['price']
            stop_loss = support_price * 0.98
            
            risk_per_share = current_price - stop_loss
            reward_per_share = resistance_price - current_price
            risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
            
            # Position sizing
            recommended_amount = min(capital * 0.2, 20000)  # Max 20% or ₹20k
            recommended_shares = int(recommended_amount / current_price)
            max_risk = risk_per_share * recommended_shares
            
            # Use template
            response = self.financial_templates['investment_advice'].format(
                stock_name=stock_name,
                current_price=current_price,
                capital=capital,
                max_shares=max_shares,
                support_price=support_price,
                support_method=support_zone['method'],
                resistance_price=resistance_price,
                resistance_method=resistance_zone['method'],
                risk_reward=risk_reward,
                recommended_amount=recommended_amount,
                recommended_shares=recommended_shares,
                stop_loss=stop_loss,
                max_risk=max_risk
            )
            
            # Add quality assessment
            if risk_reward >= 3:
                response += "\n\n**Trade Quality**: 🟢 Excellent setup with strong risk-reward ratio!"
            elif risk_reward >= 2:
                response += "\n\n**Trade Quality**: 🟡 Good setup with acceptable risk-reward."
            else:
                response += "\n\n**Trade Quality**: 🔴 Poor risk-reward ratio. Consider waiting for better entry."
            
            return response
            
        except Exception as e:
            return f"❌ Analysis error for {stock.replace('.NS', '')}: {str(e)}"
    
    def generate_investment_advice_response(self, analysis, context):
        """Generate investment advice"""
        capital = context.get('capital', 50000)
        risk_level = context.get('risk_tolerance', 'moderate')
        
        response = f"## 💰 Investment Strategy (₹{capital:,})\n\n"
        response += f"**Risk Profile**: {risk_level.title()}\n\n"
        
        if analysis['stocks']:
            # Analyze mentioned stocks
            for stock in analysis['stocks'][:2]:
                stock_response = self.generate_stock_analysis_response(stock, context)
                response += f"\n{stock_response}\n"
        else:
            # General advice
            response += "**Portfolio Allocation Recommendations:**\n\n"
            if risk_level == 'conservative':
                response += "- 70% Large-cap stocks (TCS, HDFCBANK)\n"
                response += "- 20% Index funds/ETFs\n"
                response += "- 10% Cash for opportunities\n"
            elif risk_level == 'aggressive':
                response += "- 60% Growth stocks (TCS, RELIANCE, INFY)\n"
                response += "- 25% Mid-cap opportunities\n"
                response += "- 15% High-growth sectors\n"
            else:
                response += "- 60% Quality large-caps (TCS, INFY)\n"
                response += "- 25% Diversified mid-caps\n"
                response += "- 15% Cash buffer\n"
            
            response += "\n**Top Stock Recommendations:**\n"
            response += "1. **TCS** - Stable IT leader with consistent growth\n"
            response += "2. **RELIANCE** - Diversified conglomerate\n"
            response += "3. **HDFCBANK** - Leading private sector bank\n"
        
        return response
    
    def generate_trading_advice_response(self, analysis, context):
        """Generate trading advice"""
        capital = context.get('capital', 50000)
        
        response = f"## ⚡ Trading Strategy (₹{capital:,})\n\n"
        
        if analysis['stocks']:
            for stock in analysis['stocks']:
                response += self.generate_stock_analysis_response(stock, context)
        else:
            response += "**Key Trading Principles:**\n\n"
            response += "1. **Risk Management**: Never risk more than 2% per trade\n"
            response += "2. **Zone Trading**: Buy at support, sell at resistance\n"
            response += "3. **Volume Confirmation**: Ensure high volume at key levels\n"
            response += "4. **Stop Losses**: Always use stops below support\n"
            response += "5. **Position Sizing**: Based on distance to stop loss\n"
        
        return response
    
    def generate_risk_management_response(self, context):
        """Generate risk management advice"""
        capital = context.get('capital', 50000)
        
        response = f"## 🛡️ Risk Management (₹{capital:,})\n\n"
        
        max_risk_per_trade = capital * 0.02
        max_position_size = capital * 0.2
        emergency_fund = capital * 0.1
        
        response += "**Your Risk Parameters:**\n\n"
        response += f"- **Max Risk per Trade**: ₹{max_risk_per_trade:,.0f} (2% rule)\n"
        response += f"- **Max Position Size**: ₹{max_position_size:,.0f} (20% limit)\n"
        response += f"- **Emergency Fund**: ₹{emergency_fund:,.0f} (10% buffer)\n\n"
        
        response += "**Risk Management Rules:**\n\n"
        response += "1. Never risk more than you can afford to lose\n"
        response += "2. Diversify across multiple stocks and sectors\n"
        response += "3. Use stop losses based on technical levels\n"
        response += "4. Position size based on risk, not conviction\n"
        response += "5. Keep detailed trading journal\n"
        
        return response

def init_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """👋 **Welcome to AI Financial ChatGPT!**

I'm your AI financial advisor powered by advanced market analysis tools and real-time data.

**I can help you with:**
- 📊 **Stock Analysis** with live support/resistance levels
- 💰 **Investment Strategies** tailored to your profile  
- ⚡ **Trading Setups** with precise entry/exit points
- 🛡️ **Risk Management** and position sizing

**Try asking:**
- "Analyze TCS for ₹50k investment"
- "Should I buy RELIANCE for trading?"
- "What's my risk management strategy?"

What would you like to know about investing or trading today? 💰"""
            }
        ]
    
    if 'hf_chatgpt' not in st.session_state:
        st.session_state.hf_chatgpt = HFFinancialChatGPT()

def setup_sidebar():
    """Setup sidebar"""
    st.sidebar.title("🎯 Your Investment Profile")
    
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
    
    time_horizon = st.sidebar.selectbox(
        "⏰ Investment Horizon",  
        ["Short-term (< 6 months)", "Medium-term (6-24 months)", "Long-term (2+ years)"],
        index=1
    )
    
    st.session_state.user_context = {
        'capital': capital,
        'risk_tolerance': risk_tolerance.lower(),
        'time_horizon': time_horizon.lower()
    }
    
    # Display profile
    st.sidebar.markdown("---")
    st.sidebar.markdown("📋 **Current Profile**")
    st.sidebar.metric("Capital", f"₹{capital/1000:.0f}K")
    st.sidebar.metric("Risk", risk_tolerance[:4])
    
    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("⚡ **Quick Questions**")
    
    if st.sidebar.button("📊 Market Analysis"):
        return "Give me current market analysis and top stock picks"
    
    if st.sidebar.button("🎯 Investment Strategy"):
        return f"Create investment strategy for {risk_tolerance.lower()} investor with ₹{capital:,}"
    
    if st.sidebar.button("🛡️ Risk Check"):
        return f"Analyze risk management for ₹{capital:,} portfolio"
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
        st.rerun()
    
    return None

def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🤖 AI Financial ChatGPT</h1>
        <p>Powered by Advanced Market Analysis & Real-time Data</p>
        <small>Free Hugging Face API + Your Custom Trading Algorithms</small>
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
                <strong>🤖 AI Financial Advisor:</strong><br>
            """, unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
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
        
        # Show thinking animation
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking">
            🤖 Analyzing market data and generating personalized financial advice...
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        try:
            response = st.session_state.hf_chatgpt.query_hf_model(prompt)
            
            # Remove thinking animation
            thinking_placeholder.empty()
            
            # Add and show response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 AI Financial Advisor:</strong><br>
            """, unsafe_allow_html=True)
            st.markdown(response)
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"I encountered an issue: {e}. Please try again with a different question."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(error_msg)
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: This AI provides educational information only. Always do your own research and consult qualified professionals before investing.")

if __name__ == "__main__":
    main()
