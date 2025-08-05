"""
Reusable widget components for the FPL ML System dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go


def metric_card(title: str, value: str, delta: Optional[str] = None, 
               delta_color: str = "normal", help_text: Optional[str] = None):
    """Create a styled metric card."""
    
    with st.container():
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #262730;">{title}</h4>
            <h2 style="margin: 0.5rem 0; color: #38003c;">{value}</h2>
            {f'<p style="margin: 0; color: {"green" if delta and delta.startswith("+") else "red" if delta and delta.startswith("-") else "#666"};">{delta}</p>' if delta else ''}
            {f'<small style="color: #666;">{help_text}</small>' if help_text else ''}
        </div>
        """, unsafe_allow_html=True)


def status_indicator(label: str, status: str, color: str = "green"):
    """Create a status indicator widget."""
    
    color_map = {
        "green": "🟢",
        "yellow": "🟡", 
        "red": "🔴",
        "blue": "🔵"
    }
    
    icon = color_map.get(color, "⚪")
    st.write(f"{icon} **{label}:** {status}")


def player_card(player_data: Dict[str, Any], show_prediction: bool = True):
    """Create a player information card."""
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Position color coding
            pos_colors = {
                'GK': '#FFD700',
                'DEF': '#87CEEB', 
                'MID': '#98FB98',
                'FWD': '#FFA07A'
            }
            pos_color = pos_colors.get(player_data.get('position', 'MID'), '#CCCCCC')
            
            st.markdown(f"""
            <div style="background: {pos_color}; padding: 0.5rem; border-radius: 50px; text-align: center; color: black; font-weight: bold;">
                {player_data.get('position', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.write(f"**{player_data.get('web_name', 'Unknown')}**")
            st.write(f"{player_data.get('team', 'N/A')} | £{player_data.get('now_cost', 0):.1f}M")
            
        with col3:
            st.metric("Points", player_data.get('total_points', 0))
            if show_prediction:
                st.metric("Predicted", f"{player_data.get('predicted_points', 0):.1f}")


def transfer_suggestion_card(transfer_data: Dict[str, Any]):
    """Create a transfer suggestion card."""
    
    with st.expander(f"{transfer_data['out']} → {transfer_data['in']} ({transfer_data['expected_points']} pts)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔻 OUT:**")
            st.write(f"• {transfer_data['out']} ({transfer_data['out_team']})")
            st.write(f"• Price: £{transfer_data['out_price']}M")
            st.write(f"• Current form: {transfer_data.get('out_form', 'N/A')}")
        
        with col2:
            st.write("**🔺 IN:**")
            st.write(f"• {transfer_data['in']} ({transfer_data['in_team']})")
            st.write(f"• Price: £{transfer_data['in_price']}M")
            st.write(f"• Current form: {transfer_data.get('in_form', 'N/A')}")
        
        st.markdown("---")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Expected Gain", transfer_data['expected_points'])
        with col_b:
            st.metric("Cost Impact", transfer_data['cost_change'])
        with col_c:
            confidence_color = "🟢" if transfer_data.get('confidence', 'Low') == 'High' else "🟡" if transfer_data.get('confidence', 'Low') == 'Medium' else "🔴"
            st.write(f"**Confidence:** {confidence_color} {transfer_data.get('confidence', 'Low')}")
        
        if transfer_data.get('reasoning'):
            st.write("**💡 Reasoning:**")
            for reason in transfer_data['reasoning']:
                st.write(f"  • {reason}")


def captain_recommendation_card(captain_data: Dict[str, Any], rank: int):
    """Create a captain recommendation card."""
    
    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    risk_color = risk_colors.get(captain_data.get('risk', 'Medium'), "🟡")
    
    with st.container():
        st.markdown(f"""
        <div style="border: 2px solid #38003c; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
            <h4 style="margin: 0; color: #38003c;">#{rank} {captain_data['player']} ({captain_data['team']})</h4>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <div>
                    <p style="margin: 0;"><strong>Expected:</strong> {captain_data['expected']} pts</p>
                    <p style="margin: 0;"><strong>Ownership:</strong> {captain_data['ownership']}</p>
                </div>
                <div>
                    <p style="margin: 0;"><strong>Risk:</strong> {risk_color} {captain_data['risk']}</p>
                    <p style="margin: 0;"><strong>Fixture:</strong> {captain_data.get('fixture', 'N/A')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def fixture_difficulty_widget(team: str, fixtures: List[Dict[str, Any]], weeks: int = 5):
    """Create a fixture difficulty widget for a team."""
    
    st.write(f"**{team} - Next {weeks} fixtures:**")
    
    difficulty_colors = {
        1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"
    }
    
    fixture_display = []
    for fixture in fixtures[:weeks]:
        opponent = fixture.get('opponent', 'TBD')
        difficulty = fixture.get('difficulty', 3)
        home_away = 'H' if fixture.get('is_home', True) else 'A'
        color = difficulty_colors.get(difficulty, "⚪")
        
        fixture_display.append(f"{color} {opponent} ({home_away})")
    
    st.write(" | ".join(fixture_display))


def price_change_widget(price_changes: Dict[str, List[Dict[str, Any]]]):
    """Create a price change predictions widget."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**⬆️ Likely Price Rises:**")
        for player in price_changes.get('rises', []):
            probability = player.get('probability', 0)
            color = "🟢" if probability > 0.8 else "🟡"
            st.write(f"{color} {player['name']} ({probability:.0%})")
    
    with col2:
        st.write("**⬇️ Likely Price Falls:**")
        for player in price_changes.get('falls', []):
            probability = player.get('probability', 0)
            color = "🔴" if probability > 0.8 else "🟠"
            st.write(f"{color} {player['name']} ({probability:.0%})")


def team_news_widget(news_items: List[Dict[str, Any]], max_items: int = 5):
    """Create a team news widget."""
    
    st.write("**📰 Latest Team News:**")
    
    for item in news_items[:max_items]:
        timestamp = item.get('timestamp', datetime.now().strftime('%H:%M'))
        severity = item.get('severity', 'info')
        
        severity_icons = {
            'info': 'ℹ️',
            'warning': '⚠️', 
            'important': '🚨',
            'good': '✅'
        }
        
        icon = severity_icons.get(severity, 'ℹ️')
        
        st.write(f"{icon} **{timestamp}** - {item['message']}")


def ml_confidence_gauge(confidence: float, title: str = "ML Confidence"):
    """Create a confidence gauge widget."""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#38003c"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    
    return fig


def data_freshness_indicator(last_update: datetime, max_age_minutes: int = 60):
    """Create a data freshness indicator."""
    
    if not last_update:
        st.warning("⚠️ No data update timestamp available")
        return
    
    age = datetime.now() - last_update
    age_minutes = age.total_seconds() / 60
    
    if age_minutes <= max_age_minutes:
        st.success(f"✅ Data is fresh (updated {age_minutes:.0f} min ago)")
    elif age_minutes <= max_age_minutes * 2:
        st.warning(f"⚠️ Data is getting stale (updated {age_minutes:.0f} min ago)")
    else:
        st.error(f"🔴 Data is stale (updated {age_minutes:.0f} min ago)")


def performance_summary_widget(performance_data: Dict[str, Any]):
    """Create a performance summary widget."""
    
    st.markdown("### 📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card(
            "Current Rank",
            f"{performance_data.get('current_rank', 0):,}",
            performance_data.get('rank_change', ''),
            help_text="Overall rank this season"
        )
    
    with col2:
        metric_card(
            "Total Points", 
            str(performance_data.get('total_points', 0)),
            f"+{performance_data.get('gw_points', 0)}",
            help_text="Season total (this GW)"
        )
    
    with col3:
        metric_card(
            "Team Value",
            f"£{performance_data.get('team_value', 0):.1f}M",
            f"+£{performance_data.get('value_change', 0):.1f}M",
            help_text="Current squad value"
        )
    
    with col4:
        metric_card(
            "Money ITB",
            f"£{performance_data.get('money_in_bank', 0):.1f}M",
            help_text="Available for transfers"
        )


def optimization_results_widget(optimization_data: Dict[str, Any]):
    """Create optimization results widget."""
    
    if optimization_data.get('status') == 'optimal':
        st.success("✅ Optimization completed successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Points", f"{optimization_data.get('predicted_points', 0):.1f}")
        
        with col2:
            st.metric("Total Cost", f"£{optimization_data.get('total_cost', 0):.1f}M")
        
        with col3:
            st.metric("Remaining Budget", f"£{optimization_data.get('remaining_budget', 0):.1f}M")
        
        if optimization_data.get('improvements'):
            st.write("**💡 Suggested Improvements:**")
            for improvement in optimization_data['improvements']:
                st.write(f"• {improvement}")
    
    else:
        st.error(f"❌ Optimization failed: {optimization_data.get('error', 'Unknown error')}")


def ai_chat_widget():
    """Create an AI chat interface widget."""
    
    st.markdown("### 💬 Ask the AI Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_input = st.text_input("Ask anything about FPL strategy:", key="ai_chat_input")
    
    if st.button("🤖 Send") and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate AI response (placeholder)
        ai_response = generate_ai_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    
    # Display chat history
    for message in st.session_state.chat_history[-6:]:  # Show last 6 messages
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**🤖 AI:** {message['content']}")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


def generate_ai_response(user_input: str) -> str:
    """Generate AI response based on user input."""
    
    input_lower = user_input.lower()
    
    if "captain" in input_lower:
        return ("Based on current form and fixtures, I recommend considering Haaland as captain. "
                "He has excellent expected points (11.2) and a favorable fixture against Sheffield United. "
                "While his ownership is high, his consistency makes him a reliable choice.")
    
    elif "transfer" in input_lower:
        return ("For transfers this week, consider players with improving fixtures. "
                "Wilson to Haaland could be valuable if you have the funds, as Wilson faces tough fixtures ahead. "
                "Always consider the opportunity cost and your team's weaknesses.")
    
    elif "differential" in input_lower:
        return ("Good differential options include players with low ownership but strong underlying stats. "
                "Look for players around 5-15% ownership with favorable fixtures coming up. "
                "Consider their historical performance and team's attacking potential.")
    
    elif "wildcard" in input_lower:
        return ("Wildcard timing is crucial. Consider using it before double gameweeks or when you need "
                "significant team changes. Gameweek 19 often presents good opportunities with fixture swings. "
                "Ensure you have a clear strategy for 8+ transfers.")
    
    else:
        return ("I'm here to help with your FPL strategy! I can provide insights on transfers, "
                "captain choices, fixture analysis, and team optimization. Feel free to ask about "
                "specific players, tactics, or strategic decisions.")


def loading_spinner(text: str = "Loading..."):
    """Create a loading spinner with custom text."""
    
    with st.spinner(text):
        # This would be used in conjunction with actual loading operations
        pass


def error_message(message: str, details: Optional[str] = None):
    """Display a formatted error message."""
    
    st.error(f"❌ {message}")
    
    if details:
        with st.expander("Error Details"):
            st.code(details)


def success_banner(message: str, details: Optional[str] = None):
    """Display a success banner."""
    
    st.success(f"✅ {message}")
    
    if details:
        st.info(details)


def warning_banner(message: str, action_required: bool = False):
    """Display a warning banner."""
    
    if action_required:
        st.warning(f"⚠️ **Action Required:** {message}")
    else:
        st.warning(f"⚠️ {message}")


@st.cache_data
def generate_sample_fixtures(team: str, weeks: int = 8) -> List[Dict[str, Any]]:
    """Generate sample fixture data for testing."""
    
    teams = ['ARS', 'LIV', 'MCI', 'CHE', 'MUN', 'TOT', 'NEW', 'BHA', 'AVL', 'WHU', 
             'WOL', 'EVE', 'FUL', 'CRY', 'BUR', 'SHU', 'LUT', 'NOT', 'BRE', 'AFC']
    
    fixtures = []
    for i in range(weeks):
        opponent = np.random.choice([t for t in teams if t != team])
        difficulty = np.random.randint(2, 5)
        is_home = np.random.choice([True, False])
        
        fixtures.append({
            'gameweek': i + 1,
            'opponent': opponent,
            'difficulty': difficulty,
            'is_home': is_home
        })
    
    return fixtures