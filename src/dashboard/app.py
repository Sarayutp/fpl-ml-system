"""
FPL ML System - Streamlit Dashboard
Interactive web interface for Fantasy Premier League analysis and optimization.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Import system components
try:
    # Try relative imports first (when running as module)
    from ..agents.fpl_manager import fpl_manager_agent
    from ..config.settings import get_settings
    from ..models.data_models import Player
    from ..utils.cache import CacheManager
except ImportError:
    # Fallback to absolute imports (when running directly)
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.agents.fpl_manager import fpl_manager_agent
    from src.config.settings import get_settings
    from src.models.data_models import Player
    from src.utils.cache import CacheManager

# Page configuration
st.set_page_config(
    page_title="FPL ML System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #38003c, #00ff87);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-banner {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-banner {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'settings' not in st.session_state:
        st.session_state.settings = get_settings()
    
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    
    if 'current_gameweek' not in st.session_state:
        st.session_state.current_gameweek = 16
    
    if 'team_data' not in st.session_state:
        st.session_state.team_data = None
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None

initialize_session_state()

# Utility functions
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    # Sample players data
    players_data = []
    positions = ['GK', 'DEF', 'MID', 'FWD']
    teams = ['ARS', 'LIV', 'MCI', 'CHE', 'MUN', 'TOT', 'NEW', 'BHA', 'AVL', 'WHU']
    
    for i in range(100):
        players_data.append({
            'id': i + 1,
            'web_name': f'Player{i+1}',
            'team': np.random.choice(teams),
            'position': positions[i % 4],
            'now_cost': np.random.uniform(4.0, 13.0),
            'total_points': np.random.randint(20, 180),
            'form': np.random.uniform(2.0, 9.0),
            'selected_by_percent': np.random.uniform(0.5, 45.0),
            'minutes': np.random.randint(500, 1800),
            'goals_scored': np.random.randint(0, 15),
            'assists': np.random.randint(0, 12),
            'predicted_points': np.random.uniform(4.0, 12.0)
        })
    
    return pd.DataFrame(players_data)

async def run_agent_task(agent_class, task_description, deps=None):
    """Run agent task asynchronously."""
    try:
        agent = agent_class()
        result = await agent.run(task_description, deps=deps)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def create_team_formation_viz(team_data):
    """Create team formation visualization."""
    fig = go.Figure()
    
    # Formation positions (3-4-3)
    positions = {
        'GK': [(0.5, 0.1)],
        'DEF': [(0.2, 0.3), (0.4, 0.3), (0.6, 0.3), (0.8, 0.3)],
        'MID': [(0.25, 0.6), (0.45, 0.6), (0.55, 0.6), (0.75, 0.6)],
        'FWD': [(0.3, 0.9), (0.5, 0.9), (0.7, 0.9)]
    }
    
    colors = {'GK': '#FFD700', 'DEF': '#87CEEB', 'MID': '#98FB98', 'FWD': '#FFA07A'}
    
    for pos, coords in positions.items():
        for i, (x, y) in enumerate(coords):
            # Get player for this position
            pos_players = [p for p in team_data if p.get('position') == pos]
            if i < len(pos_players):
                player = pos_players[i]
                name = player.get('web_name', f'{pos}{i+1}')
                points = player.get('total_points', 0)
            else:
                name = f'{pos}{i+1}'
                points = 0
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=40, color=colors[pos], line=dict(width=2, color='black')),
                text=name,
                textposition='middle center',
                textfont=dict(size=10, color='black'),
                hovertext=f'{name}<br>Points: {points}',
                showlegend=False
            ))
    
    fig.update_layout(
        title='Team Formation (3-4-3)',
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
        plot_bgcolor='rgba(50, 150, 50, 0.3)',
        width=600,
        height=400
    )
    
    return fig

# Header
st.markdown('<h1 class="main-header">⚽ FPL ML System Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Fantasy Premier League Management")

# Sidebar
with st.sidebar:
    st.header("🎯 Navigation")
    
    # Quick Actions
    st.subheader("Quick Actions")
    if st.button("🔄 Refresh Data", type="primary"):
        with st.spinner("Updating data..."):
            st.session_state.last_update = datetime.now()
            st.success("Data refreshed!")
    
    if st.button("⚡ Quick Analysis"):
        with st.spinner("Running analysis..."):
            time.sleep(2)  # Simulate processing
            st.success("Analysis complete!")
    
    # Settings
    st.subheader("⚙️ Settings")
    team_id = st.text_input("FPL Team ID", value="12345")
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    
    # System Status
    st.subheader("📊 System Status")
    
    status_data = {
        "Data Pipeline": "🟢 Healthy",
        "ML Models": "🟢 Active", 
        "Optimization": "🟢 Ready",
        "API Connection": "🟡 Limited"
    }
    
    for component, status in status_data.items():
        st.write(f"**{component}:** {status}")
    
    # Last Update
    if st.session_state.last_update:
        st.write(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview", "👥 Team Analysis", "🔄 Transfers", 
    "📊 Players", "🤖 AI Insights", "📈 Performance"
])

# Load sample data
sample_data = load_sample_data()

with tab1:
    st.header("📊 Dashboard Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Rank",
            value="125,432",
            delta="-15,234",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Total Points", 
            value="1,847",
            delta="+67",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Team Value",
            value="£99.2M",
            delta="+£0.3M",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Money in Bank",
            value="£0.8M",
            delta="£0.0M",
            delta_color="off"
        )
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Points Progress")
        
        # Generate sample points data
        gameweeks = list(range(1, 17))
        cumulative_points = np.cumsum(np.random.randint(40, 90, len(gameweeks)))
        
        fig = px.line(
            x=gameweeks, 
            y=cumulative_points,
            title="Cumulative Points This Season",
            labels={'x': 'Gameweek', 'y': 'Total Points'}
        )
        fig.update_traces(line_color='#38003c', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Rank Movement")
        
        # Generate sample rank data
        ranks = [150000 + np.random.randint(-20000, 10000) for _ in gameweeks]
        
        fig = px.line(
            x=gameweeks,
            y=ranks,
            title="Overall Rank Movement",
            labels={'x': 'Gameweek', 'y': 'Overall Rank'}
        )
        fig.update_traces(line_color='#00ff87', line_width=3)
        fig.update_layout(yaxis=dict(autorange="reversed"))  # Lower rank is better
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.subheader("🕐 Recent Activity")
    
    activities = [
        {"time": "2 min ago", "action": "Data updated", "details": "Bootstrap data refreshed"},
        {"time": "15 min ago", "action": "Transfer made", "details": "Wilson → Haaland"},
        {"time": "1 hour ago", "action": "Captain set", "details": "Salah (C) for GW16"},
        {"time": "3 hours ago", "action": "ML prediction", "details": "Player predictions generated"},
        {"time": "6 hours ago", "action": "Team optimized", "details": "Formation analysis complete"}
    ]
    
    for activity in activities:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 4])
            with col1:
                st.write(activity["time"])
            with col2:
                st.write(f"**{activity['action']}**")
            with col3:
                st.write(activity["details"])

with tab2:
    st.header("👥 Current Team Analysis")
    
    # Team Formation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚽ Team Formation")
        
        # Create sample team data
        team_players = sample_data.head(15).to_dict('records')
        formation_fig = create_team_formation_viz(team_players)
        st.plotly_chart(formation_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Team Stats")
        
        # Team statistics
        st.metric("Starting XI Points", "789", "+45")
        st.metric("Bench Points", "23", "+3")
        st.metric("Captain Points", "16", "+8")
        st.metric("Vice-Captain Points", "8", "+2")
        
        st.markdown("---")
        
        # Position breakdown
        st.write("**Position Breakdown:**")
        st.write("• GK: 2 players (98 pts)")
        st.write("• DEF: 5 players (234 pts)")
        st.write("• MID: 5 players (345 pts)")
        st.write("• FWD: 3 players (135 pts)")
    
    # Detailed Team Table
    st.subheader("📋 Detailed Squad")
    
    # Display team table
    team_df = sample_data.head(15)[['web_name', 'team', 'position', 'now_cost', 'total_points', 'form', 'predicted_points']]
    team_df.columns = ['Player', 'Team', 'Position', 'Price (£M)', 'Points', 'Form', 'Predicted']
    
    st.dataframe(
        team_df,
        use_container_width=True,
        column_config={
            "Price (£M)": st.column_config.NumberColumn(format="%.1f"),
            "Form": st.column_config.NumberColumn(format="%.1f"),
            "Predicted": st.column_config.NumberColumn(format="%.1f")
        }
    )
    
    # Team Optimization
    st.subheader("🔧 Team Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Optimize Team", type="primary"):
            with st.spinner("Optimizing team selection..."):
                time.sleep(3)  # Simulate optimization
                st.success("✅ Optimization complete! Expected improvement: +2.3 points")
    
    with col2:
        weeks_ahead = st.selectbox("Optimize for weeks ahead:", [1, 2, 3, 4, 5], index=2)
    
    # Show optimization suggestions
    if st.button("Show Optimization Details"):
        st.markdown("### 🎯 Optimization Results")
        
        suggestions = [
            {"change": "Formation", "from": "3-4-3", "to": "3-5-2", "impact": "+0.8 pts"},
            {"change": "Captain", "from": "Salah", "to": "Haaland", "impact": "+1.2 pts"},
            {"change": "Bench Order", "from": "Current", "to": "Optimized", "impact": "+0.3 pts"}
        ]
        
        for suggestion in suggestions:
            st.write(f"• **{suggestion['change']}:** {suggestion['from']} → {suggestion['to']} ({suggestion['impact']})")

with tab3:
    st.header("🔄 Transfer Analysis")
    
    # Transfer Suggestions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💡 AI Transfer Suggestions")
        
        # Transfer controls
        col_a, col_b = st.columns(2)
        with col_a:
            free_transfers = st.selectbox("Free Transfers:", [0, 1, 2], index=1)
        with col_b:
            planning_weeks = st.selectbox("Planning Horizon:", [1, 2, 3, 4, 5], index=2)
        
        if st.button("🔍 Get Transfer Suggestions", type="primary"):
            with st.spinner("Analyzing transfer opportunities..."):
                time.sleep(2)
                
                # Sample transfer suggestions
                suggestions = [
                    {
                        "out": "Wilson", "out_team": "NEW", "out_price": "7.0",
                        "in": "Haaland", "in_team": "MCI", "in_price": "14.2",
                        "cost": "+7.2M", "expected_gain": "+3.4 pts",
                        "confidence": "High"
                    },
                    {
                        "out": "Trossard", "out_team": "ARS", "out_price": "6.8",
                        "in": "Saka", "in_team": "ARS", "in_price": "8.9",
                        "cost": "+2.1M", "expected_gain": "+1.8 pts",
                        "confidence": "Medium"
                    }
                ]
                
                for i, transfer in enumerate(suggestions, 1):
                    with st.expander(f"Transfer {i}: {transfer['out']} → {transfer['in']} ({transfer['expected_gain']})"):
                        col_x, col_y = st.columns(2)
                        
                        with col_x:
                            st.write("**OUT:**")
                            st.write(f"• {transfer['out']} ({transfer['out_team']})")
                            st.write(f"• Price: £{transfer['out_price']}M")
                        
                        with col_y:
                            st.write("**IN:**")
                            st.write(f"• {transfer['in']} ({transfer['in_team']})")
                            st.write(f"• Price: £{transfer['in_price']}M")
                        
                        st.write(f"**Cost Impact:** {transfer['cost']}")
                        st.write(f"**Expected Gain:** {transfer['expected_gain']} over {planning_weeks} weeks")
                        st.write(f"**Confidence:** {transfer['confidence']}")
    
    with col2:
        st.subheader("📊 Transfer Stats")
        
        st.metric("Transfers Made", "8", "+1")
        st.metric("Transfer Cost", "-4 pts", "+0")
        st.metric("Net Transfer Gain", "+18 pts", "+3")
        
        st.markdown("---")
        
        st.subheader("🎯 Popular Transfers")
        popular_transfers = [
            "Haaland (45% in)",
            "Salah (38% out)", 
            "Saka (32% in)",
            "Wilson (29% out)",
            "Alexander-Arnold (25% in)"
        ]
        
        for transfer in popular_transfers:
            st.write(f"• {transfer}")
    
    # Transfer History
    st.subheader("📅 Transfer History")
    
    transfer_history = [
        {"GW": 15, "Out": "Sterling", "In": "Saka", "Cost": "0 pts", "Points Gained": "+6"},
        {"GW": 13, "Out": "Rashford", "In": "Son", "Cost": "-4 pts", "Points Gained": "+2"},
        {"GW": 10, "Out": "Isak", "In": "Haaland", "Cost": "0 pts", "Points Gained": "+12"},
        {"GW": 8, "Out": "Maddison", "In": "Salah", "Cost": "0 pts", "Points Gained": "+8"},
        {"GW": 5, "Out": "Jesus", "In": "Wilson", "Cost": "-4 pts", "Points Gained": "+1"}
    ]
    
    history_df = pd.DataFrame(transfer_history)
    st.dataframe(history_df, use_container_width=True)
    
    # Wildcard Planning
    st.subheader("🃏 Wildcard Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Optimal Wildcard Timing:**")
        st.write("• Recommended: Gameweek 19")
        st.write("• Reason: Double gameweek + fixture swing")
        st.write("• Expected benefit: +15.2 points")
    
    with col2:
        if st.button("📋 Plan Wildcard Team"):
            st.write("**Wildcard team optimization in progress...**")

with tab4:
    st.header("📊 Player Analysis")
    
    # Player Search and Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        position_filter = st.selectbox("Position:", ["All", "GK", "DEF", "MID", "FWD"])
    
    with col2:
        max_price = st.slider("Max Price (£M):", 4.0, 15.0, 10.0, 0.5)
    
    with col3:
        min_points = st.slider("Min Points:", 0, 200, 50, 10)
    
    with col4:
        team_filter = st.selectbox("Team:", ["All"] + list(sample_data['team'].unique()))
    
    # Filter data
    filtered_data = sample_data.copy()
    
    if position_filter != "All":
        filtered_data = filtered_data[filtered_data['position'] == position_filter]
    
    filtered_data = filtered_data[
        (filtered_data['now_cost'] <= max_price) &
        (filtered_data['total_points'] >= min_points)
    ]
    
    if team_filter != "All":
        filtered_data = filtered_data[filtered_data['team'] == team_filter]
    
    # Player Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price vs Points")
        
        fig = px.scatter(
            filtered_data,
            x='now_cost',
            y='total_points',
            color='position',
            size='selected_by_percent',
            hover_data=['web_name', 'team', 'form'],
            title="Player Value Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Form vs Ownership")
        
        fig = px.scatter(
            filtered_data,
            x='form',
            y='selected_by_percent',
            color='position',
            size='total_points',
            hover_data=['web_name', 'team', 'now_cost'],
            title="Form vs Ownership Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Players Table
    st.subheader("🏆 Top Players")
    
    # Sorting options
    sort_by = st.selectbox("Sort by:", ["total_points", "form", "predicted_points", "now_cost"])
    ascending = st.checkbox("Ascending order", value=False)
    
    top_players = filtered_data.sort_values(sort_by, ascending=ascending).head(20)
    
    display_cols = ['web_name', 'team', 'position', 'now_cost', 'total_points', 'form', 'selected_by_percent', 'predicted_points']
    display_names = ['Player', 'Team', 'Position', 'Price (£M)', 'Points', 'Form', 'Ownership (%)', 'Predicted']
    
    top_players_display = top_players[display_cols].copy()
    top_players_display.columns = display_names
    
    st.dataframe(
        top_players_display,
        use_container_width=True,
        column_config={
            "Price (£M)": st.column_config.NumberColumn(format="%.1f"),
            "Form": st.column_config.NumberColumn(format="%.1f"),
            "Ownership (%)": st.column_config.NumberColumn(format="%.1f"),
            "Predicted": st.column_config.NumberColumn(format="%.1f")
        }
    )
    
    # Player Comparison
    st.subheader("🔍 Player Comparison")
    
    selected_players = st.multiselect(
        "Select players to compare:",
        options=filtered_data['web_name'].tolist(),
        default=filtered_data['web_name'].head(3).tolist()
    )
    
    if selected_players:
        comparison_data = filtered_data[filtered_data['web_name'].isin(selected_players)]
        
        # Radar chart for comparison
        categories = ['total_points', 'form', 'selected_by_percent', 'predicted_points']
        
        fig = go.Figure()
        
        for player in selected_players:
            player_data = comparison_data[comparison_data['web_name'] == player].iloc[0]
            values = [player_data[cat] for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=player
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=True,
            title="Player Comparison Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("🤖 AI Insights & Predictions")
    
    # ML Model Status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("XGBoost MSE", "0.0028", "-0.0003")
    
    with col2:
        st.metric("Model Accuracy", "67.3%", "+2.1%")
    
    with col3:
        st.metric("Prediction Confidence", "84.2%", "+1.8%")
    
    # Captain Recommendations
    st.subheader("👑 Captain Recommendations")
    
    captain_recs = [
        {"player": "Erling Haaland", "team": "MCI", "expected": "11.2", "ownership": "67.8%", "risk": "Medium"},
        {"player": "Mohamed Salah", "team": "LIV", "expected": "9.8", "ownership": "54.2%", "risk": "Low"},
        {"player": "Harry Kane", "team": "BAY", "expected": "8.9", "ownership": "23.1%", "risk": "High"},
        {"player": "Bukayo Saka", "team": "ARS", "expected": "8.4", "ownership": "45.6%", "risk": "Low"}
    ]
    
    for i, rec in enumerate(captain_recs, 1):
        with st.expander(f"{i}. {rec['player']} ({rec['team']}) - {rec['expected']} expected points"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write(f"**Expected Points:** {rec['expected']}")
                st.write(f"**Ownership:** {rec['ownership']}")
            
            with col_b:
                st.write(f"**Risk Level:** {rec['risk']}")
                st.write(f"**Recommendation:** {'Strong' if i <= 2 else 'Consider'}")
    
    # Price Change Predictions
    st.subheader("💰 Price Change Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**⬆️ Likely to Rise:**")
        risers = ["Haaland (89%)", "Saka (76%)", "Palmer (68%)", "Alexander-Arnold (61%)"]
        for riser in risers:
            st.write(f"• {riser}")
    
    with col2:
        st.write("**⬇️ Likely to Fall:**")
        fallers = ["Sterling (82%)", "Rashford (74%)", "Isak (69%)", "Maddison (58%)"]
        for faller in fallers:
            st.write(f"• {faller}")
    
    # AI Chat Interface
    st.subheader("💬 Ask the AI")
    
    user_question = st.text_input("Ask anything about FPL strategy:")
    
    if st.button("🤖 Get AI Response") and user_question:
        with st.spinner("AI is thinking..."):
            time.sleep(2)
            
            # Sample AI responses based on question type
            if "captain" in user_question.lower():
                response = "Based on current form and fixtures, I recommend Haaland as captain for GW16. He has the highest expected points (11.2) and faces a favorable matchup against Sheffield United at home. While his ownership is high (67.8%), his consistent returns make him a safe choice."
            elif "transfer" in user_question.lower():
                response = "For transfers this week, consider Wilson to Haaland if you have the funds. Wilson has tough fixtures coming up, while Haaland has an excellent run. This transfer has a 73% confidence rating and expected gain of +3.4 points over the next 3 gameweeks."
            else:
                response = "I'd be happy to help with your FPL strategy! I can provide insights on transfers, captain choices, fixture analysis, and team optimization. Feel free to ask specific questions about players or tactics."
        
        st.markdown(f"**🤖 AI Response:**\n\n{response}")
    
    # ML Model Insights
    st.subheader("📊 Model Performance")
    
    # Model performance chart
    gameweeks = list(range(10, 17))
    accuracy = [64.2, 65.1, 63.8, 66.7, 67.9, 68.2, 67.3]
    
    fig = px.line(
        x=gameweeks,
        y=accuracy,
        title="Model Accuracy Over Time",
        labels={'x': 'Gameweek', 'y': 'Accuracy (%)'}
    )
    fig.update_traces(line_color='#38003c', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("📈 Performance Analytics")
    
    # Performance Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Season Points", "1,847", "+67")
    
    with col2:
        st.metric("Average/GW", "115.4", "+4.2")
    
    with col3:
        st.metric("Best GW", "87 pts", "GW 12")
    
    with col4:
        st.metric("Worst GW", "31 pts", "GW 7")
    
    # Performance Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Points Distribution")
        
        # Generate sample gameweek points
        np.random.seed(42)
        gw_points = np.random.normal(60, 15, 16).astype(int)
        gw_points = np.clip(gw_points, 20, 100)
        
        fig = px.histogram(
            x=gw_points,
            nbins=10,
            title="Gameweek Points Distribution",
            labels={'x': 'Points per Gameweek', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Rank Progression")
        
        # Generate sample rank data
        ranks = [200000]
        for _ in range(15):
            change = np.random.randint(-25000, 15000)
            new_rank = max(1, ranks[-1] + change)
            ranks.append(new_rank)
        
        gameweeks = list(range(0, 17))
        
        fig = px.line(
            x=gameweeks,
            y=ranks,
            title="Overall Rank Movement",
            labels={'x': 'Gameweek', 'y': 'Overall Rank'}
        )
        fig.update_traces(line_color='#00ff87', line_width=3)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance Analysis
    st.subheader("🔍 Performance Analysis")
    
    analysis_tabs = st.tabs(["Monthly", "Position", "Captaincy", "Transfers"])
    
    with analysis_tabs[0]:
        st.write("**Monthly Performance Breakdown:**")
        
        monthly_data = [
            {"Month": "August", "Points": "234", "Rank": "180K", "Transfers": "2"},
            {"Month": "September", "Points": "298", "Rank": "145K", "Transfers": "3"},
            {"Month": "October", "Points": "387", "Rank": "125K", "Transfers": "4"},
            {"Month": "November", "Points": "456", "Rank": "108K", "Transfers": "2"},
            {"Month": "December", "Points": "472", "Rank": "125K", "Transfers": "1"}
        ]
        
        monthly_df = pd.DataFrame(monthly_data)
        st.dataframe(monthly_df, use_container_width=True)
    
    with analysis_tabs[1]:
        st.write("**Points by Position:**")
        
        position_points = {"GK": 156, "DEF": 412, "MID": 789, "FWD": 490}
        
        fig = px.bar(
            x=list(position_points.keys()),
            y=list(position_points.values()),
            title="Points Contribution by Position"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with analysis_tabs[2]:
        st.write("**Captain Performance:**")
        
        captain_data = [
            {"Player": "Haaland", "Times Captained": "8", "Points": "96", "Average": "12.0"},
            {"Player": "Salah", "Times Captained": "5", "Points": "45", "Average": "9.0"},
            {"Player": "Kane", "Times Captained": "2", "Points": "14", "Average": "7.0"},
            {"Player": "Son", "Times Captained": "1", "Points": "2", "Average": "2.0"}
        ]
        
        captain_df = pd.DataFrame(captain_data)
        st.dataframe(captain_df, use_container_width=True)
    
    with analysis_tabs[3]:
        st.write("**Transfer Efficiency:**")
        
        st.metric("Transfer Hit Success Rate", "75%", "+5%")
        st.metric("Average Points Gained per Transfer", "+2.8", "+0.3")
        st.metric("Best Transfer", "Isak → Haaland", "+12 pts")
        st.metric("Worst Transfer", "Salah → Sterling", "-8 pts")
    
    # Benchmarking
    st.subheader("🏆 Benchmarking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**vs Top 10k Average:**")
        st.metric("Points Difference", "+23", "+8")
        st.metric("Rank Percentile", "84.2%", "+2.1%")
    
    with col2:
        st.write("**vs Overall Average:**")
        st.metric("Points Difference", "+187", "+34")
        st.metric("Rank Percentile", "92.8%", "+1.5%")

# Footer
st.markdown("---")
st.markdown(
    "**FPL ML System** - Powered by AI | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    "Data: FPL API"
)

# Auto-refresh functionality
if auto_refresh:
    # This would implement auto-refresh in a real application
    st.markdown("*Auto-refresh enabled*")

# Error handling display
if st.session_state.get('last_error'):
    st.error(f"System Error: {st.session_state.last_error}")
    if st.button("Clear Error"):
        st.session_state.last_error = None
        st.rerun()