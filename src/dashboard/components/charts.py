"""
Reusable chart components for the FPL ML System dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st


def create_points_progression_chart(gameweek_data: pd.DataFrame, 
                                  title: str = "Points Progression") -> go.Figure:
    """Create points progression line chart."""
    
    fig = px.line(
        gameweek_data,
        x='gameweek',
        y='cumulative_points',
        title=title,
        labels={'gameweek': 'Gameweek', 'cumulative_points': 'Total Points'},
        line_shape='spline'
    )
    
    fig.update_traces(
        line=dict(color='#38003c', width=3),
        mode='lines+markers',
        marker=dict(size=6, color='#00ff87')
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730',
        showlegend=False,
        height=400
    )
    
    return fig


def create_rank_movement_chart(rank_data: pd.DataFrame,
                             title: str = "Rank Movement") -> go.Figure:
    """Create rank movement chart with inverted Y-axis."""
    
    fig = px.line(
        rank_data,
        x='gameweek',
        y='overall_rank',
        title=title,
        labels={'gameweek': 'Gameweek', 'overall_rank': 'Overall Rank'},
        line_shape='spline'
    )
    
    fig.update_traces(
        line=dict(color='#00ff87', width=3),
        mode='lines+markers',
        marker=dict(size=6, color='#38003c')
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730',
        yaxis=dict(autorange='reversed'),  # Lower rank is better
        showlegend=False,
        height=400
    )
    
    return fig


def create_player_scatter_plot(player_data: pd.DataFrame,
                             x_col: str, y_col: str,
                             color_col: str = 'position',
                             size_col: str = 'total_points',
                             title: str = "Player Analysis") -> go.Figure:
    """Create interactive scatter plot for player analysis."""
    
    fig = px.scatter(
        player_data,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        hover_data=['web_name', 'team', 'now_cost', 'total_points'],
        title=title,
        color_discrete_map={
            'GK': '#FFD700',
            'DEF': '#87CEEB', 
            'MID': '#98FB98',
            'FWD': '#FFA07A'
        }
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730',
        height=500
    )
    
    return fig


def create_team_formation_chart(team_players: List[Dict],
                              formation: str = "3-4-3") -> go.Figure:
    """Create team formation visualization."""
    
    # Formation position mappings
    formations = {
        "3-4-3": {
            'GK': [(0.5, 0.1)],
            'DEF': [(0.25, 0.3), (0.5, 0.3), (0.75, 0.3)],
            'MID': [(0.2, 0.6), (0.4, 0.6), (0.6, 0.6), (0.8, 0.6)],
            'FWD': [(0.3, 0.9), (0.5, 0.9), (0.7, 0.9)]
        },
        "3-5-2": {
            'GK': [(0.5, 0.1)],
            'DEF': [(0.25, 0.3), (0.5, 0.3), (0.75, 0.3)],
            'MID': [(0.15, 0.6), (0.35, 0.6), (0.5, 0.6), (0.65, 0.6), (0.85, 0.6)],
            'FWD': [(0.4, 0.9), (0.6, 0.9)]
        },
        "4-3-3": {
            'GK': [(0.5, 0.1)],
            'DEF': [(0.2, 0.3), (0.4, 0.3), (0.6, 0.3), (0.8, 0.3)],
            'MID': [(0.3, 0.6), (0.5, 0.6), (0.7, 0.6)],
            'FWD': [(0.25, 0.9), (0.5, 0.9), (0.75, 0.9)]
        }
    }
    
    positions = formations.get(formation, formations["3-4-3"])
    colors = {'GK': '#FFD700', 'DEF': '#87CEEB', 'MID': '#98FB98', 'FWD': '#FFA07A'}
    
    fig = go.Figure()
    
    # Group players by position
    players_by_pos = {}
    for player in team_players:
        pos = player.get('position', 'MID')
        if pos not in players_by_pos:
            players_by_pos[pos] = []
        players_by_pos[pos].append(player)
    
    # Add players to formation positions
    for pos, coords in positions.items():
        pos_players = players_by_pos.get(pos, [])
        
        for i, (x, y) in enumerate(coords):
            if i < len(pos_players):
                player = pos_players[i]
                name = player.get('web_name', f'{pos}{i+1}')
                points = player.get('total_points', 0)
                price = player.get('now_cost', 0)
            else:
                name = f'{pos}{i+1}'
                points = 0
                price = 0
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=50,
                    color=colors.get(pos, '#CCCCCC'),
                    line=dict(width=3, color='white'),
                    opacity=0.9
                ),
                text=name,
                textposition='middle center',
                textfont=dict(size=10, color='black', family='Arial Black'),
                hovertext=f'{name}<br>Points: {points}<br>Price: £{price:.1f}M',
                hoverinfo='text',
                showlegend=False
            ))
    
    fig.update_layout(
        title=f'Team Formation ({formation})',
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        plot_bgcolor='rgba(34, 139, 34, 0.3)',  # Football pitch green
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730',
        width=700,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_position_performance_chart(position_data: Dict[str, int]) -> go.Figure:
    """Create position performance breakdown chart."""
    
    positions = list(position_data.keys())
    points = list(position_data.values())
    colors = ['#FFD700', '#87CEEB', '#98FB98', '#FFA07A']
    
    fig = go.Figure(data=[
        go.Bar(
            x=positions,
            y=points,
            marker_color=colors[:len(positions)],
            text=points,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Points by Position',
        xaxis_title='Position',
        yaxis_title='Total Points',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730',
        height=400
    )
    
    return fig


def create_captain_performance_chart(captain_data: pd.DataFrame) -> go.Figure:
    """Create captain performance analysis chart."""
    
    fig = go.Figure()
    
    # Add bars for total points
    fig.add_trace(go.Bar(
        name='Total Points',
        x=captain_data['player'],
        y=captain_data['total_points'],
        marker_color='#38003c',
        yaxis='y',
        offsetgroup=1
    ))
    
    # Add line for average points
    fig.add_trace(go.Scatter(
        name='Average per Game',
        x=captain_data['player'],
        y=captain_data['average_points'],
        marker_color='#00ff87',
        yaxis='y2',
        mode='lines+markers',
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title='Captain Performance Analysis',
        xaxis_title='Player',
        yaxis=dict(
            title='Total Points',
            side='left'
        ),
        yaxis2=dict(
            title='Average Points per Game',
            side='right',
            overlaying='y'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_price_change_heatmap(price_data: pd.DataFrame) -> go.Figure:
    """Create price change prediction heatmap."""
    
    fig = go.Figure(data=go.Heatmap(
        z=price_data['change_probability'].values.reshape(-1, 1),
        x=['Price Change Probability'],
        y=price_data['player'],
        colorscale=[
            [0, '#ff4444'],      # Red for likely falls
            [0.5, '#ffff44'],    # Yellow for stable
            [1, '#44ff44']       # Green for likely rises
        ],
        showscale=True,
        colorbar=dict(title="Probability"),
        text=price_data['change_probability'].apply(lambda x: f"{x:.0%}"),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title='Price Change Predictions',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730'
    )
    
    return fig


def create_transfer_efficiency_chart(transfer_data: pd.DataFrame) -> go.Figure:
    """Create transfer efficiency analysis chart."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Points Gained/Lost', 'Transfer Hits'),
        vertical_spacing=0.1
    )
    
    # Points gained/lost
    colors = ['green' if x >= 0 else 'red' for x in transfer_data['points_change']]
    
    fig.add_trace(
        go.Bar(
            x=transfer_data['gameweek'],
            y=transfer_data['points_change'],
            marker_color=colors,
            name='Points Change',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Transfer hits
    fig.add_trace(
        go.Scatter(
            x=transfer_data['gameweek'],
            y=transfer_data['transfer_cost'],
            mode='lines+markers',
            name='Transfer Cost',
            line=dict(color='orange', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Transfer Efficiency Analysis',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730'
    )
    
    fig.update_xaxes(title_text="Gameweek", row=2, col=1)
    fig.update_yaxes(title_text="Points Change", row=1, col=1)
    fig.update_yaxes(title_text="Cost (pts)", row=2, col=1)
    
    return fig


def create_model_performance_chart(performance_data: pd.DataFrame) -> go.Figure:
    """Create ML model performance tracking chart."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'MSE', 'Correlation', 'Prediction Confidence'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(
            x=performance_data['gameweek'],
            y=performance_data['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#38003c', width=2)
        ),
        row=1, col=1
    )
    
    # MSE
    fig.add_trace(
        go.Scatter(
            x=performance_data['gameweek'],
            y=performance_data['mse'],
            mode='lines+markers',
            name='MSE',
            line=dict(color='#ff6b6b', width=2)
        ),
        row=1, col=2
    )
    
    # Correlation
    fig.add_trace(
        go.Scatter(
            x=performance_data['gameweek'],
            y=performance_data['correlation'],
            mode='lines+markers',
            name='Correlation',
            line=dict(color='#4ecdc4', width=2)
        ),
        row=2, col=1
    )
    
    # Confidence
    fig.add_trace(
        go.Scatter(
            x=performance_data['gameweek'],
            y=performance_data['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#45b7d1', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='ML Model Performance Tracking',
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730',
        showlegend=False
    )
    
    return fig


def create_fixture_difficulty_heatmap(fixture_data: pd.DataFrame) -> go.Figure:
    """Create fixture difficulty heatmap for teams."""
    
    # Pivot data for heatmap
    heatmap_data = fixture_data.pivot(
        index='team',
        columns='gameweek', 
        values='difficulty'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f'GW{gw}' for gw in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale=[
            [0, '#00ff00'],      # Green for easy
            [0.25, '#90EE90'],   # Light green
            [0.5, '#ffff00'],    # Yellow for medium
            [0.75, '#FFA500'],   # Orange
            [1, '#ff0000']       # Red for hard
        ],
        showscale=True,
        colorbar=dict(
            title="Difficulty",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
        ),
        text=heatmap_data.values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Fixture Difficulty Heatmap',
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#262730'
    )
    
    return fig


@st.cache_data
def generate_sample_performance_data(gameweeks: int = 15) -> pd.DataFrame:
    """Generate sample performance data for charts."""
    np.random.seed(42)
    
    data = []
    base_accuracy = 65
    base_mse = 0.003
    base_correlation = 0.7
    base_confidence = 80
    
    for gw in range(1, gameweeks + 1):
        # Add some realistic variation
        accuracy = base_accuracy + np.random.normal(0, 2)
        mse = base_mse + np.random.normal(0, 0.0005)
        correlation = base_correlation + np.random.normal(0, 0.05)
        confidence = base_confidence + np.random.normal(0, 3)
        
        data.append({
            'gameweek': gw,
            'accuracy': max(50, min(85, accuracy)),
            'mse': max(0.001, mse),
            'correlation': max(0.4, min(0.9, correlation)),
            'confidence': max(60, min(95, confidence))
        })
    
    return pd.DataFrame(data)