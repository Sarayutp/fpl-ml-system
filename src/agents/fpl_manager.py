"""
Primary FPL Manager Agent - orchestrates all FPL operations and multi-agent delegation.
Follows main_agent_reference/research_agent.py patterns for delegation and tool integration.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd

from pydantic_ai import Agent, RunContext

from ..config.providers import get_llm_model
from ..tools.fpl_api import get_bootstrap_data, get_team_picks, get_manager_history
from ..tools.ml_tools import (
    predict_player_points, 
    optimize_team_selection, 
    optimize_transfers,
    optimize_captain_selection
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert Fantasy Premier League (FPL) manager with access to advanced ML predictions, optimization algorithms, and comprehensive data analysis. Your primary goal is to help users make data-driven FPL decisions to maximize points and improve rankings.

Your capabilities:
1. **Team Analysis**: Analyze current team performance, identify strengths and weaknesses
2. **Transfer Recommendations**: Use ML predictions and optimization to suggest optimal transfers  
3. **Player Research**: Deep dive into player statistics, form, fixtures, and value
4. **Captain Selection**: Optimize captain choices based on predictions and ownership data
5. **Team Optimization**: Build optimal 15-player squads within budget constraints
6. **Multi-Gameweek Planning**: Strategic planning for multiple gameweeks ahead

When providing FPL advice:
- Always consider both data-driven insights AND practical FPL knowledge
- Factor in fixture difficulty, player form, team news, and injury updates
- Balance risk vs reward based on the user's strategy preferences
- Explain your reasoning clearly with supporting data
- Consider ownership percentages for differential vs template decisions
- Account for price changes and transfer timing

Key FPL principles to follow:
- Maximize expected points while managing risk
- Consider both short-term (1-3 GW) and long-term (4+ GW) planning
- Balance premium vs budget players for optimal value
- Time transfers strategically around price changes and deadlines
- Use chips (Wildcard, Free Hit, Bench Boost, Triple Captain) optimally

Always provide actionable recommendations with clear reasoning and data support.
"""


@dataclass
class FPLManagerDependencies:
    """Dependencies for the FPL Manager agent - configuration only, no tool instances."""
    fpl_team_id: int
    database_url: str
    session_id: Optional[str] = None


# Initialize the FPL Manager agent
fpl_manager_agent = Agent(
    get_llm_model(),
    deps_type=FPLManagerDependencies,
    system_prompt=SYSTEM_PROMPT
)


@fpl_manager_agent.tool
async def get_team_analysis(
    ctx: RunContext[FPLManagerDependencies],
    team_id: Optional[int] = None,
    gameweeks_back: int = 5
) -> str:
    """
    Get comprehensive analysis of current FPL team performance.
    
    Args:
        team_id: FPL team ID (uses ctx.deps.fpl_team_id if not provided)
        gameweeks_back: Number of recent gameweeks to analyze
    
    Returns:
        Detailed team analysis as string
    """
    try:
        target_team_id = team_id or ctx.deps.fpl_team_id
        logger.info(f"Analyzing team {target_team_id}")
        
        # Get team history and performance data
        history_data = await get_manager_history(target_team_id)
        
        if not history_data or 'current' not in history_data:
            return f"❌ Could not retrieve data for team {target_team_id}. Please check the team ID is correct."
        
        current_season = history_data['current']
        if not current_season:
            return "❌ No current season data available for this team."
        
        # Analyze recent performance
        recent_games = current_season[-gameweeks_back:] if len(current_season) >= gameweeks_back else current_season
        
        if not recent_games:
            return "❌ No recent gameweek data available."
        
        # Calculate performance metrics
        total_points = sum(gw.get('points', 0) for gw in recent_games)
        avg_points = total_points / len(recent_games)
        
        # Get latest gameweek data
        latest_gw = recent_games[-1]
        current_rank = latest_gw.get('overall_rank', 'Unknown')
        team_value = latest_gw.get('value', 0) / 10  # Convert to millions
        
        # Calculate trends
        if len(recent_games) >= 2:
            rank_change = recent_games[-2].get('overall_rank', 0) - latest_gw.get('overall_rank', 0)
            rank_trend = "📈 Rising" if rank_change > 0 else "📉 Falling" if rank_change < 0 else "➡️ Stable"
        else:
            rank_trend = "➡️ No trend data"
        
        # Analyze transfers and hits
        total_transfers = sum(gw.get('event_transfers', 0) for gw in recent_games)
        total_hits = sum(gw.get('event_transfers_cost', 0) for gw in recent_games)
        
        # Analyze bench performance
        bench_points = sum(gw.get('points_on_bench', 0) for gw in recent_games)
        avg_bench_points = bench_points / len(recent_games)
        
        analysis = f"""
📊 **Team Analysis for ID {target_team_id}**

**Current Status:**
• Overall Rank: {current_rank:,}
• Team Value: £{team_value:.1f}M
• Rank Trend: {rank_trend}

**Recent Performance ({len(recent_games)} GWs):**
• Total Points: {total_points}
• Average Points/GW: {avg_points:.1f}
• Points on Bench: {bench_points} ({avg_bench_points:.1f}/GW)

**Transfer Activity:**
• Total Transfers: {total_transfers}
• Points Deducted: {total_hits}
• Transfer Efficiency: {'Good' if total_hits <= 4 else 'Poor - too many hits'}

**Key Insights:**
• {'Strong recent form' if avg_points >= 60 else 'Below average performance' if avg_points >= 45 else 'Poor recent form'}
• {'Bench optimization needed' if avg_bench_points >= 3 else 'Good bench management'}
• {'Transfer strategy working well' if total_hits <= total_transfers * 4 else 'Consider reducing transfer frequency'}

**Recommendations:**
{_generate_team_recommendations(avg_points, avg_bench_points, total_hits, len(recent_games))}
"""
        
        logger.info(f"Team analysis completed for {target_team_id}")
        return analysis.strip()
        
    except Exception as e:
        logger.error(f"Team analysis failed: {e}")
        return f"❌ Failed to analyze team: {str(e)}"


@fpl_manager_agent.tool  
async def get_transfer_recommendations(
    ctx: RunContext[FPLManagerDependencies],
    team_id: Optional[int] = None,
    weeks_ahead: int = 4,
    free_transfers: int = 1,
    risk_level: str = "balanced"
) -> str:
    """
    Get AI-powered transfer recommendations using ML predictions and optimization.
    
    Args:
        team_id: FPL team ID (uses ctx.deps.fpl_team_id if not provided)
        weeks_ahead: Planning horizon in gameweeks
        free_transfers: Number of free transfers available
        risk_level: "conservative", "balanced", or "aggressive"
    
    Returns:
        Transfer recommendations as string
    """
    try:
        target_team_id = team_id or ctx.deps.fpl_team_id
        logger.info(f"Getting transfer recommendations for team {target_team_id}")
        
        # Get current team data
        bootstrap = await get_bootstrap_data()
        players_df = pd.DataFrame(bootstrap['elements'])
        
        # Get current gameweek
        current_gw = None
        for event in bootstrap['events']:
            if event.get('is_current') or event.get('is_next'):
                current_gw = event['id']
                break
        
        if not current_gw:
            return "❌ Could not determine current gameweek."
        
        # Get current team picks
        try:
            team_picks = await get_team_picks(target_team_id, current_gw)
            current_team = [pick['element'] for pick in team_picks.get('picks', [])]
        except:
            return f"❌ Could not retrieve current team for gameweek {current_gw}. Team may be private or ID incorrect."
        
        if len(current_team) != 15:
            return f"❌ Invalid team data - expected 15 players, got {len(current_team)}."
        
        # Create simple predictions (in production, this would use trained ML models)
        predicted_points = {}
        for _, player in players_df.iterrows():
            # Simple prediction based on form and recent points
            form = float(player.get('form', '0') or '0')
            total_points = player.get('total_points', 0)
            minutes = player.get('minutes', 1)
            
            # Basic prediction formula
            base_prediction = form if form > 0 else total_points / max(minutes / 90, 1) * 90 / 38
            predicted_points[player['id']] = max(0, base_prediction * weeks_ahead)
        
        # Get transfer optimization
        optimization_result = await optimize_transfers(
            current_team=current_team,
            players_df=players_df,
            predicted_points=predicted_points,
            free_transfers=free_transfers,
            weeks_ahead=weeks_ahead
        )
        
        if optimization_result.status != "optimal":
            return f"❌ Transfer optimization failed: {optimization_result.error_message or 'Unknown error'}"
        
        if not optimization_result.recommended_transfers:
            return f"✅ **No beneficial transfers found**\n\nYour current team is well-optimized for the next {weeks_ahead} gameweeks. Consider holding your transfer(s) or look for opportunities closer to the deadline."
        
        # Format recommendations
        recommendations = f"🔄 **Transfer Recommendations** ({risk_level} strategy)\n\n"
        recommendations += f"**Planning horizon:** {weeks_ahead} gameweeks | **Free transfers:** {free_transfers}\n\n"
        
        for i, transfer in enumerate(optimization_result.recommended_transfers[:free_transfers], 1):
            player_out = transfer.player_out
            player_in = transfer.player_in
            
            recommendations += f"**Transfer {i}:**\n"
            recommendations += f"OUT: {player_out.web_name} (£{player_out.price_millions:.1f}M)\n"
            recommendations += f"IN:  {player_in.web_name} (£{player_in.price_millions:.1f}M)\n"
            recommendations += f"Expected gain: +{transfer.expected_points_gain:.1f} points\n"
            recommendations += f"Cost impact: £{transfer.cost_change:+.1f}M\n\n"
            
            if transfer.reasoning:
                recommendations += "**Reasoning:**\n"
                for reason in transfer.reasoning:
                    recommendations += f"• {reason}\n"
                recommendations += "\n"
        
        # Add summary
        total_gain = sum(t.expected_points_gain for t in optimization_result.recommended_transfers[:free_transfers])
        recommendations += f"**Summary:**\n"
        recommendations += f"• Total expected gain: +{total_gain:.1f} points\n"
        recommendations += f"• Transfer cost: {optimization_result.transfer_cost or 0} points\n"
        recommendations += f"• Net benefit: +{total_gain - (optimization_result.transfer_cost or 0):.1f} points\n"
        
        logger.info(f"Transfer recommendations generated for team {target_team_id}")
        return recommendations.strip()
        
    except Exception as e:
        logger.error(f"Transfer recommendations failed: {e}")
        return f"❌ Failed to get transfer recommendations: {str(e)}"


@fpl_manager_agent.tool
async def get_player_analysis(
    ctx: RunContext[FPLManagerDependencies],
    player_name: str,
    gameweeks_ahead: int = 3
) -> str:
    """
    Get detailed analysis of a specific player including ML predictions.
    
    Args:
        player_name: Player name to search for
        gameweeks_ahead: Number of gameweeks to predict
    
    Returns:
        Player analysis as string
    """
    try:
        logger.info(f"Analyzing player: {player_name}")
        
        # Get bootstrap data
        bootstrap = await get_bootstrap_data()
        players_df = pd.DataFrame(bootstrap['elements'])
        teams_df = pd.DataFrame(bootstrap['teams'])
        
        # Find player by name (fuzzy search)
        player_matches = players_df[
            players_df['web_name'].str.contains(player_name, case=False, na=False) |
            players_df['first_name'].str.contains(player_name, case=False, na=False) |
            players_df['second_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if player_matches.empty:
            return f"❌ No player found matching '{player_name}'. Please check the spelling or try a different name."
        
        if len(player_matches) > 1:
            matches_list = []
            for _, player in player_matches.head(5).iterrows():
                matches_list.append(f"• {player['web_name']} ({player['first_name']} {player['second_name']})")
            return f"🔍 Multiple players found matching '{player_name}':\n\n" + "\n".join(matches_list) + "\n\nPlease be more specific."
        
        player = player_matches.iloc[0]
        
        # Get team info
        team_info = teams_df[teams_df['id'] == player['team']].iloc[0]
        
        # Get position name
        position_map = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
        position = position_map.get(player['element_type'], 'Unknown')
        
        # Calculate key metrics
        price = player['now_cost'] / 10
        total_points = player['total_points']
        form = float(player['form']) if player['form'] else 0
        ownership = float(player['selected_by_percent']) if player['selected_by_percent'] else 0
        minutes = player['minutes']
        
        # Performance metrics
        points_per_game = total_points / max(1, minutes / 90) if minutes > 0 else 0
        value_score = total_points / price if price > 0 else 0
        
        # Create simple prediction
        base_prediction = form if form > 0 else points_per_game
        predicted_points = max(0, base_prediction * gameweeks_ahead)
        
        analysis = f"""
⚽ **Player Analysis: {player['web_name']}**

**Basic Info:**
• Full Name: {player['first_name']} {player['second_name']}
• Position: {position}
• Team: {team_info['name']}
• Price: £{price:.1f}M
• Ownership: {ownership:.1f}%

**Season Performance:**
• Total Points: {total_points}
• Points/Game: {points_per_game:.1f}
• Current Form: {form:.1f}
• Minutes Played: {minutes:,}
• Value Score: {value_score:.1f} pts/£M

**Goal Contributions:**
• Goals: {player.get('goals_scored', 0)}
• Assists: {player.get('assists', 0)}
• Clean Sheets: {player.get('clean_sheets', 0)}
• Bonus Points: {player.get('bonus', 0)}

**Advanced Stats:**
• ICT Index: {player.get('ict_index', 'N/A')}
• Influence: {player.get('influence', 'N/A')}
• Creativity: {player.get('creativity', 'N/A')}
• Threat: {player.get('threat', 'N/A')}

**Next {gameweeks_ahead} GW Prediction:**
• Expected Points: {predicted_points:.1f}
• Confidence: {'High' if form >= 6 else 'Medium' if form >= 4 else 'Low'}

**Key Insights:**
{_generate_player_insights(player, form, ownership, value_score)}

**Recommendation:**
{_generate_player_recommendation(player, form, ownership, value_score, price)}
"""
        
        logger.info(f"Player analysis completed for {player['web_name']}")
        return analysis.strip()
        
    except Exception as e:
        logger.error(f"Player analysis failed: {e}")
        return f"❌ Failed to analyze player: {str(e)}"


@fpl_manager_agent.tool
async def get_captain_advice(
    ctx: RunContext[FPLManagerDependencies],
    team_id: Optional[int] = None,
    strategy: str = "balanced"
) -> str:
    """
    Get captain selection advice based on current team and predictions.
    
    Args:
        team_id: FPL team ID (uses ctx.deps.fpl_team_id if not provided)
        strategy: "safe", "balanced", or "differential"
    
    Returns:
        Captain advice as string
    """
    try:
        target_team_id = team_id or ctx.deps.fpl_team_id
        logger.info(f"Getting captain advice for team {target_team_id}")
        
        # Get current team and bootstrap data
        bootstrap = await get_bootstrap_data()
        players_df = pd.DataFrame(bootstrap['elements'])
        
        # Get current gameweek
        current_gw = None
        for event in bootstrap['events']:
            if event.get('is_current') or event.get('is_next'):
                current_gw = event['id']
                break
        
        if not current_gw:
            return "❌ Could not determine current gameweek."
        
        # Get current team picks
        try:
            team_picks = await get_team_picks(target_team_id, current_gw)
            playing_xi = [pick['element'] for pick in team_picks.get('picks', []) if pick.get('multiplier', 0) > 0]
        except:
            return f"❌ Could not retrieve current team for gameweek {current_gw}."
        
        if len(playing_xi) < 11:
            return f"❌ Could not determine playing XI (found {len(playing_xi)} players)."
        
        # Create predictions for playing XI
        predicted_points = {}
        ownership_data = {}
        
        for player_id in playing_xi:
            player = players_df[players_df['id'] == player_id].iloc[0]
            form = float(player['form']) if player['form'] else 0
            ownership = float(player['selected_by_percent']) if player['selected_by_percent'] else 0
            
            predicted_points[player_id] = max(2, form * 1.2)  # Simple prediction
            ownership_data[player_id] = ownership
        
        # Get captain optimization
        captain_result = await optimize_captain_selection(
            playing_xi=playing_xi,
            predicted_points=predicted_points,
            ownership_data=ownership_data,
            risk_preference=strategy
        )
        
        if captain_result['status'] != 'optimal':
            return f"❌ Captain optimization failed: {captain_result.get('error', 'Unknown error')}"
        
        # Format advice
        captain_id = captain_result['recommended_captain']
        vice_id = captain_result.get('vice_captain')
        
        captain_player = players_df[players_df['id'] == captain_id].iloc[0]
        captain_advice = f"👑 **Captain Advice** ({strategy} strategy)\n\n"
        
        captain_advice += f"**Recommended Captain:**\n"
        captain_advice += f"• {captain_player['web_name']} ({captain_player['first_name']} {captain_player['second_name']})\n"
        captain_advice += f"• Expected Points: {predicted_points[captain_id]:.1f}\n"
        captain_advice += f"• Ownership: {ownership_data[captain_id]:.1f}%\n"
        captain_advice += f"• Risk Level: {_get_risk_level(ownership_data[captain_id])}\n\n"
        
        if vice_id:
            vice_player = players_df[players_df['id'] == vice_id].iloc[0]
            captain_advice += f"**Vice Captain:**\n"
            captain_advice += f"• {vice_player['web_name']}\n"
            captain_advice += f"• Expected Points: {predicted_points[vice_id]:.1f}\n"
            captain_advice += f"• Ownership: {ownership_data[vice_id]:.1f}%\n\n"
        
        # Show alternatives
        if len(captain_result.get('all_options', [])) > 1:
            captain_advice += f"**Alternative Options:**\n"
            for option in captain_result['all_options'][1:3]:  # Show top 2 alternatives
                alt_player = players_df[players_df['id'] == option['player_id']].iloc[0]
                captain_advice += f"• {alt_player['web_name']} ({option['expected_points']:.1f} pts, {option['ownership']:.1f}% owned)\n"
        
        logger.info(f"Captain advice generated for team {target_team_id}")
        return captain_advice.strip()
        
    except Exception as e:
        logger.error(f"Captain advice failed: {e}")
        return f"❌ Failed to get captain advice: {str(e)}"


def _generate_team_recommendations(avg_points: float, avg_bench_points: float, total_hits: int, num_games: int) -> str:
    """Generate team recommendations based on performance metrics."""
    recommendations = []
    
    if avg_points < 45:
        recommendations.append("• Consider major team restructuring - performance is below average")
    elif avg_points < 55:
        recommendations.append("• Look for 1-2 key upgrades to improve consistency") 
    else:
        recommendations.append("• Strong team performance - minor tweaks only")
    
    if avg_bench_points >= 3:
        recommendations.append("• Optimize starting XI selection - too many points left on bench")
    
    if total_hits > num_games:
        recommendations.append("• Reduce transfer frequency - hits are hurting overall score")
    
    if not recommendations:
        recommendations.append("• Team is performing well across all metrics")
    
    return "\n".join(recommendations)


def _generate_player_insights(player: Dict, form: float, ownership: float, value_score: float) -> str:
    """Generate player insights based on stats."""
    insights = []
    
    if form >= 6:
        insights.append("• Excellent recent form - in great scoring streak")
    elif form >= 4:
        insights.append("• Good recent form - consistent performer")
    else:
        insights.append("• Poor recent form - avoid until form improves")
    
    if ownership < 5:
        insights.append("• Low ownership - potential differential pick")
    elif ownership > 50:
        insights.append("• High ownership - template player, risky to avoid")
    
    if value_score >= 5:
        insights.append("• Excellent value for money - strong points per £")
    elif value_score < 3:
        insights.append("• Poor value - expensive for points returned")
    
    return "\n".join(insights) if insights else "• Standard performer with typical metrics"


def _generate_player_recommendation(player: Dict, form: float, ownership: float, value_score: float, price: float) -> str:
    """Generate player recommendation."""
    if form >= 6 and value_score >= 4:
        return "🟢 **STRONG BUY** - Excellent form and value"
    elif form >= 4 and value_score >= 3:
        return "🟡 **CONSIDER** - Good option worth monitoring"
    elif form < 3 or value_score < 2:
        return "🔴 **AVOID** - Poor form or value, look elsewhere"
    else:
        return "⚪ **NEUTRAL** - Average option, not priority"


def _get_risk_level(ownership: float) -> str:
    """Get risk level based on ownership."""
    if ownership < 20:
        return "High Risk/High Reward"
    elif ownership < 50:
        return "Medium Risk"
    else:
        return "Low Risk/Template"


# Convenience function to create FPL Manager agent with dependencies
def create_fpl_manager_agent(
    fpl_team_id: int,
    database_url: str,
    session_id: Optional[str] = None
) -> Agent:
    """
    Create FPL Manager agent with specified dependencies.
    
    Args:
        fpl_team_id: FPL team ID
        database_url: Database connection URL
        session_id: Optional session identifier
        
    Returns:
        Configured FPL Manager agent
    """
    return fpl_manager_agent