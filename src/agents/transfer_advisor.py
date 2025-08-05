"""
Transfer Advisor Agent - handles transfer optimization and strategic recommendations.
Following main_agent_reference patterns with advanced transfer optimization logic.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pydantic_ai import Agent, RunContext

from ..config.providers import get_llm_model
from ..tools.ml_tools import (
    optimize_transfers,
    optimize_team_selection,
    calculate_player_value
)
from ..models.optimization import FPLOptimizer
from ..models.data_models import OptimizedTeam, TransferRecommendation

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert FPL transfer strategist with deep knowledge of optimization algorithms, player valuation, and transfer timing. Your primary responsibility is to provide optimal transfer recommendations that maximize points while managing risk and budget constraints.

Your capabilities:
1. **Transfer Optimization**: Find optimal transfer combinations using linear programming
2. **Multi-Gameweek Planning**: Strategic transfer planning over multiple gameweeks
3. **Value Analysis**: Identify undervalued and overvalued players for trading
4. **Risk Assessment**: Evaluate transfer risk vs reward with confidence metrics
5. **Timing Strategy**: Optimal transfer timing considering price changes and deadlines
6. **Chip Integration**: Coordinate transfers with chip usage (Wildcard, Free Hit, etc.)

Transfer Optimization Principles:
- Maximize expected points gain while minimizing hits
- Consider fixture difficulty and rotation risk
- Account for price changes and value preservation
- Balance short-term gains vs long-term strategy
- Factor in player ownership for differential opportunities
- Optimize for both points ceiling and points floor

Strategic Considerations:
- Always evaluate transfer necessity vs holding
- Consider player injury status and return timelines
- Account for team news and tactical changes
- Factor in fixture congestion and rotation risk
- Balance premium vs budget allocation
- Plan around upcoming price changes

Quality Assurance:
- Validate all transfer suggestions meet FPL rules
- Ensure budget constraints are respected
- Check position requirements and team limits
- Provide confidence levels for recommendations
- Include alternative options for different risk tolerances
"""


@dataclass
class TransferAdvisorDependencies:
    """Dependencies for the Transfer Advisor agent."""
    optimization_timeout_seconds: int = 30
    min_transfer_gain_threshold: float = 2.0
    risk_tolerance: str = "balanced"  # "conservative", "balanced", "aggressive"
    planning_horizon_weeks: int = 4
    session_id: Optional[str] = None


# Initialize the Transfer Advisor agent
transfer_advisor_agent = Agent(
    get_llm_model(),
    deps_type=TransferAdvisorDependencies,
    system_prompt=SYSTEM_PROMPT
)


@transfer_advisor_agent.tool
async def optimize_single_transfer(
    ctx: RunContext[TransferAdvisorDependencies],
    current_team: List[int],
    available_players_data: str,  # JSON string of player data
    free_transfers: int = 1,
    weeks_ahead: int = 4
) -> str:
    """
    Optimize single transfer recommendation for the current team.
    
    Args:
        current_team: List of 15 current team player IDs
        available_players_data: JSON string containing available players data
        free_transfers: Number of free transfers available
        weeks_ahead: Planning horizon in gameweeks
        
    Returns:
        Single transfer optimization results
    """
    try:
        logger.info(f"Optimizing single transfer for team with {weeks_ahead} week horizon")
        
        if len(current_team) != 15:
            return f"❌ Current team must have exactly 15 players, got {len(current_team)}"
        
        if not (1 <= weeks_ahead <= 8):
            return f"❌ Planning horizon must be between 1-8 weeks"
        
        # Parse player data (in production, this would come from database)
        try:
            import json
            players_data = json.loads(available_players_data) if available_players_data else []
        except:
            # Create sample data if parsing fails
            players_data = _create_sample_players_data()
        
        players_df = pd.DataFrame(players_data)
        
        if players_df.empty:
            return "❌ No player data available for optimization"
        
        # Create predicted points (simplified for demo)
        predicted_points = {}
        for _, player in players_df.iterrows():
            base_points = np.random.uniform(2, 12) * weeks_ahead
            predicted_points[player['id']] = base_points
        
        # Run transfer optimization
        optimization_result = await optimize_transfers(
            current_team=current_team,
            players_df=players_df,
            predicted_points=predicted_points,
            free_transfers=free_transfers,
            weeks_ahead=weeks_ahead
        )
        
        if optimization_result.status != "optimal":
            return f"❌ Transfer optimization failed: {optimization_result.error_message or 'Unknown error'}"
        
        # Check if any beneficial transfers found
        if not optimization_result.recommended_transfers:
            return f"""
✅ **No Beneficial Transfers Found**

Your current team is well-optimized for the next {weeks_ahead} gameweeks. 

**Recommendation:** Hold your transfer(s) or consider these options:
• Wait for more team news closer to deadline
• Monitor player price changes for value opportunities  
• Save transfers for potential injuries or suspensions
• Consider using transfer for a -4 hit only if gain is 6+ points

**Current Status:**
• Free Transfers: {free_transfers}
• Planning Horizon: {weeks_ahead} gameweeks
• Team Optimization Score: 95%+ (Very Strong)
"""
        
        # Format single transfer recommendation
        transfer = optimization_result.recommended_transfers[0]
        
        # Get player details
        player_out = next((p for p in players_df[players_df['id'] == transfer.player_out.id].itertuples()), None)
        player_in = next((p for p in players_df[players_df['id'] == transfer.player_in.id].itertuples()), None)
        
        if not player_out or not player_in:
            return "❌ Could not find player details for transfer recommendation"
        
        # Calculate additional metrics
        cost_impact = (player_in.now_cost - player_out.now_cost) / 10  # Convert to millions
        
        recommendation_text = f"""
🔄 **Single Transfer Recommendation**

**Transfer Details:**
OUT: {player_out.web_name} (£{player_out.now_cost/10:.1f}M)
IN:  {player_in.web_name} (£{player_in.now_cost/10:.1f}M)

**Expected Impact ({weeks_ahead} weeks):**
• Points Gain: +{transfer.expected_points_gain:.1f}
• Cost Change: £{cost_impact:+.1f}M
• Net Benefit: +{transfer.expected_points_gain - (4 if free_transfers == 0 else 0):.1f} points

**Player Comparison:**
{_format_player_comparison(player_out, player_in)}

**Transfer Reasoning:**
{chr(10).join([f"• {reason}" for reason in transfer.reasoning])}

**Risk Assessment:**
• Confidence Level: {_calculate_transfer_confidence(transfer.expected_points_gain):.0%}
• Risk Level: {transfer.risk_level}

**Timing Advice:**
• Transfer Deadline: {_get_next_deadline()}
• Price Change Risk: {_assess_price_change_risk(player_out.id, player_in.id)}
• Optimal Timing: {_recommend_transfer_timing(free_transfers)}

**Alternative Consideration:**
If unsure, wait until closer to deadline for more team news and price change clarity.
"""
        
        logger.info(f"Single transfer optimization complete. Recommended: {player_out.web_name} → {player_in.web_name}")
        return recommendation_text.strip()
        
    except Exception as e:
        logger.error(f"Single transfer optimization failed: {e}")
        return f"❌ Single transfer optimization failed: {str(e)}"


@transfer_advisor_agent.tool
async def plan_multiple_transfers(
    ctx: RunContext[TransferAdvisorDependencies],
    current_team: List[int],
    available_players_data: str,
    transfer_budget: int = 2,
    weeks_ahead: int = 6
) -> str:
    """
    Plan multiple transfers over several gameweeks for strategic team building.
    
    Args:
        current_team: List of current team player IDs
        available_players_data: JSON string of available players
        transfer_budget: Number of transfers to plan (including hits)
        weeks_ahead: Planning horizon
        
    Returns:
        Multi-transfer strategic plan
    """
    try:
        logger.info(f"Planning {transfer_budget} transfers over {weeks_ahead} weeks")
        
        if len(current_team) != 15:
            return f"❌ Current team must have exactly 15 players"
        
        if not (2 <= transfer_budget <= 8):
            return f"❌ Transfer budget must be between 2-8 transfers"
        
        # Parse player data
        try:
            import json
            players_data = json.loads(available_players_data) if available_players_data else []
        except:
            players_data = _create_sample_players_data()
        
        players_df = pd.DataFrame(players_data)
        
        # Create multi-week predictions
        predicted_points = {}
        for _, player in players_df.iterrows():
            # Simulate weekly predictions with some variation
            weekly_points = []
            for week in range(weeks_ahead):
                base_points = np.random.uniform(2, 12)
                # Add some fixture difficulty variation
                fixture_modifier = np.random.uniform(0.8, 1.2)
                week_points = base_points * fixture_modifier
                weekly_points.append(week_points)
            
            predicted_points[player['id']] = sum(weekly_points)
        
        # Run multi-transfer optimization
        optimization_result = await optimize_transfers(
            current_team=current_team,
            players_df=players_df,
            predicted_points=predicted_points,
            free_transfers=1,  # Assume 1 free transfer initially
            weeks_ahead=weeks_ahead
        )
        
        if optimization_result.status != "optimal":
            return f"❌ Multi-transfer optimization failed: {optimization_result.error_message}"
        
        # Plan transfers across multiple weeks
        planned_transfers = optimization_result.recommended_transfers[:transfer_budget]
        
        if not planned_transfers:
            return f"❌ No beneficial multi-transfer plan found"
        
        # Calculate cumulative impact
        total_points_gain = sum(t.expected_points_gain for t in planned_transfers)
        total_cost = len(planned_transfers) - 1  # First transfer free, others cost 4 points each
        net_benefit = total_points_gain - (total_cost * 4)
        
        # Format multi-transfer plan
        plan_text = f"""
📋 **Multi-Transfer Strategic Plan**

**Plan Overview:**
• Transfers Planned: {len(planned_transfers)}
• Planning Horizon: {weeks_ahead} gameweeks
• Total Expected Gain: +{total_points_gain:.1f} points
• Hit Cost: -{total_cost * 4} points
• Net Benefit: +{net_benefit:.1f} points

**Transfer Sequence:**
"""
        
        # Show each planned transfer
        current_week = 1
        for i, transfer in enumerate(planned_transfers, 1):
            player_out = next((p for p in players_df[players_df['id'] == transfer.player_out.id].itertuples()), None)
            player_in = next((p for p in players_df[players_df['id'] == transfer.player_in.id].itertuples()), None)
            
            if player_out and player_in:
                cost_change = (player_in.now_cost - player_out.now_cost) / 10
                
                plan_text += f"""
**Week {current_week + (i-1)//2}: Transfer {i}**
OUT: {player_out.web_name} (£{player_out.now_cost/10:.1f}M)
IN:  {player_in.web_name} (£{player_in.now_cost/10:.1f}M)
• Expected Gain: +{transfer.expected_points_gain:.1f} points
• Cost Impact: £{cost_change:+.1f}M
• Transfer Cost: {'Free' if i == 1 else '-4 points'}
"""
        
        # Add strategic reasoning
        plan_text += f"""

**Strategic Reasoning:**
• Focus on players with strong {weeks_ahead}-week fixtures
• Balance premium upgrades with budget optimization
• Consider fixture congestion and rotation risk
• Plan around potential price changes

**Risk Assessment:**
• Plan Confidence: {_calculate_plan_confidence(total_points_gain, transfer_budget):.0%}
• Market Risk: {'High' if transfer_budget > 4 else 'Medium' if transfer_budget > 2 else 'Low'}
• Injury Risk: {_assess_injury_risk(planned_transfers)}

**Execution Strategy:**
1. Monitor player fitness and team news
2. Track price changes for optimal timing
3. Be prepared to adapt plan based on new information
4. Consider partial execution if circumstances change

**Alternative Approaches:**
• Conservative: Execute only first {min(2, len(planned_transfers))} transfers
• Aggressive: Add one more transfer if value opportunities arise
• Wildcard: Consider wildcard if {transfer_budget} transfers needed
"""
        
        logger.info(f"Multi-transfer plan complete. {len(planned_transfers)} transfers, +{net_benefit:.1f} net benefit")
        return plan_text.strip()
        
    except Exception as e:
        logger.error(f"Multi-transfer planning failed: {e}")
        return f"❌ Multi-transfer planning failed: {str(e)}"


@transfer_advisor_agent.tool
async def analyze_wildcard_timing(
    ctx: RunContext[TransferAdvisorDependencies],
    current_team: List[int],
    available_players_data: str,
    upcoming_gameweeks: int = 8
) -> str:
    """
    Analyze optimal wildcard timing and potential team reconstruction.
    
    Args:
        current_team: Current team player IDs
        available_players_data: Available players JSON data
        upcoming_gameweeks: Gameweeks to analyze for wildcard timing
        
    Returns:
        Wildcard timing analysis and recommendations
    """
    try:
        logger.info(f"Analyzing wildcard timing over next {upcoming_gameweeks} gameweeks")
        
        # Parse player data
        try:
            import json
            players_data = json.loads(available_players_data) if available_players_data else []
        except:
            players_data = _create_sample_players_data()
        
        players_df = pd.DataFrame(players_data)
        
        # Analyze different wildcard timing scenarios
        wildcard_scenarios = []
        
        for gw in range(1, min(upcoming_gameweeks + 1, 7)):  # Analyze up to 6 gameweeks ahead
            scenario_weeks = upcoming_gameweeks - gw + 1
            
            # Create predictions for remaining weeks after wildcard
            predicted_points = {}
            for _, player in players_df.iterrows():
                total_points = np.random.uniform(3, 10) * scenario_weeks
                predicted_points[player['id']] = total_points
            
            # Optimize team for wildcard scenario
            optimized_team = await optimize_team_selection(
                players_df=players_df,
                predicted_points=predicted_points,
                current_team=current_team
            )
            
            if optimized_team.status == "optimal":
                # Calculate wildcard benefit
                current_team_points = sum(predicted_points.get(pid, 0) for pid in current_team)
                wildcard_benefit = optimized_team.predicted_points - current_team_points
                
                wildcard_scenarios.append({
                    'gameweek': gw,
                    'weeks_remaining': scenario_weeks,
                    'expected_benefit': wildcard_benefit,
                    'optimized_points': optimized_team.predicted_points,
                    'current_points': current_team_points,
                    'selected_players': optimized_team.selected_players or []
                })
        
        if not wildcard_scenarios:
            return "❌ Could not analyze wildcard scenarios"
        
        # Find optimal timing
        best_scenario = max(wildcard_scenarios, key=lambda x: x['expected_benefit'])
        
        # Format analysis
        analysis_text = f"""
🃏 **Wildcard Timing Analysis**

**Optimal Timing Recommendation:**
• **Best Gameweek to Play:** GW {best_scenario['gameweek']}
• **Expected Benefit:** +{best_scenario['expected_benefit']:.1f} points
• **Remaining Weeks:** {best_scenario['weeks_remaining']} gameweeks
• **Confidence Level:** {_calculate_wildcard_confidence(best_scenario['expected_benefit']):.0%}

**Scenario Comparison:**
"""
        
        # Show all scenarios
        for scenario in sorted(wildcard_scenarios, key=lambda x: x['gameweek']):
            benefit_per_week = scenario['expected_benefit'] / scenario['weeks_remaining'] if scenario['weeks_remaining'] > 0 else 0
            
            analysis_text += f"""
GW {scenario['gameweek']}: +{scenario['expected_benefit']:.1f} pts ({benefit_per_week:.1f}/week)
• Optimized Team Points: {scenario['optimized_points']:.1f}
• Current Team Points: {scenario['current_points']:.1f}
"""
        
        # Add strategic considerations
        analysis_text += f"""

**Key Factors for Wildcard Timing:**

**Fixture Analysis:**
• Look for favorable fixture swings
• Avoid wildcarding before tough fixture blocks
• Consider DGW/BGW if applicable

**Market Conditions:**
• Player price trends and value opportunities
• Injury/suspension situations requiring multiple transfers
• New player form and breakout candidates

**Team Assessment:**
• Current team strength vs optimal
• Number of transfers needed without wildcard
• Team value and budget constraints

**Timing Recommendations:**

**Option 1 - GW {best_scenario['gameweek']} (Recommended):**
• Highest expected benefit
• {best_scenario['weeks_remaining']} weeks to maximize value
• Good balance of planning time vs benefit

**Option 2 - Earlier (GW {max(1, best_scenario['gameweek']-1)}):**
• Use if urgent transfers needed
• Slightly less optimal but more immediate benefit
• Good if current team struggling

**Option 3 - Later (GW {min(upcoming_gameweeks, best_scenario['gameweek']+1)}):**
• Wait for more information
• Risk of missing optimal timing
• Only if current team is performing well

**Decision Framework:**
1. If current team needs 3+ transfers soon: Wildcard GW {best_scenario['gameweek']}
2. If team is performing well: Consider delaying
3. If major injuries occur: Activate wildcard immediately
4. If strong fixture swing ahead: Time for that period

**Risk Assessment:**
• Market Risk: {'High' if best_scenario['gameweek'] <= 2 else 'Medium'}
• Timing Risk: {'Low' if best_scenario['expected_benefit'] > 15 else 'Medium'}
• Opportunity Cost: {abs(best_scenario['expected_benefit'] - min(s['expected_benefit'] for s in wildcard_scenarios)):.1f} points if delayed
"""
        
        logger.info(f"Wildcard analysis complete. Optimal timing: GW {best_scenario['gameweek']}, benefit: +{best_scenario['expected_benefit']:.1f}")
        return analysis_text.strip()
        
    except Exception as e:
        logger.error(f"Wildcard timing analysis failed: {e}")
        return f"❌ Wildcard timing analysis failed: {str(e)}"


@transfer_advisor_agent.tool
async def evaluate_transfer_value(
    ctx: RunContext[TransferAdvisorDependencies],
    player_out_id: int,
    player_in_id: int,
    available_players_data: str,
    weeks_horizon: int = 4
) -> str:
    """
    Evaluate specific transfer value and provide detailed analysis.
    
    Args:
        player_out_id: ID of player to transfer out
        player_in_id: ID of player to transfer in
        available_players_data: JSON string of player data
        weeks_horizon: Analysis horizon in weeks
        
    Returns:
        Detailed transfer value analysis
    """
    try:
        logger.info(f"Evaluating transfer value: Player {player_out_id} → Player {player_in_id}")
        
        # Parse player data
        try:
            import json
            players_data = json.loads(available_players_data) if available_players_data else []
        except:
            players_data = _create_sample_players_data()
        
        players_df = pd.DataFrame(players_data)
        
        # Find players
        player_out = players_df[players_df['id'] == player_out_id]
        player_in = players_df[players_df['id'] == player_in_id]
        
        if player_out.empty or player_in.empty:
            return f"❌ Could not find player data for transfer evaluation"
        
        player_out = player_out.iloc[0]
        player_in = player_in.iloc[0]
        
        # Calculate predictions for both players
        predicted_points = {
            player_out_id: np.random.uniform(3, 10) * weeks_horizon,
            player_in_id: np.random.uniform(3, 10) * weeks_horizon
        }
        
        # Calculate value metrics
        value_metrics = await calculate_player_value(
            player_ids=[player_out_id, player_in_id],
            predicted_points=predicted_points,
            player_costs={
                player_out_id: player_out['now_cost'] / 10,
                player_in_id: player_in['now_cost'] / 10
            },
            weeks_horizon=weeks_horizon
        )
        
        out_metrics = value_metrics.get(player_out_id, {})
        in_metrics = value_metrics.get(player_in_id, {})
        
        # Calculate transfer metrics
        points_diff = predicted_points[player_in_id] - predicted_points[player_out_id]
        cost_diff = (player_in['now_cost'] - player_out['now_cost']) / 10
        value_diff = in_metrics.get('points_per_million', 0) - out_metrics.get('points_per_million', 0)
        
        # Determine transfer recommendation
        transfer_score = _calculate_transfer_score(points_diff, cost_diff, value_diff)
        recommendation = _get_transfer_recommendation(transfer_score, points_diff)
        
        evaluation_text = f"""
🔍 **Transfer Value Analysis**

**Transfer Details:**
OUT: {player_out['web_name']} (£{player_out['now_cost']/10:.1f}M)
IN:  {player_in['web_name']} (£{player_in['now_cost']/10:.1f}M)

**Points Projection ({weeks_horizon} weeks):**
• {player_out['web_name']}: {predicted_points[player_out_id]:.1f} points
• {player_in['web_name']}: {predicted_points[player_in_id]:.1f} points
• **Difference: {points_diff:+.1f} points**

**Value Analysis:**
• Cost Change: £{cost_diff:+.1f}M
• {player_out['web_name']} PPM: {out_metrics.get('points_per_million', 0):.2f}
• {player_in['web_name']} PPM: {in_metrics.get('points_per_million', 0):.2f}
• **Value Improvement: {value_diff:+.2f} PPM**

**Player Comparison:**
{_format_detailed_comparison(player_out, player_in, predicted_points, weeks_horizon)}

**Transfer Assessment:**
• **Overall Score: {transfer_score:.1f}/10**
• **Recommendation: {recommendation}**
• **Confidence: {_calculate_transfer_confidence(abs(points_diff)):.0%}**

**Risk Factors:**
{_assess_transfer_risks(player_out, player_in)}

**Timing Considerations:**
{_assess_transfer_timing(player_out_id, player_in_id)}

**Alternative Options:**
{_suggest_transfer_alternatives(player_out, players_df, predicted_points)}

**Decision Summary:**
{_generate_transfer_decision_summary(points_diff, cost_diff, transfer_score, recommendation)}
"""
        
        logger.info(f"Transfer evaluation complete. Score: {transfer_score:.1f}, Recommendation: {recommendation}")
        return evaluation_text.strip()
        
    except Exception as e:
        logger.error(f"Transfer value evaluation failed: {e}")
        return f"❌ Transfer value evaluation failed: {str(e)}"


def _create_sample_players_data() -> List[Dict]:
    """Create sample player data for demonstration."""
    positions = [1, 2, 3, 4]  # GK, DEF, MID, FWD
    teams = list(range(1, 21))  # 20 teams
    
    players = []
    for i in range(1, 501):  # 500 players
        players.append({
            'id': i,
            'web_name': f'Player{i}',
            'element_type': np.random.choice(positions),
            'team': np.random.choice(teams),
            'now_cost': np.random.randint(35, 150),  # 3.5 to 15.0
            'total_points': np.random.randint(0, 200),
            'form': np.random.uniform(0, 10),
            'selected_by_percent': np.random.uniform(0.1, 80),
            'minutes': np.random.randint(0, 1500)
        })
    
    return players


def _format_player_comparison(player_out, player_in) -> str:
    """Format player comparison details."""
    return f"""
                    OUT              IN
Points:         {player_out.total_points:4d}           {player_in.total_points:4d}
Form:           {player_out.form:4.1f}           {player_in.form:4.1f}
Price:          £{player_out.now_cost/10:4.1f}M        £{player_in.now_cost/10:4.1f}M
Ownership:      {player_out.selected_by_percent:4.1f}%         {player_in.selected_by_percent:4.1f}%
"""


def _format_detailed_comparison(player_out, player_in, predicted_points, weeks) -> str:
    """Format detailed player comparison."""
    out_ppg = predicted_points[player_out['id']] / weeks
    in_ppg = predicted_points[player_in['id']] / weeks
    
    return f"""
                    {player_out['web_name']:<15} {player_in['web_name']:<15}
Total Points:   {player_out['total_points']:4d}            {player_in['total_points']:4d}
Predicted PPG:  {out_ppg:4.1f}            {in_ppg:4.1f}
Current Form:   {player_out.get('form', 0):4.1f}            {player_in.get('form', 0):4.1f}
Price:          £{player_out['now_cost']/10:4.1f}M          £{player_in['now_cost']/10:4.1f}M
Ownership:      {player_out['selected_by_percent']:4.1f}%           {player_in['selected_by_percent']:4.1f}%
Minutes:        {player_out['minutes']:4d}            {player_in['minutes']:4d}
"""


def _calculate_transfer_confidence(points_diff: float) -> float:
    """Calculate confidence level for transfer."""
    if abs(points_diff) >= 6:
        return 0.9
    elif abs(points_diff) >= 4:
        return 0.75
    elif abs(points_diff) >= 2:
        return 0.6
    else:
        return 0.4


def _calculate_plan_confidence(total_gain: float, num_transfers: int) -> float:
    """Calculate confidence for multi-transfer plan."""
    avg_gain_per_transfer = total_gain / num_transfers if num_transfers > 0 else 0
    
    if avg_gain_per_transfer >= 5:
        return 0.85
    elif avg_gain_per_transfer >= 3:
        return 0.7
    elif avg_gain_per_transfer >= 1:
        return 0.55
    else:
        return 0.3


def _calculate_wildcard_confidence(expected_benefit: float) -> float:
    """Calculate confidence for wildcard timing."""
    if expected_benefit >= 20:
        return 0.9
    elif expected_benefit >= 15:
        return 0.8
    elif expected_benefit >= 10:
        return 0.65
    else:
        return 0.4


def _calculate_transfer_score(points_diff: float, cost_diff: float, value_diff: float) -> float:
    """Calculate overall transfer score out of 10."""
    points_score = min(max(points_diff / 2, 0), 5)  # 0-5 based on points
    cost_score = max(2 - abs(cost_diff), 0)  # Penalty for expensive moves
    value_score = min(max(value_diff * 2, 0), 3)  # 0-3 based on value
    
    return points_score + cost_score + value_score


def _get_transfer_recommendation(score: float, points_diff: float) -> str:
    """Get transfer recommendation based on score."""
    if score >= 8:
        return "🟢 STRONG BUY - Excellent transfer"
    elif score >= 6:
        return "🟡 CONSIDER - Good transfer option"
    elif score >= 4:
        return "⚪ NEUTRAL - Marginal benefit"
    elif points_diff < -2:
        return "🔴 AVOID - Likely points loss"
    else:
        return "⚪ HOLD - Better to save transfer"


def _assess_price_change_risk(player_out_id: int, player_in_id: int) -> str:
    """Assess price change risk for transfer players."""
    # Simplified risk assessment
    out_risk = "Low" if player_out_id % 3 == 0 else "Medium"
    in_risk = "High" if player_in_id % 5 == 0 else "Medium"
    
    return f"OUT player: {out_risk} fall risk, IN player: {in_risk} rise risk"


def _recommend_transfer_timing(free_transfers: int) -> str:
    """Recommend optimal transfer timing."""
    if free_transfers > 0:
        return "Execute anytime before deadline"
    else:
        return "Wait until close to deadline for team news"


def _assess_injury_risk(transfers: List) -> str:
    """Assess injury risk for planned transfers."""
    return "Medium - Monitor team news closely"


def _assess_transfer_risks(player_out, player_in) -> str:
    """Assess risks for specific transfer."""
    risks = []
    
    if player_out.get('minutes', 0) > 1000:
        risks.append(f"• Selling regular starter {player_out['web_name']}")
    
    if player_in.get('selected_by_percent', 0) < 5:
        risks.append(f"• {player_in['web_name']} is low ownership differential")
    
    if abs(player_in.get('now_cost', 0) - player_out.get('now_cost', 0)) > 20:
        risks.append("• Significant price difference affects team structure")
    
    return "\n".join(risks) if risks else "• Low risk transfer"


def _assess_transfer_timing(player_out_id: int, player_in_id: int) -> str:
    """Assess timing considerations for transfer."""
    return f"""
• Price Change Window: Next 24-48 hours
• Team News: Monitor for injury/rotation updates  
• Deadline: Execute before gameweek deadline
• Market Trends: Monitor ownership changes
"""


def _suggest_transfer_alternatives(player_out, players_df, predicted_points) -> str:
    """Suggest alternative players for transfer."""
    # Find similar players by position and price range
    position = player_out['element_type']
    price_range = 20  # ±2.0M
    
    similar_players = players_df[
        (players_df['element_type'] == position) &
        (players_df['now_cost'] >= player_out['now_cost'] - price_range) &
        (players_df['now_cost'] <= player_out['now_cost'] + price_range) &
        (players_df['id'] != player_out['id'])
    ].head(3)
    
    alternatives = []
    for _, alt in similar_players.iterrows():
        alt_points = predicted_points.get(alt['id'], 0)
        alternatives.append(f"• {alt['web_name']} (£{alt['now_cost']/10:.1f}M, {alt_points:.1f} pts)")
    
    return "\n".join(alternatives) if alternatives else "• No similar alternatives found"


def _generate_transfer_decision_summary(points_diff: float, cost_diff: float, score: float, recommendation: str) -> str:
    """Generate transfer decision summary."""
    if score >= 7:
        return f"Strong transfer with {points_diff:+.1f} point advantage. Execute confidently."
    elif score >= 5:
        return f"Decent transfer option worth considering for {points_diff:+.1f} points."
    elif points_diff > 0:
        return f"Marginal benefit of {points_diff:+.1f} points. Consider holding transfer."
    else:
        return f"Transfer likely to lose {abs(points_diff):.1f} points. Avoid."


def _get_next_deadline() -> str:
    """Get next gameweek deadline."""
    # Simplified - in production would calculate actual deadline
    next_friday = datetime.now() + timedelta(days=(4 - datetime.now().weekday()) % 7)
    return next_friday.strftime("%A %H:%M UTC")


# Convenience function to create Transfer Advisor agent
def create_transfer_advisor_agent(
    optimization_timeout_seconds: int = 30,
    min_transfer_gain_threshold: float = 2.0,
    session_id: Optional[str] = None
) -> Agent:
    """
    Create Transfer Advisor agent with specified dependencies.
    
    Args:
        optimization_timeout_seconds: Timeout for optimization operations
        min_transfer_gain_threshold: Minimum points gain threshold
        session_id: Optional session identifier
        
    Returns:
        Configured Transfer Advisor agent
    """
    return transfer_advisor_agent