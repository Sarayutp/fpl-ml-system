"""
PuLP-based team optimization for FPL following research patterns.
Target: Optimization completing within 5 seconds (PRP requirement).
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pulp import *

from ..config.settings import settings
from ..models.data_models import Player, OptimizedTeam, TransferRecommendation

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Configuration for FPL optimization constraints."""
    
    # Team structure constraints (FPL rules)
    total_players: int = 15
    goalkeepers: int = 2
    defenders: int = 5
    midfielders: int = 5
    forwards: int = 3
    
    # Budget constraints
    max_budget: float = 100.0  # £100M
    min_budget: float = 85.0   # Minimum to spend
    
    # Team diversity constraints
    max_players_per_team: int = 3
    min_teams_represented: int = 8
    
    # Transfer constraints
    max_free_transfers: int = 1
    transfer_cost: int = 4  # Points per extra transfer
    
    # Playing XI constraints
    playing_gk: int = 1
    min_playing_def: int = 3
    max_playing_def: int = 5
    min_playing_mid: int = 2
    max_playing_mid: int = 5
    min_playing_fwd: int = 1
    max_playing_fwd: int = 3
    playing_total: int = 11


class FPLOptimizer:
    """
    Advanced team optimization using linear programming with PuLP.
    Implements FPL-specific constraints and multi-objective optimization.
    """
    
    def __init__(self, constraints: Optional[OptimizationConstraints] = None):
        self.constraints = constraints or OptimizationConstraints()
        self.timeout = settings.optimization_timeout
        self.solver = PULP_CBC_CMD(msg=0, timeLimit=self.timeout)
        
    def optimize_team(
        self, 
        players_df: pd.DataFrame, 
        predicted_points: Dict[int, float],
        current_team: Optional[List[int]] = None,
        force_players: Optional[List[int]] = None,
        exclude_players: Optional[List[int]] = None
    ) -> OptimizedTeam:
        """
        Find optimal 15-player squad using linear programming.
        
        Args:
            players_df: DataFrame with player data
            predicted_points: Dictionary mapping player_id to predicted points
            current_team: Current team player IDs (for transfer optimization)
            force_players: Players that must be included
            exclude_players: Players that must be excluded
            
        Returns:
            OptimizedTeam with results and metadata
        """
        start_time = time.time()
        
        logger.info(f"Starting team optimization with {len(players_df)} players")
        
        try:
            # Prepare data
            players_df = self._prepare_data(players_df, predicted_points)
            
            if players_df.empty:
                return OptimizedTeam(
                    status="error",
                    error_message="No valid players available for optimization"
                )
            
            # Create decision variables
            player_vars = {}
            for _, player in players_df.iterrows():
                player_vars[player['id']] = LpVariable(
                    f"player_{player['id']}", 
                    cat='Binary'
                )
            
            # Create optimization problem
            prob = LpProblem("FPL_Team_Optimization", LpMaximize)
            
            # Objective: maximize predicted points
            prob += lpSum([
                predicted_points.get(player['id'], 0) * player_vars[player['id']]
                for _, player in players_df.iterrows()
            ])
            
            # Add constraints
            self._add_squad_constraints(prob, players_df, player_vars)
            self._add_budget_constraints(prob, players_df, player_vars)
            self._add_position_constraints(prob, players_df, player_vars)
            self._add_team_diversity_constraints(prob, players_df, player_vars)
            
            # Handle forced/excluded players
            if force_players:
                self._add_force_constraints(prob, player_vars, force_players)
            if exclude_players:
                self._add_exclude_constraints(prob, player_vars, exclude_players)
            
            # Solve the problem
            logger.info("Solving optimization problem...")
            prob.solve(self.solver)
            
            optimization_time = time.time() - start_time
            
            # Extract and return results
            return self._extract_results(
                prob, players_df, player_vars, predicted_points, optimization_time
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizedTeam(
                status="error",
                error_message=str(e),
                optimization_time=time.time() - start_time
            )
    
    def optimize_transfers(
        self,
        current_team: List[int],
        players_df: pd.DataFrame,
        predicted_points: Dict[int, float],
        free_transfers: int = 1,
        weeks_ahead: int = 4,
        chip_active: Optional[str] = None
    ) -> OptimizedTeam:
        """
        Optimize transfer decisions for multiple gameweeks.
        
        Args:
            current_team: Current team player IDs
            players_df: Available players
            predicted_points: Predicted points per player
            free_transfers: Number of free transfers available
            weeks_ahead: Planning horizon in gameweeks
            chip_active: Active chip (wildcard, free_hit, etc.)
            
        Returns:
            OptimizedTeam with transfer recommendations
        """
        start_time = time.time()
        
        logger.info(f"Optimizing transfers for {weeks_ahead} weeks ahead")
        
        try:
            if chip_active == "wildcard":
                # Wildcard allows unlimited transfers
                return self.optimize_team(players_df, predicted_points)
            
            # Multi-period transfer optimization
            best_transfers = []
            cumulative_gain = 0
            current_team_copy = current_team.copy()
            remaining_transfers = free_transfers
            
            for week in range(weeks_ahead):
                # Discount future points
                week_predictions = {
                    pid: predicted_points.get(pid, 0) * (0.9 ** week)
                    for pid in predicted_points
                }
                
                # Find best single transfer for this week
                transfer = self._find_best_single_transfer(
                    current_team_copy, players_df, week_predictions, remaining_transfers > 0
                )
                
                if transfer and transfer['points_gain'] > 0:
                    best_transfers.append(transfer)
                    cumulative_gain += transfer['points_gain']
                    
                    # Update current team for next iteration
                    if transfer['player_out'] in current_team_copy:
                        current_team_copy.remove(transfer['player_out'])
                    current_team_copy.append(transfer['player_in'])
                    
                    # Track transfer usage
                    if remaining_transfers > 0:
                        remaining_transfers -= 1
            
            # Limit to available transfers
            recommended_transfers = best_transfers[:free_transfers]
            transfer_cost = max(0, len(recommended_transfers) - free_transfers) * self.constraints.transfer_cost
            
            optimization_time = time.time() - start_time
            
            return OptimizedTeam(
                status="optimal",
                recommended_transfers=self._convert_transfers_to_recommendations(
                    recommended_transfers, players_df
                ),
                expected_points_gain=cumulative_gain,
                transfer_cost=transfer_cost,
                optimization_time=optimization_time
            )
            
        except Exception as e:
            logger.error(f"Transfer optimization failed: {e}")
            return OptimizedTeam(
                status="error",
                error_message=str(e),
                optimization_time=time.time() - start_time
            )
    
    def optimize_playing_xi(
        self,
        squad_players: List[int],
        players_df: pd.DataFrame,
        predicted_points: Dict[int, float],
        captain_multiplier: float = 2.0
    ) -> Dict[str, Any]:
        """
        Optimize playing XI selection from 15-player squad.
        
        Args:
            squad_players: 15-player squad IDs
            players_df: Player data
            predicted_points: Predicted points
            captain_multiplier: Points multiplier for captain
            
        Returns:
            Dictionary with playing XI and captain selection
        """
        try:
            squad_df = players_df[players_df['id'].isin(squad_players)]
            
            if len(squad_df) != 15:
                raise ValueError(f"Squad must have exactly 15 players, got {len(squad_df)}")
            
            # Create decision variables for playing and captaincy
            playing_vars = {}
            captain_vars = {}
            
            for _, player in squad_df.iterrows():
                playing_vars[player['id']] = LpVariable(f"play_{player['id']}", cat='Binary')
                captain_vars[player['id']] = LpVariable(f"captain_{player['id']}", cat='Binary')
            
            # Create problem
            prob = LpProblem("FPL_Playing_XI", LpMaximize)
            
            # Objective: maximize points with captain bonus
            prob += lpSum([
                predicted_points.get(player['id'], 0) * playing_vars[player['id']] +
                predicted_points.get(player['id'], 0) * (captain_multiplier - 1) * captain_vars[player['id']]
                for _, player in squad_df.iterrows()
            ])
            
            # Playing XI constraints
            self._add_playing_xi_constraints(prob, squad_df, playing_vars, captain_vars)
            
            # Solve
            prob.solve(self.solver)
            
            # Extract results
            if prob.status == LpStatusOptimal:
                playing_xi = []
                captain = None
                bench = []
                
                for _, player in squad_df.iterrows():
                    if playing_vars[player['id']].varValue == 1:
                        playing_xi.append(player['id'])
                        if captain_vars[player['id']].varValue == 1:
                            captain = player['id']
                    else:
                        bench.append(player['id'])
                
                return {
                    'status': 'optimal',
                    'playing_xi': playing_xi,
                    'captain': captain,
                    'bench': bench,
                    'predicted_points': value(prob.objective),
                }
            else:
                return {'status': 'infeasible', 'error': f"Status: {LpStatus[prob.status]}"}
                
        except Exception as e:
            logger.error(f"Playing XI optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _prepare_data(self, players_df: pd.DataFrame, predicted_points: Dict[int, float]) -> pd.DataFrame:
        """Prepare and validate data for optimization."""
        df = players_df.copy()
        
        # Ensure required columns exist
        required_cols = ['id', 'element_type', 'team', 'now_cost']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add predicted points
        df['predicted_points'] = df['id'].map(predicted_points).fillna(0)
        
        # Convert price to millions
        if df['now_cost'].max() > 20:  # Assume it's in 0.1M units
            df['price_millions'] = df['now_cost'] / 10.0
        else:
            df['price_millions'] = df['now_cost']
        
        # Filter out invalid players
        df = df[
            (df['price_millions'] >= 3.5) & 
            (df['price_millions'] <= 15.0) &
            (df['element_type'].isin([1, 2, 3, 4]))
        ]
        
        return df
    
    def _add_squad_constraints(self, prob: LpProblem, players_df: pd.DataFrame, player_vars: Dict) -> None:
        """Add basic squad size constraints."""
        # Exactly 15 players
        prob += lpSum([
            player_vars[player['id']] for _, player in players_df.iterrows()
        ]) == self.constraints.total_players
    
    def _add_budget_constraints(self, prob: LpProblem, players_df: pd.DataFrame, player_vars: Dict) -> None:
        """Add budget constraints."""
        prob += lpSum([
            player['price_millions'] * player_vars[player['id']]
            for _, player in players_df.iterrows()
        ]) <= self.constraints.max_budget
        
        prob += lpSum([
            player['price_millions'] * player_vars[player['id']]
            for _, player in players_df.iterrows()
        ]) >= self.constraints.min_budget
    
    def _add_position_constraints(self, prob: LpProblem, players_df: pd.DataFrame, player_vars: Dict) -> None:
        """Add FPL position constraints."""
        position_limits = {
            1: self.constraints.goalkeepers,    # GK
            2: self.constraints.defenders,      # DEF
            3: self.constraints.midfielders,    # MID
            4: self.constraints.forwards        # FWD
        }
        
        for position, limit in position_limits.items():
            position_players = players_df[players_df['element_type'] == position]
            prob += lpSum([
                player_vars[player['id']] for _, player in position_players.iterrows()
            ]) == limit
    
    def _add_team_diversity_constraints(self, prob: LpProblem, players_df: pd.DataFrame, player_vars: Dict) -> None:
        """Add team diversity constraints."""
        # Max players per team
        for team_id in players_df['team'].unique():
            team_players = players_df[players_df['team'] == team_id]
            prob += lpSum([
                player_vars[player['id']] for _, player in team_players.iterrows()
            ]) <= self.constraints.max_players_per_team
    
    def _add_playing_xi_constraints(self, prob: LpProblem, squad_df: pd.DataFrame, playing_vars: Dict, captain_vars: Dict) -> None:
        """Add playing XI specific constraints."""
        # Exactly 11 players playing
        prob += lpSum([playing_vars[player['id']] for _, player in squad_df.iterrows()]) == 11
        
        # Exactly 1 captain
        prob += lpSum([captain_vars[player['id']] for _, player in squad_df.iterrows()]) == 1
        
        # Captain must be playing
        for _, player in squad_df.iterrows():
            prob += captain_vars[player['id']] <= playing_vars[player['id']]
        
        # Position constraints for playing XI
        position_constraints = {
            1: (1, 1),  # GK: exactly 1
            2: (3, 5),  # DEF: 3-5
            3: (2, 5),  # MID: 2-5  
            4: (1, 3),  # FWD: 1-3
        }
        
        for position, (min_play, max_play) in position_constraints.items():
            position_players = squad_df[squad_df['element_type'] == position]
            position_sum = lpSum([playing_vars[player['id']] for _, player in position_players.iterrows()])
            prob += position_sum >= min_play
            prob += position_sum <= max_play
    
    def _add_force_constraints(self, prob: LpProblem, player_vars: Dict, force_players: List[int]) -> None:
        """Force specific players to be selected."""
        for player_id in force_players:
            if player_id in player_vars:
                prob += player_vars[player_id] == 1
    
    def _add_exclude_constraints(self, prob: LpProblem, player_vars: Dict, exclude_players: List[int]) -> None:
        """Exclude specific players from selection."""
        for player_id in exclude_players:
            if player_id in player_vars:
                prob += player_vars[player_id] == 0
    
    def _extract_results(
        self, 
        prob: LpProblem, 
        players_df: pd.DataFrame, 
        player_vars: Dict,
        predicted_points: Dict[int, float],
        optimization_time: float
    ) -> OptimizedTeam:
        """Extract results from solved optimization problem."""
        if prob.status == LpStatusOptimal:
            selected_players = []
            total_cost = 0
            total_predicted_points = 0
            
            for _, player in players_df.iterrows():
                if player_vars[player['id']].varValue == 1:
                    selected_players.append(player['id'])
                    total_cost += player['price_millions']
                    total_predicted_points += predicted_points.get(player['id'], 0)
            
            # Determine formation
            selected_df = players_df[players_df['id'].isin(selected_players)]
            position_counts = selected_df['element_type'].value_counts()
            formation = f"{position_counts.get(2, 0)}-{position_counts.get(3, 0)}-{position_counts.get(4, 0)}"
            
            return OptimizedTeam(
                status="optimal",
                selected_players=selected_players,
                total_cost=total_cost,
                predicted_points=total_predicted_points,
                remaining_budget=self.constraints.max_budget - total_cost,
                formation=formation,
                optimization_time=optimization_time,
                iterations=None  # PuLP doesn't expose iteration count
            )
        
        elif prob.status == LpStatusInfeasible:
            return OptimizedTeam(
                status="infeasible",
                error_message="No feasible solution found - constraints may be too restrictive",
                optimization_time=optimization_time
            )
        
        elif prob.status == LpStatusNotSolved:
            return OptimizedTeam(
                status="timeout",
                error_message=f"Optimization timed out after {self.timeout} seconds",
                optimization_time=optimization_time
            )
        
        else:
            return OptimizedTeam(
                status="error",
                error_message=f"Optimization failed with status: {LpStatus[prob.status]}",
                optimization_time=optimization_time
            )
    
    def _find_best_single_transfer(
        self, 
        current_team: List[int], 
        players_df: pd.DataFrame, 
        predicted_points: Dict[int, float],
        is_free: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Find the best single transfer option."""
        best_transfer = None
        best_gain = 0 if is_free else self.constraints.transfer_cost  # Must beat transfer cost
        
        current_df = players_df[players_df['id'].isin(current_team)]
        available_df = players_df[~players_df['id'].isin(current_team)]
        
        for _, out_player in current_df.iterrows():
            out_points = predicted_points.get(out_player['id'], 0)
            out_cost = out_player.get('price_millions', out_player['now_cost'] / 10.0)
            
            # Find replacements in same position
            same_position = available_df[available_df['element_type'] == out_player['element_type']]
            
            for _, in_player in same_position.iterrows():
                in_points = predicted_points.get(in_player['id'], 0)
                in_cost = in_player.get('price_millions', in_player['now_cost'] / 10.0)
                
                # Check if transfer is affordable (simplified)
                cost_diff = in_cost - out_cost
                if cost_diff > 2.0:  # Assume max 2M budget flexibility
                    continue
                
                points_gain = in_points - out_points
                
                if points_gain > best_gain:
                    best_gain = points_gain
                    best_transfer = {
                        'player_out': out_player['id'],
                        'player_in': in_player['id'],
                        'points_gain': points_gain,
                        'cost_change': cost_diff,
                        'reasoning': [f"Expected {points_gain:.1f} point improvement"]
                    }
        
        return best_transfer
    
    def _convert_transfers_to_recommendations(
        self, 
        transfers: List[Dict[str, Any]], 
        players_df: pd.DataFrame
    ) -> List[TransferRecommendation]:
        """Convert transfer dictionaries to TransferRecommendation objects."""
        recommendations = []
        
        for transfer in transfers:
            try:
                out_player_data = players_df[players_df['id'] == transfer['player_out']].iloc[0]
                in_player_data = players_df[players_df['id'] == transfer['player_in']].iloc[0]
                
                # Create Player objects (simplified)
                out_player = Player(
                    id=int(out_player_data['id']),
                    first_name=str(out_player_data.get('first_name', '')),
                    second_name=str(out_player_data.get('second_name', '')),
                    web_name=str(out_player_data.get('web_name', '')),
                    element_type=int(out_player_data['element_type']),
                    team=int(out_player_data['team']),
                    now_cost=int(out_player_data.get('now_cost', 50))
                )
                
                in_player = Player(
                    id=int(in_player_data['id']),
                    first_name=str(in_player_data.get('first_name', '')),
                    second_name=str(in_player_data.get('second_name', '')),
                    web_name=str(in_player_data.get('web_name', '')),
                    element_type=int(in_player_data['element_type']),
                    team=int(in_player_data['team']),
                    now_cost=int(in_player_data.get('now_cost', 50))
                )
                
                recommendation = TransferRecommendation(
                    player_out=out_player,
                    player_in=in_player,
                    expected_points_gain=transfer['points_gain'],
                    confidence=0.7,  # Default confidence
                    cost_change=transfer['cost_change'],
                    reasoning=transfer['reasoning'],
                    weeks_analyzed=4,  # Default planning horizon
                    risk_level="Medium"
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"Failed to convert transfer to recommendation: {e}")
        
        return recommendations