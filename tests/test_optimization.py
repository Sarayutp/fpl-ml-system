"""
Optimization Test Suite - Following TestModel patterns.
Tests for PuLP optimization engine with performance benchmarks.
Target: Optimization within 5 seconds, 95% feasibility rate.
"""

import pytest
import time
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.models.optimization import FPLOptimizer, WildcardOptimizer
from src.models.data_models import OptimizationResult, Player


@pytest.mark.optimization
class TestFPLOptimizer:
    """Test suite for FPL optimization engine following TestModel patterns."""
    
    def test_optimizer_initialization(self):
        """Test FPLOptimizer initializes correctly."""
        optimizer = FPLOptimizer()
        
        assert optimizer is not None
        assert hasattr(optimizer, 'position_limits')
        assert hasattr(optimizer, 'budget')
        assert hasattr(optimizer, 'max_players_per_team')
        
        # Verify FPL constraints
        assert optimizer.budget == 100.0
        assert optimizer.position_limits['GK']['min'] == 2
        assert optimizer.position_limits['DEF']['min'] == 5
        assert optimizer.position_limits['MID']['min'] == 5
        assert optimizer.position_limits['FWD']['min'] == 3
    
    @pytest.mark.benchmark
    def test_team_optimization_speed_benchmark(self, sample_players_data, performance_benchmarks, test_assertions):
        """CRITICAL TEST: Team optimization within 5 seconds benchmark."""
        optimizer = FPLOptimizer()
        
        # Create predicted points for all players
        predicted_points = {
            player_id: np.random.uniform(2, 12) 
            for player_id in sample_players_data['id']
        }
        
        # Ensure we have enough players per position for valid optimization
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        # Time the optimization
        start_time = time.time()
        result = optimizer.optimize_team(sample_players_data, predicted_points)
        solve_time = time.time() - start_time
        
        # CRITICAL BENCHMARK TEST
        test_assertions.assert_optimization_performance(
            solve_time,
            result.status,
            performance_benchmarks
        )
        
        # Verify optimization result quality
        if result.status == 'optimal':
            assert result.total_cost <= 100.0
            assert len(result.selected_players) == 15
            assert result.predicted_points > 0
    
    def test_position_constraints_validation(self, sample_players_data):
        """Test optimization respects FPL position constraints."""
        optimizer = FPLOptimizer()
        
        # Ensure balanced player distribution
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        # Create reasonable predicted points
        predicted_points = {
            player_id: np.random.uniform(4, 10) 
            for player_id in sample_players_data['id']
        }
        
        result = optimizer.optimize_team(sample_players_data, predicted_points)
        
        if result.status == 'optimal':
            selected_data = sample_players_data[sample_players_data['id'].isin(result.selected_players)]
            position_counts = selected_data['position'].value_counts()
            
            # Verify exact position requirements
            assert position_counts.get('GK', 0) == 2, f"Should have 2 GK, got {position_counts.get('GK', 0)}"
            assert position_counts.get('DEF', 0) == 5, f"Should have 5 DEF, got {position_counts.get('DEF', 0)}"
            assert position_counts.get('MID', 0) == 5, f"Should have 5 MID, got {position_counts.get('MID', 0)}"
            assert position_counts.get('FWD', 0) == 3, f"Should have 3 FWD, got {position_counts.get('FWD', 0)}"
    
    def test_budget_constraint_validation(self, sample_players_data):
        """Test optimization respects budget constraints."""
        optimizer = FPLOptimizer()
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        predicted_points = {
            player_id: np.random.uniform(5, 9) 
            for player_id in sample_players_data['id']
        }
        
        result = optimizer.optimize_team(sample_players_data, predicted_points)
        
        if result.status == 'optimal':
            assert result.total_cost <= optimizer.budget, \
                f"Total cost {result.total_cost} exceeds budget {optimizer.budget}"
            assert result.remaining_budget >= 0, \
                f"Remaining budget {result.remaining_budget} is negative"
    
    def test_team_constraint_validation(self, sample_players_data):
        """Test optimization respects max players per team constraint."""
        optimizer = FPLOptimizer()
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        predicted_points = {
            player_id: np.random.uniform(4, 8) 
            for player_id in sample_players_data['id']
        }
        
        result = optimizer.optimize_team(sample_players_data, predicted_points)
        
        if result.status == 'optimal':
            selected_data = sample_players_data[sample_players_data['id'].isin(result.selected_players)]
            team_counts = selected_data['team'].value_counts()
            
            # No team should have more than 3 players
            max_players_from_team = team_counts.max()
            assert max_players_from_team <= optimizer.max_players_per_team, \
                f"Team has {max_players_from_team} players, max allowed is {optimizer.max_players_per_team}"
    
    @pytest.mark.performance
    def test_large_dataset_optimization(self, performance_benchmarks):
        """Test optimization performance with large dataset."""
        optimizer = FPLOptimizer()
        
        # Create large realistic dataset (600+ players)
        large_dataset = []
        for i in range(1, 601):
            position = ((i - 1) % 4) + 1  # Cycle through positions
            team = ((i - 1) % 20) + 1     # Cycle through teams
            
            large_dataset.append({
                'id': i,
                'element_type': position,
                'team': team,
                'now_cost': np.random.randint(40, 130),
                'total_points': np.random.randint(10, 200),
                'minutes': np.random.randint(500, 1800)
            })
        
        large_df = pd.DataFrame(large_dataset)
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        large_df['position'] = large_df['element_type'].map(position_mapping)
        
        predicted_points = {i: np.random.uniform(3, 11) for i in range(1, 601)}
        
        # Time optimization with large dataset
        start_time = time.time()
        result = optimizer.optimize_team(large_df, predicted_points)
        solve_time = time.time() - start_time
        
        # Should still meet performance benchmark
        max_solve_time = performance_benchmarks['optimization']['max_solve_time_seconds']
        assert solve_time <= max_solve_time, \
            f"Large dataset optimization took {solve_time:.2f}s, should be under {max_solve_time}s"
        
        assert result.status == 'optimal', f"Large dataset optimization failed: {result.status}"
    
    def test_transfer_optimization_logic(self, sample_players_data):
        """Test transfer optimization finds beneficial moves."""
        optimizer = FPLOptimizer()
        
        # Current team (first 15 players)
        current_team = list(range(1, 16))
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        # Create predictions where some non-owned players are better
        predicted_points = {}
        for i in range(1, 101):
            if i in current_team:
                predicted_points[i] = np.random.uniform(4, 7)  # Lower for current team
            else:
                predicted_points[i] = np.random.uniform(6, 9)  # Higher for others
        
        result = optimizer.optimize_transfers(
            current_team=current_team,
            available_players=sample_players_data,
            predicted_points=predicted_points,
            free_transfers=1,
            weeks_ahead=3
        )
        
        if result.status == 'optimal' and result.recommended_transfers:
            # Should find beneficial transfers
            assert result.expected_points_gain > 0, \
                "Should find transfers with positive expected gain"
            
            for transfer in result.recommended_transfers:
                assert transfer['player_out'] in current_team, \
                    "Transfer out player should be in current team"
                assert transfer['player_in'] not in current_team, \
                    "Transfer in player should not be in current team"
    
    def test_optimization_with_custom_constraints(self, sample_players_data):
        """Test optimization with additional custom constraints."""
        optimizer = FPLOptimizer()
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        predicted_points = {
            player_id: np.random.uniform(4, 8) 
            for player_id in sample_players_data['id']
        }
        
        # Custom constraints: must include specific players
        constraints = {
            'must_include': [1, 2],  # Must include players 1 and 2
            'exclude': [99, 100]     # Must exclude players 99 and 100
        }
        
        result = optimizer.optimize_team(
            sample_players_data, 
            predicted_points,
            constraints=constraints
        )
        
        if result.status == 'optimal':
            # Verify custom constraints are respected
            assert 1 in result.selected_players, "Must include constraint violated"
            assert 2 in result.selected_players, "Must include constraint violated"
            assert 99 not in result.selected_players, "Exclude constraint violated"
            assert 100 not in result.selected_players, "Exclude constraint violated"
    
    def test_infeasible_optimization_handling(self):
        """Test handling of infeasible optimization problems."""
        optimizer = FPLOptimizer()
        
        # Create impossible scenario (all players too expensive)
        impossible_data = pd.DataFrame({
            'id': range(1, 16),
            'element_type': [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 1],
            'team': range(1, 16),
            'now_cost': [150] * 15,  # All players cost 15.0M (impossible budget)
            'position': ['GK', 'DEF', 'DEF', 'DEF', 'DEF', 'DEF', 'MID', 'MID', 'MID', 'MID', 'MID', 'FWD', 'FWD', 'FWD', 'GK']
        })
        
        predicted_points = {i: 8.0 for i in range(1, 16)}
        
        result = optimizer.optimize_team(impossible_data, predicted_points)
        
        # Should detect infeasibility
        assert result.status == 'infeasible', \
            f"Should detect infeasible problem, got status: {result.status}"
        assert result.error is not None, "Should provide error message for infeasible problem"


@pytest.mark.optimization
class TestWildcardOptimizer:
    """Test suite for Wildcard optimization."""
    
    def test_wildcard_optimizer_initialization(self):
        """Test WildcardOptimizer initializes correctly."""
        optimizer = WildcardOptimizer()
        
        assert optimizer is not None
        assert hasattr(optimizer, 'base_optimizer')
        assert isinstance(optimizer.base_optimizer, FPLOptimizer)
    
    def test_optimal_wildcard_timing(self, sample_fixtures_data):
        """Test finding optimal wildcard timing."""
        optimizer = WildcardOptimizer()
        
        with patch('src.models.optimization.FPLOptimizer.optimize_team') as mock_opt:
            mock_opt.return_value = Mock(
                status='optimal',
                predicted_points=85.2,
                total_cost=99.5
            )
            
            result = optimizer.find_optimal_timing(
                fixtures=sample_fixtures_data,
                weeks_ahead=8
            )
            
            assert isinstance(result, dict)
            assert 'optimal_gameweek' in result
            assert 'expected_benefit' in result
            assert 'confidence' in result
            assert 1 <= result['optimal_gameweek'] <= 38
    
    def test_wildcard_team_generation(self, sample_players_data):
        """Test generating optimal wildcard team."""
        optimizer = WildcardOptimizer()
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        predicted_points = {
            player_id: np.random.uniform(5, 10) 
            for player_id in sample_players_data['id']
        }
        
        wildcard_team = optimizer.generate_optimal_team(
            players=sample_players_data,
            predicted_points=predicted_points,
            gameweek=16
        )
        
        assert wildcard_team.status == 'optimal'
        assert len(wildcard_team.selected_players) == 15
        assert wildcard_team.total_cost <= 100.0


@pytest.mark.benchmark
class TestOptimizationBenchmarks:
    """Comprehensive optimization benchmark tests following PRP requirements."""
    
    def test_feasibility_rate_benchmark(self, performance_benchmarks):
        """Test optimization achieves required feasibility rate."""
        optimizer = FPLOptimizer()
        
        successful_optimizations = 0
        total_attempts = 20
        
        for attempt in range(total_attempts):
            # Generate random but realistic optimization problems
            test_data = []
            for i in range(1, 101):
                position = ((i - 1) % 4) + 1
                team = ((i - 1) % 20) + 1
                
                test_data.append({
                    'id': i,
                    'element_type': position,
                    'team': team,
                    'now_cost': np.random.randint(40, 120),  # Realistic price range
                    'total_points': np.random.randint(20, 180)
                })
            
            test_df = pd.DataFrame(test_data)
            position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            test_df['position'] = test_df['element_type'].map(position_mapping)
            
            predicted_points = {i: np.random.uniform(4, 9) for i in range(1, 101)}
            
            try:
                result = optimizer.optimize_team(test_df, predicted_points)
                if result.status == 'optimal':
                    successful_optimizations += 1
            except Exception:
                # Count exceptions as failures
                pass
        
        feasibility_rate = successful_optimizations / total_attempts
        required_rate = performance_benchmarks['optimization']['feasibility_rate']
        
        assert feasibility_rate >= required_rate, \
            f"Feasibility rate {feasibility_rate:.2%} below benchmark {required_rate:.0%}"
    
    @pytest.mark.slow 
    def test_optimization_consistency(self, sample_players_data):
        """Test optimization produces consistent results."""
        optimizer = FPLOptimizer()
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        # Fixed predicted points for consistency
        np.random.seed(42)
        predicted_points = {
            player_id: np.random.uniform(4, 8) 
            for player_id in sample_players_data['id']
        }
        
        # Run optimization multiple times
        results = []
        for _ in range(5):
            result = optimizer.optimize_team(sample_players_data, predicted_points)
            results.append(result)
        
        # All results should be identical (deterministic)
        if all(r.status == 'optimal' for r in results):
            base_result = results[0]
            for result in results[1:]:
                assert result.total_cost == base_result.total_cost, \
                    "Optimization should be deterministic"
                assert set(result.selected_players) == set(base_result.selected_players), \
                    "Selected players should be identical across runs"
    
    def test_optimization_improvement_quality(self, sample_players_data, performance_benchmarks):
        """Test optimization provides meaningful improvements."""
        optimizer = FPLOptimizer()
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        # Create scenario where optimization can provide clear improvement
        predicted_points = {}
        for i, row in sample_players_data.iterrows():
            # Make some players clearly better than others
            if row['id'] <= 50:
                predicted_points[row['id']] = np.random.uniform(7, 10)  # Good players
            else:
                predicted_points[row['id']] = np.random.uniform(3, 6)   # Worse players
        
        result = optimizer.optimize_team(sample_players_data, predicted_points)
        
        if result.status == 'optimal':
            # Calculate baseline (random team selection)
            random_team = sample_players_data.sample(15)['id'].tolist()
            baseline_points = sum(predicted_points[pid] for pid in random_team)
            
            improvement = (result.predicted_points - baseline_points) / baseline_points
            min_improvement = performance_benchmarks['optimization']['improvement_threshold']
            
            assert improvement >= min_improvement, \
                f"Optimization improvement {improvement:.1%} below threshold {min_improvement:.0%}"
    
    @pytest.mark.performance
    def test_memory_efficiency_during_optimization(self, sample_players_data):
        """Test optimization memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        optimizer = FPLOptimizer()
        
        # Add position mapping
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        sample_players_data['position'] = sample_players_data['element_type'].map(position_mapping)
        
        predicted_points = {
            player_id: np.random.uniform(4, 8) 
            for player_id in sample_players_data['id']
        }
        
        # Run multiple optimizations
        for _ in range(10):
            result = optimizer.optimize_team(sample_players_data, predicted_points)
            assert result is not None
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (under 30MB for 10 optimizations)
        assert memory_increase < 30, \
            f"Optimization used {memory_increase:.1f}MB, should be under 30MB"