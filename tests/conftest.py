"""
Pytest configuration and fixtures for FPL ML System tests.
Following TestModel patterns with comprehensive test data and mocking.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Import system components
from src.config.settings import FPLSettings
from src.models.data_models import Player, Team, Fixture, Event
from src.agents import (
    FPLManagerDependencies,
    DataPipelineDependencies, 
    MLPredictionDependencies,
    TransferAdvisorDependencies
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Create test configuration settings."""
    return FPLSettings(
        fpl_team_id=12345,
        database_url="sqlite:///:memory:",
        log_level="DEBUG",
        openai_api_key="test-key-12345",
        environment="test"
    )


@pytest.fixture
def sample_players_data():
    """Create comprehensive sample player data for testing."""
    players = []
    positions = [1, 2, 3, 4]  # GK, DEF, MID, FWD
    teams = list(range(1, 21))  # 20 teams
    
    for i in range(1, 101):  # 100 sample players
        position = positions[(i-1) % 4]
        team = teams[(i-1) % 20]
        
        # Create realistic player data
        base_price = {1: 45, 2: 45, 3: 55, 4: 65}[position]  # Base prices by position
        price_variation = np.random.randint(-15, 35)
        
        players.append({
            'id': i,
            'web_name': f'Player{i}',
            'first_name': f'First{i}',
            'second_name': f'Last{i}',
            'element_type': position,
            'team': team,
            'now_cost': base_price + price_variation,
            'total_points': np.random.randint(0, 200),
            'form': np.random.uniform(0, 10),
            'selected_by_percent': np.random.uniform(0.1, 80),
            'minutes': np.random.randint(0, 1500),
            'goals_scored': np.random.randint(0, 20),
            'assists': np.random.randint(0, 15),
            'clean_sheets': np.random.randint(0, 15),
            'saves': np.random.randint(0, 100) if position == 1 else 0,
            'bonus': np.random.randint(0, 30),
            'bps': np.random.randint(0, 500),
            'influence': np.random.uniform(0, 100),
            'creativity': np.random.uniform(0, 100),
            'threat': np.random.uniform(0, 100),
            'ict_index': np.random.uniform(0, 300),
            'yellow_cards': np.random.randint(0, 10),
            'red_cards': np.random.randint(0, 2),
            'news': '',
            'chance_of_playing_this_round': 100 if np.random.random() > 0.1 else np.random.randint(0, 75),
            'cost_change_start': np.random.randint(-10, 15),
            'dreamteam_count': np.random.randint(0, 5),
            'in_dreamteam': False
        })
    
    return pd.DataFrame(players)


@pytest.fixture
def sample_teams_data():
    """Create sample team data."""
    teams = []
    team_names = ['ARS', 'AVL', 'BHA', 'BUR', 'CHE', 'CRY', 'EVE', 'FUL', 'LIV', 'LUT',
                  'MCI', 'MUN', 'NEW', 'NOR', 'SHU', 'TOT', 'WHU', 'WOL', 'BRE', 'NOT']
    
    for i, name in enumerate(team_names, 1):
        teams.append({
            'id': i,
            'name': f'{name} FC',
            'short_name': name,
            'strength': np.random.randint(2, 5),
            'strength_overall_home': np.random.randint(2, 5),
            'strength_overall_away': np.random.randint(2, 5),
            'strength_attack_home': np.random.randint(2, 5),
            'strength_attack_away': np.random.randint(2, 5),
            'strength_defence_home': np.random.randint(2, 5),
            'strength_defence_away': np.random.randint(2, 5)
        })
    
    return pd.DataFrame(teams)


@pytest.fixture
def sample_fixtures_data():
    """Create sample fixture data."""
    fixtures = []
    
    for gw in range(1, 39):  # 38 gameweeks
        for match in range(10):  # 10 matches per gameweek
            home_team = (match * 2) + 1
            away_team = (match * 2) + 2
            
            # Ensure teams are within range
            if away_team > 20:
                continue
                
            fixtures.append({
                'id': (gw - 1) * 10 + match + 1,
                'gameweek': gw,
                'team_h': home_team,
                'team_a': away_team,
                'team_h_difficulty': np.random.randint(2, 5),
                'team_a_difficulty': np.random.randint(2, 5),
                'kickoff_time': datetime.now() + timedelta(days=gw*7, hours=match*2),
                'finished': gw <= 15,  # First 15 gameweeks finished
                'team_h_score': np.random.randint(0, 4) if gw <= 15 else None,
                'team_a_score': np.random.randint(0, 4) if gw <= 15 else None
            })
    
    return pd.DataFrame(fixtures)


@pytest.fixture
def sample_gameweek_history():
    """Create sample gameweek history data."""
    history = []
    
    for gw in range(1, 16):  # 15 completed gameweeks
        for player_id in range(1, 101):  # 100 players
            history.append({
                'player_id': player_id,
                'gameweek': gw,
                'total_points': np.random.randint(0, 15),
                'minutes': np.random.randint(0, 90),
                'goals_scored': np.random.randint(0, 3),
                'assists': np.random.randint(0, 2),
                'clean_sheets': np.random.randint(0, 1),
                'saves': np.random.randint(0, 10),
                'bonus': np.random.randint(0, 3),
                'bps': np.random.randint(0, 50),
                'opponent_team': np.random.randint(1, 21),
                'was_home': np.random.choice([True, False]),
                'expected_goals': np.random.uniform(0, 2),
                'expected_assists': np.random.uniform(0, 1),
                'expected_goal_involvements': np.random.uniform(0, 2.5)
            })
    
    return pd.DataFrame(history)


@pytest.fixture
def mock_fpl_api_response():
    """Mock FPL API response data."""
    return {
        'elements': [
            {
                'id': 1,
                'web_name': 'Salah',
                'first_name': 'Mohamed',
                'second_name': 'Salah',
                'element_type': 3,
                'team': 1,
                'now_cost': 130,
                'total_points': 187,
                'form': '8.5',
                'selected_by_percent': '54.2',
                'minutes': 1200,
                'goals_scored': 12,
                'assists': 8,
                'bonus': 18,
                'influence': '98.4',
                'creativity': '87.2',
                'threat': '156.8',
                'ict_index': '342.4'
            }
        ],
        'teams': [
            {
                'id': 1,
                'name': 'Liverpool',
                'short_name': 'LIV',
                'strength': 5,
                'strength_overall_home': 5,
                'strength_overall_away': 5
            }
        ],
        'events': [
            {
                'id': 16,
                'name': 'Gameweek 16',
                'is_current': True,
                'is_next': False,
                'finished': False,
                'deadline_time': '2024-12-16T11:30:00Z'
            }
        ]
    }


@pytest.fixture
def mock_team_picks():
    """Mock team picks response."""
    return {
        'picks': [
            {'element': i, 'position': i, 'multiplier': 2 if i == 1 else 1, 'is_captain': i == 1, 'is_vice_captain': i == 2}
            for i in range(1, 16)
        ],
        'entry_history': {
            'event': 16,
            'points': 67,
            'total_points': 1847,
            'rank': 125432,
            'rank_sort': 125432,
            'overall_rank': 125432,
            'event_transfers': 1,
            'event_transfers_cost': 0,
            'value': 985,
            'points_on_bench': 8
        }
    }


@pytest.fixture
def sample_ml_training_data(sample_gameweek_history):
    """Create ML training data with engineered features."""
    df = sample_gameweek_history.copy()
    
    # Add engineered features
    df = df.sort_values(['player_id', 'gameweek'])
    df['form_last_5'] = df.groupby('player_id')['total_points'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['minutes_last_5'] = df.groupby('player_id')['minutes'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['goals_per_90'] = (df['goals_scored'] / df['minutes'] * 90).fillna(0)
    df['assists_per_90'] = (df['assists'] / df['minutes'] * 90).fillna(0)
    df['fixture_difficulty'] = np.random.uniform(1.5, 4.5, len(df))
    df['is_home'] = df['was_home'].astype(int)
    df['team_strength'] = np.random.uniform(2, 5, len(df))
    df['opponent_weakness'] = np.random.uniform(2, 5, len(df))
    df['price_momentum'] = np.random.uniform(-0.2, 0.2, len(df))
    
    return df


# Agent fixtures with proper dependencies
@pytest.fixture
def fpl_manager_deps(test_settings):
    """Create FPL Manager dependencies for testing."""
    return FPLManagerDependencies(
        fpl_team_id=test_settings.fpl_team_id,
        database_url=str(test_settings.database_url),
        session_id="test-session"
    )


@pytest.fixture
def data_pipeline_deps(test_settings):
    """Create Data Pipeline dependencies for testing."""
    return DataPipelineDependencies(
        database_url=str(test_settings.database_url),
        cache_duration_minutes=30,
        batch_size=50,
        session_id="test-session"
    )


@pytest.fixture
def ml_prediction_deps():
    """Create ML Prediction dependencies for testing."""
    return MLPredictionDependencies(
        model_path="test_models",
        retrain_threshold_mse=0.005,
        min_training_samples=100,
        session_id="test-session"
    )


@pytest.fixture
def transfer_advisor_deps():
    """Create Transfer Advisor dependencies for testing."""
    return TransferAdvisorDependencies(
        optimization_timeout_seconds=10,
        min_transfer_gain_threshold=1.0,
        risk_tolerance="balanced",
        session_id="test-session"
    )


# Mock agents
@pytest.fixture
def mock_fpl_manager_agent():
    """Create mock FPL Manager agent."""
    agent = AsyncMock()
    agent.run = AsyncMock()
    return agent


@pytest.fixture
def mock_data_pipeline_agent():
    """Create mock Data Pipeline agent."""
    agent = AsyncMock()
    agent.run = AsyncMock()
    return agent


@pytest.fixture
def mock_ml_prediction_agent():
    """Create mock ML Prediction agent."""
    agent = AsyncMock()
    agent.run = AsyncMock()
    return agent


@pytest.fixture
def mock_transfer_advisor_agent():
    """Create mock Transfer Advisor agent."""
    agent = AsyncMock()
    agent.run = AsyncMock()
    return agent


# Performance benchmarks from PRP
@pytest.fixture
def performance_benchmarks():
    """Define performance benchmarks from PRP requirements."""
    return {
        'ml_models': {
            'mse_threshold': 0.003,  # Research benchmark: MSE < 0.003
            'correlation_threshold': 0.65,
            'accuracy_threshold': 0.60  # 60% accuracy within ±2 points
        },
        'optimization': {
            'max_solve_time_seconds': 5,  # PuLP optimization within 5 seconds
            'feasibility_rate': 0.95,  # 95% of problems should be feasible
            'improvement_threshold': 0.05  # 5% improvement over baseline
        },
        'api_performance': {
            'max_response_time_ms': 2000,
            'success_rate_threshold': 0.98,
            'cache_hit_rate': 0.90
        },
        'system_performance': {
            'agent_response_time_seconds': 10,
            'data_freshness_minutes': 60,
            'uptime_threshold': 0.999
        }
    }


# Test data validation helpers
@pytest.fixture
def validate_player_data():
    """Helper function to validate player data structure."""
    def _validate(player_data: Dict[str, Any]) -> bool:
        required_fields = ['id', 'web_name', 'element_type', 'team', 'now_cost', 'total_points']
        return all(field in player_data for field in required_fields)
    return _validate


@pytest.fixture
def validate_prediction_result():
    """Helper function to validate ML prediction results."""
    def _validate(prediction: Dict[str, Any], benchmarks: Dict[str, Any]) -> bool:
        if 'mse' in prediction:
            return prediction['mse'] < benchmarks['ml_models']['mse_threshold']
        return True
    return _validate


# Async test helpers
@pytest.fixture
def async_test_runner():
    """Helper to run async tests."""
    def _run_async(coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    return _run_async


# Database fixtures
@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    import sqlite3
    conn = sqlite3.connect(':memory:')
    
    # Create basic tables
    conn.execute('''
        CREATE TABLE players (
            id INTEGER PRIMARY KEY,
            web_name TEXT,
            element_type INTEGER,
            team INTEGER,
            now_cost INTEGER,
            total_points INTEGER
        )
    ''')
    
    conn.execute('''
        CREATE TABLE gameweek_history (
            id INTEGER PRIMARY KEY,
            player_id INTEGER,
            gameweek INTEGER,
            total_points INTEGER,
            minutes INTEGER,
            FOREIGN KEY (player_id) REFERENCES players (id)
        )
    ''')
    
    yield conn
    conn.close()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "ml: Machine learning model tests")
    config.addinivalue_line("markers", "api: API integration tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "benchmark: Benchmark validation tests")


# Custom assertions for TestModel pattern
class TestModelAssertions:
    """Custom assertions following TestModel patterns."""
    
    @staticmethod
    def assert_performance_benchmark(actual_value: float, benchmark: float, metric_name: str):
        """Assert that performance meets benchmark requirements."""
        assert actual_value <= benchmark, f"{metric_name} {actual_value} exceeds benchmark {benchmark}"
    
    @staticmethod
    def assert_ml_model_quality(mse: float, correlation: float, accuracy: float, benchmarks: Dict):
        """Assert ML model meets quality benchmarks."""
        ml_benchmarks = benchmarks['ml_models']
        assert mse < ml_benchmarks['mse_threshold'], f"MSE {mse} exceeds threshold {ml_benchmarks['mse_threshold']}"
        assert correlation > ml_benchmarks['correlation_threshold'], f"Correlation {correlation} below threshold"
        assert accuracy > ml_benchmarks['accuracy_threshold'], f"Accuracy {accuracy} below threshold"
    
    @staticmethod
    def assert_optimization_performance(solve_time: float, status: str, benchmarks: Dict):
        """Assert optimization meets performance requirements."""
        opt_benchmarks = benchmarks['optimization']
        assert solve_time <= opt_benchmarks['max_solve_time_seconds'], f"Solve time {solve_time}s exceeds {opt_benchmarks['max_solve_time_seconds']}s"
        assert status == "optimal", f"Optimization status {status} is not optimal"


@pytest.fixture
def test_assertions():
    """Provide TestModel assertion helpers."""
    return TestModelAssertions()