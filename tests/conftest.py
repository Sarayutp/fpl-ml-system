"""
Pytest configuration and fixtures for FPL ML System tests.
Following TestModel patterns with comprehensive test data and mocking.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Import pandas and numpy with error handling for testing
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Warning: Failed to import pandas/numpy: {e}")
    # Create mock replacements for testing
    pd = Mock()
    np = Mock()

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


# Enhanced multi-season test data fixtures
@pytest.fixture
def comprehensive_test_data():
    """Multi-season FPL data for thorough testing."""
    seasons = ['2021-22', '2022-23', '2023-24']
    all_season_data = []
    
    for season_idx, season in enumerate(seasons):
        # Generate realistic data for each season
        season_data = []
        
        for gw in range(1, 39):  # 38 gameweeks
            for player_id in range(1, 101):  # 100 players per season
                # Create realistic variation across seasons
                base_points = np.random.poisson(5)
                season_modifier = 1.0 + (season_idx * 0.1)  # Players improve over seasons
                
                season_data.append({
                    'season': season,
                    'player_id': player_id + (season_idx * 1000),  # Unique IDs per season
                    'gameweek': gw,
                    'total_points': max(0, int(base_points * season_modifier)),
                    'minutes': np.random.randint(0, 90),
                    'goals_scored': np.random.poisson(0.3),
                    'assists': np.random.poisson(0.2),
                    'clean_sheets': np.random.binomial(1, 0.3),
                    'saves': np.random.poisson(2) if (player_id % 4) == 1 else 0,  # GK only
                    'bonus': np.random.randint(0, 4),
                    'bps': np.random.randint(0, 50),
                    'opponent_team': np.random.randint(1, 21),
                    'was_home': np.random.choice([True, False]),
                    'expected_goals': np.random.exponential(0.5),
                    'expected_assists': np.random.exponential(0.3),
                    'fixture_difficulty': np.random.randint(2, 6)
                })
        
        all_season_data.extend(season_data)
    
    return pd.DataFrame(all_season_data)


@pytest.fixture
def ml_benchmark_data():
    """Curated data for ML benchmark validation with target performance."""
    # Generate high-quality training data that should meet research benchmarks
    np.random.seed(42)  # Reproducible benchmark data
    
    data = []
    for i in range(2000):  # Sufficient data for reliable benchmarks
        # Create correlated features for better model performance
        form_last_5 = np.random.uniform(0, 10)
        minutes_last_5 = np.random.uniform(30, 90)
        goals_per_90 = np.random.exponential(0.5)
        assists_per_90 = np.random.exponential(0.3)
        
        # Target variable correlated with features for benchmark achievement
        total_points = (
            form_last_5 * 0.8 + 
            (minutes_last_5 / 90) * 3 + 
            goals_per_90 * 4 + 
            assists_per_90 * 3 + 
            np.random.normal(0, 1)  # Some noise
        )
        total_points = max(0, min(20, total_points))  # Realistic range
        
        data.append({
            'player_id': i,
            'gameweek': np.random.randint(1, 39),
            'form_last_5': form_last_5,
            'minutes_last_5': minutes_last_5,
            'goals_per_90': goals_per_90,
            'assists_per_90': assists_per_90,
            'fixture_difficulty': np.random.uniform(2, 5),
            'is_home': np.random.choice([0, 1]),
            'team_strength': np.random.uniform(2, 5),
            'opponent_weakness': np.random.uniform(2, 5),
            'price_momentum': np.random.uniform(-0.2, 0.2),
            'total_points': total_points
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def performance_test_scenarios():
    """Load testing scenarios for performance validation."""
    return {
        'concurrent_users': 50,
        'operations_per_user': 10,
        'agent_operations': [
            'get_current_team',
            'optimize_team',
            'get_transfer_advice',
            'predict_player_points',
            'analyze_captain_options'
        ],
        'test_scenarios': [
            {
                'name': 'light_load',
                'concurrent_requests': 10,
                'duration_seconds': 30
            },
            {
                'name': 'normal_load', 
                'concurrent_requests': 25,
                'duration_seconds': 60
            },
            {
                'name': 'heavy_load',
                'concurrent_requests': 50,
                'duration_seconds': 120
            }
        ]
    }


@pytest.fixture
def historical_seasons_data():
    """Historical FPL data spanning multiple seasons for backtesting."""
    seasons = {
        '2020-21': {'players': 700, 'gameweeks': 38, 'avg_points': 4.8},
        '2021-22': {'players': 720, 'gameweeks': 38, 'avg_points': 5.1},
        '2022-23': {'players': 715, 'gameweeks': 38, 'avg_points': 4.9},
        '2023-24': {'players': 730, 'gameweeks': 38, 'avg_points': 5.2}
    }
    
    historical_data = {}
    
    for season, meta in seasons.items():
        season_records = []
        
        for player_id in range(1, meta['players'] + 1):
            for gw in range(1, meta['gameweeks'] + 1):
                # Generate realistic historical patterns
                base_points = np.random.poisson(meta['avg_points'])
                
                season_records.append({
                    'season': season,
                    'player_id': player_id,
                    'gameweek': gw,
                    'total_points': max(0, base_points),
                    'minutes': np.random.choice([0, 90], p=[0.1, 0.9]),
                    'goals_scored': np.random.poisson(0.25),
                    'assists': np.random.poisson(0.15),
                    'clean_sheets': np.random.binomial(1, 0.25),
                    'bonus': np.random.randint(0, 4),
                    'was_home': np.random.choice([True, False]),
                    'opponent_team': np.random.randint(1, 21)
                })
        
        historical_data[season] = pd.DataFrame(season_records)
    
    return historical_data


@pytest.fixture
def mock_api_responses_comprehensive():
    """Comprehensive FPL API mock responses including error scenarios."""
    return {
        'bootstrap_static_success': {
            'elements': [
                {
                    'id': i,
                    'web_name': f'Player{i}',
                    'element_type': ((i-1) % 4) + 1,
                    'team': ((i-1) % 20) + 1,
                    'now_cost': np.random.randint(40, 150),
                    'total_points': np.random.randint(0, 200),
                    'form': f"{np.random.uniform(0, 10):.1f}",
                    'selected_by_percent': f"{np.random.uniform(0.1, 50):.1f}",
                    'minutes': np.random.randint(0, 1500),
                    'goals_scored': np.random.randint(0, 25),
                    'assists': np.random.randint(0, 20)
                }
                for i in range(1, 101)
            ],
            'teams': [
                {
                    'id': i,
                    'name': f'Team {i}',
                    'short_name': f'T{i:02d}',
                    'strength': np.random.randint(2, 6)
                }
                for i in range(1, 21)
            ],
            'events': [
                {
                    'id': i,
                    'name': f'Gameweek {i}',
                    'is_current': i == 16,
                    'is_next': i == 17,
                    'finished': i < 16
                }
                for i in range(1, 39)
            ]
        },
        'bootstrap_static_error': {
            'error': 'Service temporarily unavailable',
            'status_code': 503
        },
        'rate_limit_error': {
            'error': 'Rate limit exceeded',
            'status_code': 429,
            'retry_after': 60
        },
        'team_picks_success': {
            'picks': [
                {
                    'element': i,
                    'position': i,
                    'multiplier': 2 if i == 1 else 1,
                    'is_captain': i == 1,
                    'is_vice_captain': i == 2
                }
                for i in range(1, 16)
            ],
            'entry_history': {
                'event': 16,
                'points': 67,
                'total_points': 1847,
                'rank': 125432,
                'event_transfers': 1,
                'event_transfers_cost': 0,
                'value': 985
            }
        }
    }


# Enhanced TestModel fixtures for PydanticAI testing
@pytest.fixture
def test_model_configurations():
    """Various TestModel configurations for comprehensive agent testing."""
    from pydantic_ai.models.test import TestModel, FunctionModel
    
    def create_success_response(agent_type: str = "generic"):
        """Generate successful agent response."""
        responses = {
            "fpl_manager": '{"analysis": "Team analysis complete", "recommendations": ["Consider Salah transfer"], "confidence": 0.85}',
            "ml_prediction": '{"predictions": {"player_1": 8.5, "player_2": 6.2}, "model_confidence": 0.78}',
            "transfer_advisor": '{"transfers": [{"out": "Wilson", "in": "Haaland", "gain": 2.3}], "total_cost": 4}',
            "data_pipeline": '{"status": "success", "records_processed": 1247, "data_quality": 0.96}'
        }
        return responses.get(agent_type, '{"status": "success", "message": "Operation completed"}')
    
    def create_error_response(error_type: str = "generic"):
        """Generate error response for testing error handling."""
        errors = {
            "api_timeout": '{"error": "API timeout", "retry_suggested": true, "status": "error"}',
            "data_validation": '{"error": "Invalid data format", "details": "Missing required fields", "status": "error"}',
            "ml_model_error": '{"error": "Model prediction failed", "model_status": "unavailable", "status": "error"}',
            "optimization_failed": '{"error": "Optimization infeasible", "constraints_violated": ["budget"], "status": "error"}'
        }
        return errors.get(error_type, '{"error": "Unknown error occurred", "status": "error"}')
    
    return {
        'success_models': {
            agent_type: TestModel(custom_output_text=create_success_response(agent_type))
            for agent_type in ['fpl_manager', 'ml_prediction', 'transfer_advisor', 'data_pipeline']
        },
        'error_models': {
            error_type: TestModel(custom_output_text=create_error_response(error_type))
            for error_type in ['api_timeout', 'data_validation', 'ml_model_error', 'optimization_failed']
        },
        'function_models': {
            'conditional': FunctionModel(
                function=lambda messages, tools: create_success_response() if 'success' in str(messages[-1].content).lower() else create_error_response()
            ),
            'tool_calling': FunctionModel(
                function=lambda messages, tools: '{"tool_calls": [' + ', '.join([f'"{{\\"tool\\": \\"{tool.name}\\", \\"called\\": true}}"' for tool in tools]) + '], "status": "success"}'
            )
        }
    }


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


# Enhanced pytest configuration
def pytest_configure(config):
    """Configure pytest markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interaction")
    config.addinivalue_line("markers", "e2e: End-to-end tests for complete workflows")
    config.addinivalue_line("markers", "ml: Machine learning model tests")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "slow: Slow-running tests (>30s)")
    config.addinivalue_line("markers", "benchmark: Research benchmark validation tests")
    config.addinivalue_line("markers", "api: API integration tests")
    config.addinivalue_line("markers", "agent: PydanticAI agent tests")
    config.addinivalue_line("markers", "load: Load and stress testing")
    
    # Set test discovery patterns
    config.option.python_files = ['test_*.py', '*_test.py']
    config.option.python_classes = ['Test*', '*Test', '*Tests']
    config.option.python_functions = ['test_*']


# Memory and resource monitoring fixtures
@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    class MemoryMonitor:
        def __init__(self):
            self.start_memory = None
            self.peak_memory = None
            
        def start(self):
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
            
        def check(self):
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            return current_memory
            
        def get_usage(self):
            current_memory = self.check()
            return {
                'start_mb': self.start_memory,
                'current_mb': current_memory,
                'peak_mb': self.peak_memory,
                'increase_mb': current_memory - self.start_memory if self.start_memory else 0
            }
    
    return MemoryMonitor()


@pytest.fixture
def performance_timer():
    """High-precision timer for performance testing."""
    import time
    
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        def get_duration(self):
            if self.start_time is None:
                return 0
            end = self.end_time if self.end_time else time.perf_counter()
            return end - self.start_time
            
        def assert_duration_under(self, max_seconds: float, operation_name: str = "Operation"):
            duration = self.get_duration()
            assert duration <= max_seconds, f"{operation_name} took {duration:.3f}s, should be under {max_seconds}s"
    
    return PerformanceTimer()