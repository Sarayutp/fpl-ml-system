# FPL ML System - Implementation Guide

This guide provides implementation patterns and standards for building an AI-powered Fantasy Premier League management system using Python, Pydantic AI agents, and machine learning. For WHAT to build, see the INITIAL.md (Initial Requirements Document).

## Core Principles

**IMPORTANT: You MUST follow these principles in all code changes and requirements generation:**

### KISS (Keep It Simple, Stupid)

- Simplicity should be a key goal in design
- Choose proven ML algorithms over experimental ones whenever possible
- Simple solutions are easier to understand, maintain, and debug
- Start with basic models and iterate rather than building complex systems upfront

### YAGNI (You Aren't Gonna Need It)

- Avoid building ML features on speculation
- Implement prediction models only when they are needed for specific FPL decisions
- Don't over-engineer optimization algorithms until proven necessary

### Open/Closed Principle

- Design agent system so that new prediction models can be added with minimal changes to existing code
- ML pipeline should be extensible for new data sources and features
- Optimization engine should support new constraint types without core modifications

## Package Management & Tooling

**CRITICAL: This project uses pip for Python package management and modern Python development practices.**

### Essential pip Commands

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Add a dependency and update requirements
pip install package-name
pip freeze > requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Remove a package
pip uninstall package-name

# Update dependencies
pip install --upgrade package-name
pip list --outdated
```

### Essential Development Commands

**CRITICAL: Use these commands for all development, testing, and deployment.**

```bash
# Code Quality
black src/                    # Format code
isort src/                    # Sort imports
flake8 src/                   # Lint code
mypy src/                     # Type checking

# Testing
pytest                        # Run all tests
pytest tests/test_agents/     # Run specific test directory
pytest -v                     # Verbose output
pytest --cov=src             # Coverage report

# ML Model Training & Validation
python scripts/train_models.py          # Train all ML models
python scripts/validate_models.py       # Validate model performance
python scripts/backtest.py --seasons 3  # Backtest strategies

# FPL Data Management
python scripts/update_data.py           # Update FPL data
python scripts/setup_database.py       # Initialize database
python scripts/health_check.py         # Check system health
```

## Project Architecture

**IMPORTANT: This is a Pydantic AI agent-based FPL management system with ML prediction pipeline and optimization engine.**

### Current Project Structure

```
fpl-ml-system/
├── src/                              # Python source code
│   ├── __init__.py
│   ├── agents/                       # Pydantic AI agents
│   │   ├── __init__.py
│   │   ├── base.py                   # Base agent class
│   │   ├── fpl_manager.py           # Main orchestrator agent
│   │   ├── data_pipeline.py         # Data fetching & processing agent
│   │   ├── ml_prediction.py         # ML model agent
│   │   ├── transfer_advisor.py      # Transfer optimization agent
│   │   ├── dashboard.py             # Dashboard generation agent
│   │   └── notification.py          # Alert & notification agent
│   ├── models/                       # ML models and data models
│   │   ├── __init__.py
│   │   ├── data_models.py           # Pydantic data models
│   │   ├── ml_models.py             # ML prediction models
│   │   └── optimization.py          # Optimization algorithms
│   ├── data/                         # Data pipeline
│   │   ├── __init__.py
│   │   ├── fetchers.py              # API and web scraping
│   │   ├── processors.py            # Data cleaning & feature engineering
│   │   └── validators.py            # Data validation
│   ├── cli/                          # Command line interface
│   │   ├── __init__.py
│   │   ├── main.py                  # CLI entry point
│   │   ├── team.py                  # Team management commands
│   │   ├── transfer.py              # Transfer commands
│   │   ├── player.py                # Player analysis commands
│   │   └── utils.py                 # CLI utilities
│   ├── dashboard/                    # Streamlit web interface
│   │   ├── __init__.py
│   │   ├── app.py                   # Main dashboard app
│   │   ├── components/              # Reusable components
│   │   └── pages/                   # Dashboard pages
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py              # Application settings
│   │   └── database.py              # Database configuration
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── logging.py               # Logging configuration
│       ├── cache.py                 # Caching utilities
│       └── helpers.py               # General helpers
├── data/                             # Data storage
│   ├── raw/                         # Raw FPL data
│   ├── processed/                   # Cleaned data
│   └── models/                      # Trained ML models
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_agents/                 # Agent tests
│   ├── test_models/                 # ML model tests
│   ├── test_data/                   # Data pipeline tests
│   └── fixtures/                    # Test fixtures
├── notebooks/                        # Jupyter notebooks
│   ├── exploration/                 # Data exploration
│   ├── modeling/                    # Model development
│   └── analysis/                    # Performance analysis
├── scripts/                          # Utility scripts
│   ├── setup_database.py           # Database initialization
│   ├── train_models.py             # Model training
│   ├── update_data.py              # Data updates
│   └── deploy.py                   # Deployment scripts
├── docs/                            # Documentation
│   ├── setup.md                    # Setup instructions
│   ├── usage.md                    # Usage guide
│   └── api.md                      # API documentation
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── pyproject.toml                  # Project configuration
└── CLAUDE.md                       # This implementation guide
```

### Key File Purposes (ALWAYS ADD NEW FILES HERE)

**Main Implementation Files:**

- `src/agents/fpl_manager.py` - Primary orchestrator agent that coordinates all FPL operations
- `src/agents/ml_prediction.py` - ML-powered prediction agent for player points, prices, minutes

**Core ML Pipeline:**

- `src/models/ml_models.py` - XGBoost, Random Forest, LSTM models for predictions
- `src/models/optimization.py` - PuLP-based team optimization and transfer planning
- `src/data/fetchers.py` - FPL API client, web scraping for additional data sources

**User Interfaces:**

- `src/cli/main.py` - Click-based CLI for power users and automation
- `src/dashboard/app.py` - Streamlit web dashboard for interactive analysis

**Configuration Files:**

- `pyproject.toml` - Modern Python project configuration with tool settings
- `requirements.txt` - Production dependencies (pandas, scikit-learn, xgboost, pydantic-ai)
- `.env.example` - Environment variables template for FPL credentials and API keys

## Development Commands

### Core Workflow Commands

```bash
# Setup & Dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Development
python -m src.cli.main team show        # CLI development
streamlit run src/dashboard/app.py      # Dashboard development
python -m src.agents.fpl_manager        # Direct agent testing

# Code Quality & Type Checking
black src/ tests/                       # Format code
isort src/ tests/                       # Sort imports  
flake8 src/ tests/                      # Lint code
mypy src/                               # Type checking
pytest --cov=src --cov-report=html     # Test with coverage

# ML Pipeline
python scripts/train_models.py --retrain-all    # Retrain all ML models
python scripts/validate_models.py --benchmark   # Validate against benchmarks
python scripts/backtest.py --start-season 2021  # Historical backtesting
```

### Environment Configuration

**Environment Variables Setup:**

```bash
# Create .env file for local development based on .env.example
cp .env.example .env

# Edit .env with your credentials
FPL_TEAM_ID=your_team_id_here
FPL_EMAIL=your_fpl_email@example.com
FPL_PASSWORD=your_fpl_password
DATABASE_URL=sqlite:///data/fpl.db
LOG_LEVEL=INFO
OPENAI_API_KEY=your_openai_key_for_agents  # For Pydantic AI
```

## Pydantic AI Development Context

**IMPORTANT: This project builds production-ready FPL management agents using Pydantic AI framework with comprehensive ML pipeline.**

### Agent Technology Stack

**Core Technologies:**

- **pydantic-ai** - Modern AI agent framework with type safety
- **pydantic** - Data validation and settings management
- **scikit-learn** - ML algorithms and preprocessing
- **xgboost** - Gradient boosting for player predictions
- **pulp** - Linear programming for team optimization

**Data & ML Stack:**

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **requests** - HTTP client for FPL API
- **beautifulsoup4** - Web scraping for additional data
- **sqlalchemy** - Database ORM

### Agent Architecture

This project implements a multi-agent system with specialized agents:

**1. FPL Manager Agent (`src/agents/fpl_manager.py`):**

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class FPLManagerAgent(Agent):
    """Main orchestrator agent for FPL operations"""
    
    def __init__(self):
        super().__init__(
            'openai:gpt-4',  # or other LLM
            system_prompt="""You are an expert FPL manager with access to 
            ML predictions, optimization tools, and comprehensive data analysis."""
        )
        
        # Tools available to this agent
        self.tools = [
            self.get_team_analysis,
            self.suggest_transfers,
            self.predict_player_points,
            self.optimize_team,
            self.analyze_captain_options
        ]
    
    @tool
    async def get_team_analysis(self, team_id: int) -> TeamAnalysis:
        """Get comprehensive analysis of current team"""
        # Coordinate with other agents
        return await self.data_agent.analyze_team(team_id)
```

**2. ML Prediction Agent (`src/agents/ml_prediction.py`):**

```python
from pydantic_ai import Agent
from src.models.ml_models import PlayerPredictor, PricePredictor

class MLPredictionAgent(Agent):
    """Specialized agent for ML-powered predictions"""
    
    def __init__(self):
        super().__init__(
            'openai:gpt-4-turbo',
            system_prompt="""You are an ML expert specializing in FPL predictions.
            You can predict player points, price changes, and minutes played."""
        )
        
        self.player_predictor = PlayerPredictor()
        self.price_predictor = PricePredictor()
    
    @tool
    async def predict_player_points(self, player_id: int, gameweeks: int = 3) -> PlayerPrediction:
        """Predict player points for upcoming gameweeks"""
        features = await self.prepare_features(player_id)
        prediction = self.player_predictor.predict(features, gameweeks)
        
        return PlayerPrediction(
            player_id=player_id,
            expected_points=prediction.expected_points,
            confidence_interval=prediction.confidence_interval,
            reasoning=prediction.feature_importance
        )
```

### Agent Development Commands

**Local Development & Testing:**

```bash
# Test individual agents
python -c "from src.agents.fpl_manager import FPLManagerAgent; agent = FPLManagerAgent(); print(agent.test_connection())"

# Run agent with CLI
python -m src.cli.main agent chat --agent fpl_manager

# Test ML predictions
python -m src.agents.ml_prediction --test-predictions

# Run full system integration test
python -m tests.integration.test_full_system
```

### Agent Integration Patterns

**Agent Communication:**

```python
# Primary agent coordinates with specialized agents
class FPLManagerAgent(Agent):
    def __init__(self):
        super().__init__()
        
        # Initialize sub-agents
        self.data_agent = DataPipelineAgent()
        self.ml_agent = MLPredictionAgent()
        self.transfer_agent = TransferAdvisorAgent()
        self.dashboard_agent = DashboardAgent()
    
    @tool
    async def get_transfer_advice(self, weeks_ahead: int = 4) -> TransferAdvice:
        """Get comprehensive transfer advice using all agents"""
        
        # 1. Get current data
        current_data = await self.data_agent.get_latest_data()
        
        # 2. Get ML predictions
        predictions = await self.ml_agent.predict_all_players(weeks_ahead)
        
        # 3. Get transfer recommendations
        advice = await self.transfer_agent.optimize_transfers(
            current_data, predictions, weeks_ahead
        )
        
        # 4. Generate explanation
        explanation = await self.generate_explanation(advice)
        
        return TransferAdvice(
            recommended_transfers=advice.transfers,
            reasoning=explanation,
            expected_points_gain=advice.expected_gain,
            risk_assessment=advice.risk_level
        )
```

## ML Pipeline & Model Management

**CRITICAL: This project implements comprehensive ML pipeline for FPL predictions with proper validation and backtesting.**

### Model Architecture

**Player Points Prediction (`src/models/ml_models.py`):**

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib

class PlayerPredictor:
    """ML model for predicting player FPL points"""
    
    def __init__(self):
        self.models = {
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        }
        self.ensemble_weights = {'xgboost': 0.7, 'random_forest': 0.3}
        self.feature_columns = [
            'form_last_5', 'minutes_last_5', 'goals_per_90',
            'assists_per_90', 'fixture_difficulty', 'is_home',
            'team_strength', 'opponent_weakness', 'price_momentum'
        ]
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """Train ensemble model with cross-validation"""
        X = training_data[self.feature_columns]
        y = training_data['total_points']
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = {}
        
        for name, model in self.models.items():
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                cv_scores.append(score)
            
            scores[name] = np.mean(cv_scores)
            
        # Train final models on full data
        for model in self.models.values():
            model.fit(X, y)
            
        # Save models
        self.save_models()
        
        return scores
    
    def predict_ensemble(self, features: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(features)
        
        # Weighted ensemble
        ensemble_pred = sum(
            predictions[name] * self.ensemble_weights[name] 
            for name in self.models.keys()
        )
        
        return ensemble_pred
```

### Feature Engineering Pipeline

**Feature Creation (`src/data/processors.py`):**

```python
import pandas as pd
from typing import Dict, List
from src.models.data_models import PlayerFeatures

class FeatureEngineer:
    """Advanced feature engineering for FPL predictions"""
    
    def create_player_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""
        
        # Sort by player and gameweek
        df = player_data.sort_values(['player_id', 'gameweek'])
        
        # Rolling statistics (last 5 games)
        df['form_last_5'] = df.groupby('player_id')['total_points'].rolling(5, min_periods=1).mean()
        df['minutes_last_5'] = df.groupby('player_id')['minutes'].rolling(5, min_periods=1).mean()
        df['goals_last_5'] = df.groupby('player_id')['goals_scored'].rolling(5, min_periods=1).sum()
        
        # Per-90 minute statistics
        df['goals_per_90'] = (df['goals_scored'] / df['minutes']) * 90
        df['assists_per_90'] = (df['assists'] / df['minutes']) * 90
        df['points_per_90'] = (df['total_points'] / df['minutes']) * 90
        
        # Expected vs actual performance
        df['xg_vs_goals'] = df['expected_goals'] - df['goals_scored']
        df['xa_vs_assists'] = df['expected_assists'] - df['assists']
        
        # Fixture difficulty
        df['fixture_difficulty'] = self.calculate_fixture_difficulty(df)
        df['upcoming_fixtures'] = self.get_upcoming_fixture_difficulty(df)
        
        # Team and opponent strength
        df['team_strength'] = self.get_team_strength(df['team'])
        df['opponent_strength'] = self.get_opponent_strength(df['opponent_team'])
        
        # Price and ownership trends
        df['price_change_momentum'] = df.groupby('player_id')['now_cost'].diff()
        df['ownership_trend'] = df.groupby('player_id')['selected_by_percent'].diff()
        
        # Position-specific features
        df = self.add_position_features(df)
        
        # Clean and validate
        df = df.fillna(0)  # Handle NaN values
        
        return df
    
    def calculate_fixture_difficulty(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fixture difficulty based on opponent strength"""
        difficulty_mapping = {
            'easy': 2, 'medium': 3, 'hard': 4, 'very_hard': 5
        }
        
        # This would be more sophisticated in practice
        return df['opponent_team'].map(self.get_team_difficulty_mapping())
    
    def add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific features"""
        
        # Defenders
        df.loc[df['position'] == 'DEF', 'clean_sheet_probability'] = self.calculate_cs_probability(df)
        
        # Midfielders  
        df.loc[df['position'] == 'MID', 'creativity_index'] = df['key_passes'] + df['assists']
        
        # Forwards
        df.loc[df['position'] == 'FWD', 'penalty_taker'] = self.identify_penalty_takers(df)
        
        return df
```

### Model Validation & Backtesting

**Validation Framework (`scripts/validate_models.py`):**

```python
import pandas as pd
from src.models.ml_models import PlayerPredictor
from src.data.fetchers import FPLDataFetcher
from typing import Dict, List

class ModelValidator:
    """Comprehensive model validation and backtesting"""
    
    def __init__(self):
        self.predictor = PlayerPredictor()
        self.data_fetcher = FPLDataFetcher()
    
    def backtest_predictions(self, seasons: List[str]) -> Dict[str, float]:
        """Backtest model performance on historical seasons"""
        
        results = {}
        
        for season in seasons:
            print(f"Backtesting season {season}...")
            
            # Get historical data
            historical_data = self.data_fetcher.get_season_data(season)
            
            # Split into training and testing
            train_weeks = list(range(1, 29))  # GW 1-28 for training
            test_weeks = list(range(29, 39))   # GW 29-38 for testing
            
            train_data = historical_data[historical_data['gameweek'].isin(train_weeks)]
            test_data = historical_data[historical_data['gameweek'].isin(test_weeks)]
            
            # Train model
            self.predictor.train(train_data)
            
            # Make predictions
            test_features = test_data[self.predictor.feature_columns]
            predictions = self.predictor.predict_ensemble(test_features)
            actual = test_data['total_points'].values
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            correlation = np.corrcoef(predictions, actual)[0, 1]
            
            results[season] = {
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation,
                'accuracy': self.calculate_accuracy(predictions, actual)
            }
            
            print(f"Season {season} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Correlation: {correlation:.3f}")
        
        return results
    
    def validate_transfer_decisions(self, seasons: List[str]) -> Dict[str, float]:
        """Validate transfer recommendation quality"""
        
        from src.agents.transfer_advisor import TransferAdvisorAgent
        transfer_agent = TransferAdvisorAgent()
        
        total_roi = []
        
        for season in seasons:
            season_data = self.data_fetcher.get_season_data(season)
            
            # Simulate weekly transfer decisions
            for gw in range(2, 38):  # Start from GW2 (can make transfers)
                current_data = season_data[season_data['gameweek'] <= gw]
                future_data = season_data[season_data['gameweek'] == gw + 1]
                
                # Get transfer recommendation
                advice = transfer_agent.suggest_transfer(current_data, weeks_ahead=1)
                
                if advice.recommended_transfers:
                    # Calculate actual ROI
                    roi = self.calculate_transfer_roi(advice, future_data)
                    total_roi.append(roi)
        
        return {
            'average_roi': np.mean(total_roi),
            'success_rate': len([roi for roi in total_roi if roi > 0]) / len(total_roi),
            'total_transfers': len(total_roi)
        }
```

## Optimization Engine

**CRITICAL: This project uses advanced optimization algorithms for team selection and transfer planning.**

### Team Optimization (`src/models/optimization.py`):**

```python
from pulp import *
import pandas as pd
from typing import Dict, List, Optional
from src.models.data_models import OptimizationResult, Player

class FPLOptimizer:
    """Advanced team optimization using linear programming"""
    
    def __init__(self):
        self.position_limits = {
            'GK': {'min': 2, 'max': 2},
            'DEF': {'min': 5, 'max': 5},
            'MID': {'min': 5, 'max': 5},
            'FWD': {'min': 3, 'max': 3}
        }
        self.budget = 100.0
        self.max_players_per_team = 3
    
    def optimize_team(
        self, 
        players: pd.DataFrame, 
        predicted_points: Dict[int, float],
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """Find optimal 15-player squad using linear programming"""
        
        # Create decision variables
        player_vars = {}
        for idx, player in players.iterrows():
            player_vars[player['id']] = LpVariable(
                f"player_{player['id']}", 
                cat='Binary'
            )
        
        # Create optimization problem
        prob = LpProblem("FPL_Team_Optimization", LpMaximize)
        
        # Objective: maximize predicted points
        prob += lpSum([
            predicted_points.get(player['id'], 0) * player_vars[player['id']]
            for _, player in players.iterrows()
        ])
        
        # Constraints
        self._add_budget_constraint(prob, players, player_vars)
        self._add_squad_size_constraint(prob, players, player_vars)
        self._add_position_constraints(prob, players, player_vars)
        self._add_team_constraints(prob, players, player_vars)
        
        # Additional constraints
        if constraints:
            self._add_custom_constraints(prob, players, player_vars, constraints)
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if prob.status == LpStatusOptimal:
            selected_players = []
            total_cost = 0
            total_predicted_points = 0
            
            for _, player in players.iterrows():
                if player_vars[player['id']].varValue == 1:
                    selected_players.append(player['id'])
                    total_cost += player['now_cost']
                    total_predicted_points += predicted_points.get(player['id'], 0)
            
            return OptimizationResult(
                status='optimal',
                selected_players=selected_players,
                total_cost=total_cost,
                predicted_points=total_predicted_points,
                remaining_budget=self.budget - total_cost
            )
        else:
            return OptimizationResult(
                status='infeasible',
                error=f"Optimization failed with status: {LpStatus[prob.status]}"
            )
    
    def optimize_transfers(
        self,
        current_team: List[int],
        available_players: pd.DataFrame,
        predicted_points: Dict[int, float],
        free_transfers: int = 1,
        weeks_ahead: int = 4
    ) -> OptimizationResult:
        """Optimize transfer decisions for multiple gameweeks"""
        
        # Multi-period optimization
        best_transfers = []
        cumulative_gain = 0
        
        for week in range(weeks_ahead):
            # Get predictions for this week
            week_predictions = {
                pid: predicted_points.get(pid, 0) * (0.9 ** week)  # Discount future
                for pid in predicted_points
            }
            
            # Find best single transfer for this week
            transfer = self._find_best_single_transfer(
                current_team, available_players, week_predictions
            )
            
            if transfer and transfer['points_gain'] > 0:
                best_transfers.append(transfer)
                cumulative_gain += transfer['points_gain']
                
                # Update current team
                current_team.remove(transfer['player_out'])
                current_team.append(transfer['player_in'])
        
        return OptimizationResult(
            status='optimal',
            recommended_transfers=best_transfers[:free_transfers],
            expected_points_gain=cumulative_gain,
            transfer_cost=max(0, len(best_transfers) - free_transfers) * 4
        )
    
    def _add_budget_constraint(self, prob, players, player_vars):
        """Add budget constraint to optimization problem"""
        prob += lpSum([
            player['now_cost'] * player_vars[player['id']]
            for _, player in players.iterrows()
        ]) <= self.budget
    
    def _add_position_constraints(self, prob, players, player_vars):
        """Add position constraints (2 GK, 5 DEF, 5 MID, 3 FWD)"""
        for position, limits in self.position_limits.items():
            position_players = players[players['element_type'] == position]
            prob += lpSum([
                player_vars[player['id']] 
                for _, player in position_players.iterrows()
            ]) >= limits['min']
            prob += lpSum([
                player_vars[player['id']] 
                for _, player in position_players.iterrows()
            ]) <= limits['max']
```

## CLI Interface Development

**CRITICAL: Comprehensive CLI interface using Click framework for all FPL operations.**

### CLI Architecture (`src/cli/main.py`):**

```python
import click
from rich.console import Console
from rich.table import Table
from src.agents.fpl_manager import FPLManagerAgent
from src.config.settings import get_settings

console = Console()

@click.group()
@click.pass_context
def fpl(ctx):
    """FPL ML System - AI-powered Fantasy Premier League management"""
    ctx.ensure_object(dict)
    ctx.obj['agent'] = FPLManagerAgent()
    ctx.obj['settings'] = get_settings()

@fpl.group()
def team():
    """Team management commands"""
    pass

@team.command()
@click.pass_context
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def show(ctx, format):
    """Display current team with analysis"""
    agent = ctx.obj['agent']
    
    with console.status("Fetching team data..."):
        team_data = agent.get_current_team()
    
    if format == 'table':
        table = Table(title="Current FPL Team")
        table.add_column("Player", style="cyan")
        table.add_column("Position", style="magenta")
        table.add_column("Price", style="green")
        table.add_column("Points", style="yellow")
        table.add_column("Form", style="blue")
        
        for player in team_data.players:
            table.add_row(
                player.name,
                player.position,
                f"£{player.price}M",
                str(player.total_points),
                f"{player.form:.1f}"
            )
        
        console.print(table)
        console.print(f"\nTotal Value: £{team_data.total_value}M")
        console.print(f"Money in Bank: £{team_data.money_in_bank}M")
    else:
        console.print_json(team_data.dict())

@team.command()
@click.pass_context
@click.option('--weeks', default=4, help='Weeks to optimize ahead')
@click.option('--strategy', type=click.Choice(['balanced', 'aggressive', 'conservative']), default='balanced')
def optimize(ctx, weeks, strategy):
    """Get optimal team selection"""
    agent = ctx.obj['agent']
    
    with console.status(f"Optimizing team for next {weeks} weeks..."):
        result = agent.optimize_team(weeks_ahead=weeks, strategy=strategy)
    
    console.print(f"[green]✓[/green] Optimization complete!")
    console.print(f"Expected points: {result.predicted_points:.1f}")
    console.print(f"Total cost: £{result.total_cost:.1f}M")
    
    # Display optimal team
    table = Table(title="Optimal Team")
    table.add_column("Position")
    table.add_column("Player")
    table.add_column("Price")
    table.add_column("Predicted Points")
    
    for player in result.selected_players:
        table.add_row(
            player.position,
            player.name,
            f"£{player.price}M",
            f"{player.predicted_points:.1f}"
        )
    
    console.print(table)

@fpl.group()
def transfer():
    """Transfer management commands"""
    pass

@transfer.command()
@click.pass_context
@click.option('--weeks', default=3, help='Weeks to plan ahead')
@click.option('--free-transfers', default=1, help='Number of free transfers available')
def suggest(ctx, weeks, free_transfers):
    """Get transfer recommendations"""
    agent = ctx.obj['agent']
    
    with console.status("Analyzing transfer options..."):
        advice = agent.get_transfer_advice(weeks_ahead=weeks, free_transfers=free_transfers)
    
    if not advice.recommended_transfers:
        console.print("[yellow]No beneficial transfers found. Hold your transfer![/yellow]")
        return
    
    console.print(f"[green]✓[/green] Found {len(advice.recommended_transfers)} recommended transfer(s)")
    
    for i, transfer in enumerate(advice.recommended_transfers, 1):
        console.print(f"\n[bold]Transfer {i}:[/bold]")
        console.print(f"  OUT: {transfer.player_out.name} (£{transfer.player_out.price}M)")
        console.print(f"  IN:  {transfer.player_in.name} (£{transfer.player_in.price}M)")
        console.print(f"  Expected gain: +{transfer.expected_points_gain:.1f} points")
        console.print(f"  Cost impact: £{transfer.cost_change:+.1f}M")
        
        # Show reasoning
        console.print("\n[bold]Reasoning:[/bold]")
        for reason in transfer.reasoning:
            console.print(f"  • {reason}")

@transfer.command()
@click.argument('player_out')
@click.argument('player_in')
@click.pass_context
def analyze(ctx, player_out, player_in):
    """Analyze a specific transfer option"""
    agent = ctx.obj['agent']
    
    with console.status(f"Analyzing {player_out} → {player_in}..."):
        analysis = agent.analyze_transfer(player_out, player_in)
    
    console.print(f"\n[bold]Transfer Analysis: {player_out} → {player_in}[/bold]")
    console.print(f"Expected points gain: {analysis.points_gain:+.1f}")
    console.print(f"Cost change: £{analysis.cost_change:+.1f}M")
    console.print(f"Risk level: {analysis.risk_level}")
    
    console.print(f"\n[bold]Comparison:[/bold]")
    table = Table()
    table.add_column("Metric")
    table.add_column(player_out, style="red")
    table.add_column(player_in, style="green")
    
    for metric, values in analysis.comparison.items():
        table.add_row(metric, str(values['out']), str(values['in']))
    
    console.print(table)

@fpl.group()
def player():
    """Player analysis commands"""
    pass

@player.command()
@click.argument('player_name')
@click.pass_context
@click.option('--weeks', default=5, help='Number of weeks to analyze')
def stats(ctx, player_name, weeks):
    """Show detailed player statistics"""
    agent = ctx.obj['agent']
    
    with console.status(f"Fetching stats for {player_name}..."):
        stats = agent.get_player_stats(player_name, weeks=weeks)
    
    console.print(f"\n[bold]{stats.name}[/bold] ({stats.team})")
    console.print(f"Position: {stats.position} | Price: £{stats.price}M")
    console.print(f"Ownership: {stats.ownership:.1f}% | Form: {stats.form:.1f}")
    
    # Recent performance table
    table = Table(title=f"Last {weeks} Gameweeks")
    table.add_column("GW")
    table.add_column("Opponent")
    table.add_column("Minutes")
    table.add_column("Points")
    table.add_column("Goals")
    table.add_column("Assists")
    
    for gw_stats in stats.recent_gameweeks:
        table.add_row(
            str(gw_stats.gameweek),
            gw_stats.opponent,
            str(gw_stats.minutes),
            str(gw_stats.points),
            str(gw_stats.goals),
            str(gw_stats.assists)
        )
    
    console.print(table)
    
    # Predictions
    console.print(f"\n[bold]Next 3 GW Prediction:[/bold]")
    console.print(f"Expected points: {stats.predicted_points:.1f}")
    console.print(f"Confidence: {stats.prediction_confidence:.0%}")

@player.command()
@click.argument('players', nargs=-1, required=True)
@click.pass_context
def compare(ctx, players):
    """Compare multiple players"""
    agent = ctx.obj['agent']
    
    with console.status(f"Comparing {len(players)} players..."):
        comparison = agent.compare_players(list(players))
    
    table = Table(title="Player Comparison")
    table.add_column("Metric")
    for player in players:
        table.add_column(player, style="cyan")
    
    for metric in comparison.metrics:
        row = [metric.name]
        for player in players:
            value = metric.values.get(player, "N/A")
            row.append(str(value))
        table.add_row(*row)
    
    console.print(table)
    
    # Show best in each category
    console.print(f"\n[bold]Best in Category:[/bold]")
    for category, best_player in comparison.best_in_category.items():
        console.print(f"  {category}: [green]{best_player}[/green]")

@fpl.group()
def captain():
    """Captain selection commands"""
    pass

@captain.command()
@click.pass_context
@click.option('--strategy', type=click.Choice(['safe', 'balanced', 'differential']), default='balanced')
def suggest(ctx, strategy):
    """Get captain recommendations"""
    agent = ctx.obj['agent']
    
    with console.status("Analyzing captain options..."):
        recommendations = agent.get_captain_recommendations(strategy=strategy)
    
    console.print(f"[bold]Captain Recommendations ({strategy} strategy):[/bold]")
    
    for i, rec in enumerate(recommendations.options[:3], 1):
        console.print(f"\n{i}. [green]{rec.player.name}[/green] vs {rec.opponent}")
        console.print(f"   Expected: {rec.expected_points:.1f} points")
        console.print(f"   Ownership: {rec.ownership:.1f}%")
        console.print(f"   Risk: {rec.risk_level}")
        
        for reason in rec.reasoning[:2]:
            console.print(f"   • {reason}")

@fpl.group()  
def data():
    """Data management commands"""
    pass

@data.command()
@click.pass_context
@click.option('--force', is_flag=True, help='Force update even if recent')
def update(ctx, force):
    """Update FPL data from all sources"""
    agent = ctx.obj['agent']
    
    with console.status("Updating FPL data..."):
        result = agent.update_all_data(force=force)
    
    console.print(f"[green]✓[/green] Data update complete")
    console.print(f"Players updated: {result.players_updated}")
    console.print(f"Fixtures updated: {result.fixtures_updated}")
    console.print(f"Results updated: {result.results_updated}")
    console.print(f"Update time: {result.update_time:.1f}s")

@data.command()
@click.pass_context
def health(ctx):
    """Check data pipeline health"""
    agent = ctx.obj['agent']
    
    with console.status("Checking system health..."):
        health = agent.check_system_health()
    
    console.print(f"[bold]System Health Report[/bold]")
    console.print(f"Overall status: {'[green]HEALTHY[/green]' if health.is_healthy else '[red]UNHEALTHY[/red]'}")
    
    table = Table()
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Last Updated")
    table.add_column("Issues")
    
    for component in health.components:
        status_color = "green" if component.is_healthy else "red"
        status_text = f"[{status_color}]{component.status}[/{status_color}]"
        
        table.add_row(
            component.name,
            status_text,
            component.last_updated.strftime("%Y-%m-%d %H:%M"),
            str(len(component.issues)) if component.issues else "None"
        )
    
    console.print(table)

@fpl.group()
def chip():
    """Chip strategy commands"""
    pass

@chip.command()
@click.pass_context
def plan(ctx):
    """Show optimal chip usage timeline"""
    agent = ctx.obj['agent']
    
    with console.status("Planning chip strategy..."):
        strategy = agent.plan_chip_strategy()
    
    console.print(f"[bold]Optimal Chip Strategy[/bold]")
    
    for chip in strategy.chips:
        console.print(f"\n[cyan]{chip.name}[/cyan]")
        console.print(f"  Optimal window: GW {chip.optimal_gameweek}")
        console.print(f"  Expected gain: +{chip.expected_points:.1f} points")
        console.print(f"  Confidence: {chip.confidence:.0%}")
        
        console.print(f"  Reasoning:")
        for reason in chip.reasoning:
            console.print(f"    • {reason}")

@chip.command()
@click.pass_context
@click.option('--gameweek', type=int, help='Target gameweek')
def wildcard(ctx, gameweek):
    """Plan wildcard team"""
    agent = ctx.obj['agent']
    
    if not gameweek:
        # Suggest optimal gameweek
        with console.status("Finding optimal wildcard timing..."):
            suggestion = agent.suggest_wildcard_timing()
        
        console.print(f"[yellow]Suggested wildcard gameweek: {suggestion.gameweek}[/yellow]")
        console.print(f"Expected benefit: +{suggestion.expected_points:.1f} points")
        return
    
    with console.status(f"Planning wildcard team for GW{gameweek}..."):
        wildcard_team = agent.plan_wildcard_team(gameweek=gameweek)
    
    console.print(f"[bold]Wildcard Team for GW{gameweek}[/bold]")
    console.print(f"Expected points: {wildcard_team.predicted_points:.1f}")
    console.print(f"Total cost: £{wildcard_team.total_cost:.1f}M")
    
    # Show team by position
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        position_players = [p for p in wildcard_team.players if p.position == position]
        console.print(f"\n[bold]{position}:[/bold]")
        for player in position_players:
            console.print(f"  {player.name} (£{player.price}M) - {player.predicted_points:.1f}pts")

if __name__ == '__main__':
    fpl()
```

## Dashboard Development

**CRITICAL: Interactive Streamlit dashboard for visual FPL analysis and team management.**

### Dashboard Architecture (`src/dashboard/app.py`):**

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.agents.fpl_manager import FPLManagerAgent
from src.config.settings import get_settings

# Page configuration
st.set_page_config(
    page_title="FPL ML System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = FPLManagerAgent()
if 'settings' not in st.session_state:
    st.session_state.settings = get_settings()

def main():
    st.title("⚽ FPL ML System Dashboard")
    st.markdown("AI-powered Fantasy Premier League management")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Team Overview", "Transfer Analysis", "Player Research", 
             "Captain Selection", "Chip Strategy", "Performance Tracking"]
        )
        
        # Quick stats
        st.header("Quick Stats")
        if st.button("Refresh Data"):
            with st.spinner("Updating data..."):
                st.session_state.agent.update_all_data()
            st.success("Data updated!")
        
        try:
            current_team = st.session_state.agent.get_current_team()
            st.metric("Team Value", f"£{current_team.total_value:.1f}M")
            st.metric("Money in Bank", f"£{current_team.money_in_bank:.1f}M")
            st.metric("Current Rank", f"{current_team.current_rank:,}")
        except Exception as e:
            st.error(f"Error loading team data: {e}")
    
    # Main content based on selected page
    if page == "Team Overview":
        show_team_overview()
    elif page == "Transfer Analysis":
        show_transfer_analysis()
    elif page == "Player Research":
        show_player_research()
    elif page == "Captain Selection":
        show_captain_selection()
    elif page == "Chip Strategy":
        show_chip_strategy()
    elif page == "Performance Tracking":
        show_performance_tracking()

def show_team_overview():
    st.header("Current Team Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current team display
        with st.spinner("Loading team data..."):
            team_data = st.session_state.agent.get_current_team()
        
        # Team table
        st.subheader("Starting XI")
        
        # Formation selector
        formation = st.selectbox("Formation", ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "5-3-2"])
        
        # Display team by position
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            position_players = [p for p in team_data.starting_xi if p.position == position]
            
            if position_players:
                st.write(f"**{position}**")
                cols = st.columns(len(position_players))
                
                for i, player in enumerate(position_players):
                    with cols[i]:
                        # Player card
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center;">
                                <h4>{player.name}</h4>
                                <p>£{player.price}M | {player.total_points}pts</p>
                                <p>Form: {player.form:.1f}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Bench
        st.subheader("Bench")
        bench_cols = st.columns(4)
        for i, player in enumerate(team_data.bench):
            with bench_cols[i]:
                st.write(f"{player.name} ({player.position})")
                st.write(f"£{player.price}M | {player.total_points}pts")
    
    with col2:
        # Team statistics
        st.subheader("Team Statistics")
        
        # Key metrics
        st.metric("Total Points", team_data.total_points)
        st.metric("Average Points/Player", f"{team_data.average_points:.1f}")
        st.metric("Team Form", f"{team_data.team_form:.1f}")
        
        # Next fixtures difficulty
        st.subheader("Next 5 Fixtures")
        
        fixture_difficulty = st.session_state.agent.get_fixture_difficulty(weeks=5)
        
        # Difficulty heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[[player['difficulty'] for player in week] for week in fixture_difficulty],
            x=[f"GW{i+1}" for i in range(5)],
            y=[player['name'] for player in fixture_difficulty[0]],
            colorscale='RdYlGn_r'
        ))
        
        fig.update_layout(title="Fixture Difficulty Heatmap", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization button
        if st.button("Optimize Team", type="primary"):
            with st.spinner("Optimizing team..."):
                optimization = st.session_state.agent.optimize_team()
            
            st.success(f"Optimization complete! Expected improvement: +{optimization.points_improvement:.1f} points")

def show_transfer_analysis():
    st.header("Transfer Analysis")
    
    # Transfer options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Transfer Recommendations")
        
        weeks_ahead = st.slider("Weeks to plan ahead", 1, 8, 4)
        free_transfers = st.number_input("Free transfers available", 1, 2, 1)
        
        if st.button("Get Recommendations"):
            with st.spinner("Analyzing transfer options..."):
                advice = st.session_state.agent.get_transfer_advice(
                    weeks_ahead=weeks_ahead,
                    free_transfers=free_transfers
                )
            
            if advice.recommended_transfers:
                for i, transfer in enumerate(advice.recommended_transfers, 1):
                    with st.expander(f"Transfer {i}: {transfer.player_out.name} → {transfer.player_in.name}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**OUT:**")
                            st.write(f"{transfer.player_out.name}")
                            st.write(f"Price: £{transfer.player_out.price}M")
                            st.write(f"Form: {transfer.player_out.form:.1f}")
                        
                        with col_b:
                            st.write("**IN:**")
                            st.write(f"{transfer.player_in.name}")
                            st.write(f"Price: £{transfer.player_in.price}M")
                            st.write(f"Form: {transfer.player_in.form:.1f}")
                        
                        st.write(f"**Expected gain:** +{transfer.expected_points_gain:.1f} points")
                        st.write(f"**Cost impact:** £{transfer.cost_change:+.1f}M")
                        
                        st.write("**Reasoning:**")
                        for reason in transfer.reasoning:
                            st.write(f"• {reason}")
            else:
                st.info("No beneficial transfers found. Consider holding your transfer!")
    
    with col2:
        st.subheader("Transfer Analyzer")
        
        # Manual transfer analysis
        player_out = st.selectbox("Player to transfer out", 
                                  options=[p.name for p in st.session_state.agent.get_current_team().all_players])
        
        # Get available replacements
        if player_out:
            available_players = st.session_state.agent.get_available_replacements(player_out)
            player_in = st.selectbox("Player to transfer in", 
                                     options=[p.name for p in available_players])
            
            if player_in and st.button("Analyze Transfer"):
                with st.spinner("Analyzing transfer..."):
                    analysis = st.session_state.agent.analyze_transfer(player_out, player_in)
                
                # Show analysis
                st.write(f"**Expected points gain:** {analysis.points_gain:+.1f} over {weeks_ahead} weeks")
                st.write(f"**Cost change:** £{analysis.cost_change:+.1f}M")
                st.write(f"**Risk level:** {analysis.risk_level}")
                
                # Comparison chart
                metrics = ['Total Points', 'Form', 'Minutes/Game', 'Price']
                player_out_values = [analysis.comparison[m]['out'] for m in metrics]
                player_in_values = [analysis.comparison[m]['in'] for m in metrics]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=player_out_values,
                    theta=metrics,
                    fill='toself',
                    name=player_out,
                    line_color='red'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=player_in_values,
                    theta=metrics,
                    fill='toself',
                    name=player_in,
                    line_color='green'
                ))
                
                fig.update_layout(title="Player Comparison", polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig, use_container_width=True)

def show_player_research():
    st.header("Player Research")
    
    # Player search
    player_name = st.text_input("Search for player")
    
    if player_name:
        with st.spinner(f"Loading data for {player_name}..."):
            player_stats = st.session_state.agent.get_player_stats(player_name)
        
        if player_stats:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Player info
                st.subheader(f"{player_stats.name} ({player_stats.team})")
                
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.metric("Position", player_stats.position)
                with info_cols[1]:
                    st.metric("Price", f"£{player_stats.price}M")
                with info_cols[2]:
                    st.metric("Total Points", player_stats.total_points)
                with info_cols[3]:
                    st.metric("Ownership", f"{player_stats.ownership:.1f}%")
                
                # Performance chart
                fig = px.line(
                    x=range(1, len(player_stats.gameweek_points) + 1),
                    y=player_stats.gameweek_points,
                    title=f"{player_stats.name} - Points per Gameweek",
                    labels={'x': 'Gameweek', 'y': 'Points'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent form table
                st.subheader("Recent Performance")
                recent_data = []
                for gw in player_stats.recent_gameweeks[-5:]:
                    recent_data.append({
                        'GW': gw.gameweek,
                        'Opponent': gw.opponent,
                        'H/A': 'H' if gw.was_home else 'A',
                        'Minutes': gw.minutes,
                        'Points': gw.points,
                        'Goals': gw.goals,
                        'Assists': gw.assists
                    })
                
                st.dataframe(recent_data, use_container_width=True)
            
            with col2:
                # Predictions and analysis
                st.subheader("Predictions")
                
                st.metric("Next GW Prediction", f"{player_stats.next_gw_prediction:.1f} pts")
                st.metric("3 GW Prediction", f"{player_stats.predicted_points:.1f} pts")
                st.metric("Confidence", f"{player_stats.prediction_confidence:.0%}")
                
                # Strengths and weaknesses
                st.subheader("Analysis")
                
                st.write("**Strengths:**")
                for strength in player_stats.strengths:
                    st.write(f"• {strength}")
                
                st.write("**Concerns:**")
                for concern in player_stats.concerns:
                    st.write(f"• {concern}")
                
                # Price change prediction
                price_change_prob = st.session_state.agent.predict_price_change(player_name)
                
                if price_change_prob['rise'] > 0.7:
                    st.warning(f"⬆️ Likely to rise ({price_change_prob['rise']:.0%} chance)")
                elif price_change_prob['fall'] > 0.7:
                    st.warning(f"⬇️ Likely to fall ({price_change_prob['fall']:.0%} chance)")
                else:
                    st.info("💰 Price likely stable")

if __name__ == "__main__":
    main()
```

## Data Models & Validation

**CRITICAL: All data structures MUST use Pydantic models for validation and type safety.**

### Core Data Models (`src/models/data_models.py`):**

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class Position(str, Enum):
    GOALKEEPER = "GK"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"

class Player(BaseModel):
    """Core player model with validation"""
    id: int = Field(..., description="FPL player ID")
    name: str = Field(..., min_length=1, description="Player full name")
    web_name: str = Field(..., description="Short display name")
    position: Position = Field(..., description="Player position")
    team: str = Field(..., description="Team name")
    team_id: int = Field(..., description="FPL team ID")
    
    # Financial data
    now_cost: float = Field(..., ge=3.5, le=15.0, description="Current price in millions")
    cost_change_start: float = Field(0, description="Price change from season start")
    
    # Performance metrics
    total_points: int = Field(0, ge=0, description="Total points this season")
    points_per_game: float = Field(0, ge=0, description="Average points per game")
    form: float = Field(0, ge=0, le=10, description="Form rating over last 5 games")
    
    # Usage statistics
    selected_by_percent: float = Field(0, ge=0, le=100, description="Ownership percentage")
    minutes: int = Field(0, ge=0, description="Total minutes played")
    
    # Status
    chance_of_playing_this_round: Optional[int] = Field(None, ge=0, le=100)
    news: Optional[str] = Field(None, description="Latest news")
    
    @validator('now_cost')
    def validate_price(cls, v):
        """Ensure price is in valid FPL range"""
        if not (3.5 <= v <= 15.0):
            raise ValueError('Price must be between £3.5M and £15.0M')
        return v
    
    @validator('selected_by_percent')
    def validate_ownership(cls, v):
        """Ensure ownership is valid percentage"""
        if not (0 <= v <= 100):
            raise ValueError('Ownership must be between 0% and 100%')
        return v

class GameweekPerformance(BaseModel):
    """Player performance in a specific gameweek"""
    player_id: int
    gameweek: int = Field(..., ge=1, le=38)
    opponent_team: str
    was_home: bool
    
    # Match statistics
    minutes: int = Field(0, ge=0, le=90)
    goals_scored: int = Field(0, ge=0)
    assists: int = Field(0, ge=0)
    clean_sheets: int = Field(0, ge=0, le=1)
    goals_conceded: int = Field(0, ge=0)
    saves: int = Field(0, ge=0)
    
    # Bonus and advanced stats
    bonus: int = Field(0, ge=0, le=3)
    bps: int = Field(0, ge=0, description="Bonus points system score")
    total_points: int = Field(0, ge=0)
    
    # Expected stats
    expected_goals: Optional[float] = Field(None, ge=0)
    expected_assists: Optional[float] = Field(None, ge=0)
    expected_goal_involvements: Optional[float] = Field(None, ge=0)

class Team(BaseModel):
    """FPL team model"""
    id: int
    name: str
    short_name: str
    
    # Strength ratings
    strength: int = Field(..., ge=1, le=5)
    strength_overall_home: int = Field(..., ge=1, le=5)
    strength_overall_away: int = Field(..., ge=1, le=5)
    strength_attack_home: int = Field(..., ge=1, le=5)
    strength_attack_away: int = Field(..., ge=1, le=5)
    strength_defence_home: int = Field(..., ge=1, le=5)
    strength_defence_away: int = Field(..., ge=1, le=5)

class Fixture(BaseModel):
    """Match fixture model"""
    id: int
    gameweek: int = Field(..., ge=1, le=38)
    team_h: int = Field(..., description="Home team ID")
    team_a: int = Field(..., description="Away team ID")
    
    kickoff_time: Optional[datetime] = None
    finished: bool = False
    
    # Results (when finished)
    team_h_score: Optional[int] = Field(None, ge=0)
    team_a_score: Optional[int] = Field(None, ge=0)
    
    # Difficulty ratings
    team_h_difficulty: int = Field(..., ge=1, le=5)
    team_a_difficulty: int = Field(..., ge=1, le=5)

class TransferSuggestion(BaseModel):
    """Transfer recommendation model"""
    player_out: Player
    player_in: Player
    
    expected_points_gain: float = Field(..., description="Expected points gain over planning period")
    cost_change: float = Field(..., description="Price difference")
    
    reasoning: List[str] = Field(..., description="Reasons for this transfer")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in recommendation")
    risk_level: str = Field(..., regex="^(Low|Medium|High)$")
    
    # Contextual information
    weeks_analyzed: int = Field(..., ge=1, description="Number of weeks this analysis covers")
    fixture_analysis: Dict[str, Any] = Field(default_factory=dict)

class OptimizationResult(BaseModel):
    """Team optimization result"""
    status: str = Field(..., regex="^(optimal|infeasible|timeout)$")
    
    # Results (when optimal)
    selected_players: Optional[List[int]] = None
    total_cost: Optional[float] = None
    predicted_points: Optional[float] = None
    remaining_budget: Optional[float] = None
    
    # Transfer-specific results
    recommended_transfers: Optional[List[TransferSuggestion]] = None
    transfer_cost: Optional[int] = None
    expected_points_gain: Optional[float] = None
    
    # Error information
    error: Optional[str] = None
    
    @validator('total_cost')
    def validate_total_cost(cls, v):
        if v is not None and v > 100.0:
            raise ValueError('Total cost cannot exceed £100.0M')
        return v

class PredictionResult(BaseModel):
    """ML prediction result"""
    player_id: int
    gameweeks_ahead: int = Field(..., ge=1, le=8)
    
    expected_points: List[float] = Field(..., description="Expected points for each gameweek")
    confidence_intervals: List[Dict[str, float]] = Field(..., description="95% confidence intervals")
    
    # Model information
    model_used: str = Field(..., description="Primary model used for prediction")
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    prediction_date: datetime = Field(default_factory=datetime.now)
    
    @validator('expected_points')
    def validate_expected_points(cls, v, values):
        if 'gameweeks_ahead' in values and len(v) != values['gameweeks_ahead']:
            raise ValueError('Expected points length must match gameweeks_ahead')
        return v

class ChipStrategy(BaseModel):
    """Chip usage strategy"""
    chip_name: str = Field(..., regex="^(wildcard|free_hit|bench_boost|triple_captain)$")
    optimal_gameweek: int = Field(..., ge=1, le=38)
    expected_points_gain: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
    
    reasoning: List[str] = Field(..., min_items=1)
    requirements: List[str] = Field(default_factory=list, description="Prerequisites for optimal usage")
    
    # Timing considerations
    earliest_recommended: int = Field(..., ge=1, le=38)
    latest_recommended: int = Field(..., ge=1, le=38)
    
    @validator('latest_recommended')
    def validate_timing(cls, v, values):
        if 'earliest_recommended' in values and v < values['earliest_recommended']:
            raise ValueError('Latest recommended must be >= earliest recommended')
        return v

class SystemHealth(BaseModel):
    """System health status"""
    is_healthy: bool
    last_check: datetime = Field(default_factory=datetime.now)
    
    components: List['ComponentHealth'] = Field(..., min_items=1)
    
    @property
    def unhealthy_components(self) -> List['ComponentHealth']:
        return [c for c in self.components if not c.is_healthy]

class ComponentHealth(BaseModel):
    """Individual component health"""
    name: str
    is_healthy: bool
    status: str
    last_updated: datetime
    
    issues: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

# Enable forward references
SystemHealth.model_rebuild()
```

## Error Handling & Logging

**CRITICAL: Comprehensive error handling with structured logging for production debugging.**

### Logging Configuration (`src/utils/logging.py`):**

```python
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

class FPLFormatter(logging.Formatter):
    """Custom formatter for FPL system logs"""
    
    def format(self, record):
        # Add contextual information
        record.timestamp = datetime.now().isoformat()
        record.component = getattr(record, 'component', 'unknown')
        record.user_id = getattr(record, 'user_id', None)
        record.operation = getattr(record, 'operation', None)
        
        # Structured logging for production
        if hasattr(record, 'structured') and record.structured:
            log_data = {
                'timestamp': record.timestamp,
                'level': record.levelname,
                'component': record.component,
                'message': record.getMessage(),
                'operation': record.operation,
                'user_id': record.user_id,
                'pathname': record.pathname,
                'lineno': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_data)
        
        # Human-readable format for development
        format_str = (
            "%(asctime)s | %(levelname)-8s | %(component)-15s | "
            "%(message)s"
        )
        
        if record.operation:
            format_str += " | Op: %(operation)s"
        
        if record.user_id:
            format_str += " | User: %(user_id)s"
        
        formatter = logging.Formatter(format_str)
        return formatter.format(record)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False
) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logger
    logger = logging.getLogger('fpl_system')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(FPLFormatter())
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FPLFormatter())
        logger.addHandler(file_handler)
    
    # Add structured logging flag
    logger.structured = structured
    
    return logger

class FPLLogger:
    """Context-aware logger for FPL system"""
    
    def __init__(self, component: str, user_id: Optional[str] = None):
        self.logger = logging.getLogger('fpl_system')
        self.component = component
        self.user_id = user_id
    
    def _log(self, level: int, message: str, operation: Optional[str] = None, **kwargs):
        """Internal logging method with context"""
        extra = {
            'component': self.component,
            'user_id': self.user_id,
            'operation': operation,
            'structured': getattr(self.logger, 'structured', False),
            **kwargs
        }
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, operation: Optional[str] = None, **kwargs):
        self._log(logging.INFO, message, operation, **kwargs)
    
    def error(self, message: str, operation: Optional[str] = None, **kwargs):
        self._log(logging.ERROR, message, operation, **kwargs)
    
    def warning(self, message: str, operation: Optional[str] = None, **kwargs):
        self._log(logging.WARNING, message, operation, **kwargs)
    
    def debug(self, message: str, operation: Optional[str] = None, **kwargs):
        self._log(logging.DEBUG, message, operation, **kwargs)

# Usage in agents
class AgentMixin:
    """Mixin to add logging to agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = FPLLogger(
            component=self.__class__.__name__,
            user_id=getattr(self, 'user_id', None)
        )
    
    def log_operation(self, operation: str, success: bool = True, **kwargs):
        """Log agent operation with context"""
        level = 'info' if success else 'error'
        message = f"Operation {'completed' if success else 'failed'}: {operation}"
        
        getattr(self.logger, level)(message, operation=operation, **kwargs)
```

### Exception Handling (`src/utils/exceptions.py`):**

```python
class FPLException(Exception):
    """Base exception for FPL system"""
    
    def __init__(self, message: str, component: str = None, operation: str = None, **kwargs):
        super().__init__(message)
        self.component = component
        self.operation = operation
        self.context = kwargs

class DataFetchError(FPLException):
    """Error fetching data from external sources"""
    pass

class ModelPredictionError(FPLException):
    """Error in ML model prediction"""
    pass

class OptimizationError(FPLException):
    """Error in team optimization"""
    pass

class ValidationError(FPLException):
    """Data validation error"""
    pass

def handle_fpl_exception(func):
    """Decorator for consistent exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FPLException:
            # Re-raise FPL exceptions
            raise
        except Exception as e:
            # Convert other exceptions to FPL exceptions
            component = getattr(args[0], '__class__.__name__', 'unknown') if args else 'unknown'
            raise FPLException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                component=component,
                operation=func.__name__
            ) from e
    
    return wrapper
```

## Testing Framework

**CRITICAL: Comprehensive testing strategy with unit tests, integration tests, and ML model validation.**

### Test Configuration (`tests/conftest.py`):**

```python
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.agents.fpl_manager import FPLManagerAgent
from src.models.ml_models import PlayerPredictor
from src.config.settings import get_settings

@pytest.fixture
def sample_players():
    """Sample player data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E'],
        'position': ['GK', 'DEF', 'MID', 'MID', 'FWD'],
        'team': [1, 1, 2, 3, 3],
        'now_cost': [4.5, 5.0, 6.5, 8.0, 9.5],
        'total_points': [50, 75, 100, 120, 80],
        'form': [3.2, 4.5, 6.8, 7.2, 5.1],
        'selected_by_percent': [15.5, 25.2, 45.8, 12.3, 8.9],
        'minutes': [1200, 1500, 1800, 1650, 1400]
    })

@pytest.fixture
def mock_fpl_agent():
    """Mock FPL agent for testing"""
    agent = Mock(spec=FPLManagerAgent)
    agent.get_current_team.return_value = Mock(
        total_value=95.5,
        money_in_bank=4.5,
        current_rank=250000
    )
    return agent

@pytest.fixture
def test_settings():
    """Test configuration settings"""
    return {
        'FPL_TEAM_ID': '12345',
        'DATABASE_URL': 'sqlite:///:memory:',
        'LOG_LEVEL': 'DEBUG',
        'MODEL_RETRAIN_THRESHOLD': 0.1
    }

@pytest.fixture
def player_predictor():
    """Initialize player predictor for testing"""
    predictor = PlayerPredictor()
    # Mock trained models for testing
    predictor.models = {
        'xgboost': Mock(),
        'random_forest': Mock()
    }
    return predictor

@pytest.fixture
def mock_api_response():
    """Mock FPL API response"""
    return {
        'elements': [
            {
                'id': 1,
                'web_name': 'Salah',
                'element_type': 3,
                'team': 1,
                'now_cost': 130,
                'total_points': 250,
                'form': '8.5',
                'selected_by_percent': '45.2'
            }
        ],
        'teams': [
            {
                'id': 1,
                'name': 'Liverpool',
                'strength': 5
            }
        ]
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Mock external API calls
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'elements': [], 'teams': []}
        yield
```

### Agent Testing (`tests/test_agents/test_fpl_manager.py`):**

```python
import pytest
from unittest.mock import patch, Mock
from src.agents.fpl_manager import FPLManagerAgent
from src.models.data_models import Player, TransferSuggestion

class TestFPLManagerAgent:
    """Test suite for FPL Manager Agent"""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        agent = FPLManagerAgent()
        assert agent is not None
        assert hasattr(agent, 'data_agent')
        assert hasattr(agent, 'ml_agent')
        assert hasattr(agent, 'transfer_agent')
    
    @patch('src.agents.fpl_manager.FPLManagerAgent.get_current_team')
    def test_get_current_team(self, mock_get_team):
        """Test getting current team data"""
        # Mock response
        mock_team = Mock()
        mock_team.total_value = 95.5
        mock_team.money_in_bank = 4.5
        mock_get_team.return_value = mock_team
        
        agent = FPLManagerAgent()
        team = agent.get_current_team()
        
        assert team.total_value == 95.5
        assert team.money_in_bank == 4.5
        mock_get_team.assert_called_once()
    
    def test_transfer_advice_validation(self, sample_players):
        """Test transfer advice validation"""
        agent = FPLManagerAgent()
        
        # Mock prediction results
        with patch.object(agent.ml_agent, 'predict_all_players') as mock_predict:
            mock_predict.return_value = {1: 8.5, 2: 6.2, 3: 9.1}
            
            advice = agent.get_transfer_advice(weeks_ahead=3)
            
            assert advice is not None
            if advice.recommended_transfers:
                for transfer in advice.recommended_transfers:
                    assert isinstance(transfer, TransferSuggestion)
                    assert transfer.expected_points_gain >= 0
                    assert transfer.reasoning
    
    @pytest.mark.integration
    def test_full_optimization_workflow(self, sample_players):
        """Integration test for full optimization workflow"""
        agent = FPLManagerAgent()
        
        with patch.object(agent.data_agent, 'get_latest_data') as mock_data:
            mock_data.return_value = sample_players
            
            with patch.object(agent.ml_agent, 'predict_all_players') as mock_predict:
                mock_predict.return_value = {
                    i: float(7 + i) for i in sample_players['id']
                }
                
                result = agent.optimize_team(weeks_ahead=4)
                
                assert result.status == 'optimal'
                assert result.predicted_points > 0
                assert result.total_cost <= 100.0

class TestMLPredictionAgent:
    """Test suite for ML Prediction Agent"""
    
    def test_player_prediction(self, player_predictor, sample_players):
        """Test individual player prediction"""
        # Mock model predictions
        player_predictor.models['xgboost'].predict.return_value = [8.5]
        player_predictor.models['random_forest'].predict.return_value = [7.8]
        
        prediction = player_predictor.predict_player_points(
            player_id=1,
            features=sample_players.iloc[0:1],
            gameweeks=1
        )
        
        assert prediction is not None
        assert len(prediction.expected_points) == 1
        assert prediction.expected_points[0] > 0
    
    def test_model_ensemble(self, player_predictor):
        """Test ensemble prediction method"""
        import numpy as np
        
        # Mock individual model predictions
        features = np.array([[1, 2, 3, 4, 5]])
        
        player_predictor.models['xgboost'].predict.return_value = [8.0]
        player_predictor.models['random_forest'].predict.return_value = [7.0]
        
        ensemble_pred = player_predictor.predict_ensemble(features)
        
        # Should be weighted average: 0.7 * 8.0 + 0.3 * 7.0 = 7.7
        expected = 0.7 * 8.0 + 0.3 * 7.0
        assert abs(ensemble_pred[0] - expected) < 0.01
    
    @pytest.mark.slow
    def test_model_training_validation(self, sample_players):
        """Test model training with validation"""
        predictor = PlayerPredictor()
        
        # Create expanded training data
        training_data = sample_players.copy()
        for i in range(5):  # Create more samples
            new_data = sample_players.copy()
            new_data['total_points'] += i * 10
            training_data = pd.concat([training_data, new_data], ignore_index=True)
        
        # Add required feature columns
        for col in predictor.feature_columns:
            if col not in training_data.columns:
                training_data[col] = np.random.uniform(0, 5, len(training_data))
        
        scores = predictor.train(training_data)
        
        assert isinstance(scores, dict)
        assert 'xgboost' in scores
        assert 'random_forest' in scores
        assert all(score >= 0 for score in scores.values())

class TestOptimizationEngine:
    """Test suite for optimization algorithms"""
    
    def test_team_optimization_constraints(self, sample_players):
        """Test optimization respects FPL constraints"""
        from src.models.optimization import FPLOptimizer
        
        optimizer = FPLOptimizer()
        predicted_points = {i: float(5 + i) for i in sample_players['id']}
        
        result = optimizer.optimize_team(sample_players, predicted_points)
        
        if result.status == 'optimal':
            # Verify constraints
            assert result.total_cost <= 100.0
            assert len(result.selected_players) == 15
            
            # Check position constraints
            selected_data = sample_players[sample_players['id'].isin(result.selected_players)]
            position_counts = selected_data['position'].value_counts()
            
            assert position_counts.get('GK', 0) == 2
            assert position_counts.get('DEF', 0) == 5  
            assert position_counts.get('MID', 0) == 5
            assert position_counts.get('FWD', 0) == 3
    
    def test_transfer_optimization(self, sample_players):
        """Test transfer optimization logic"""
        from src.models.optimization import FPLOptimizer
        
        optimizer = FPLOptimizer()
        current_team = [1, 2, 3, 4, 5]  # Sample current team
        predicted_points = {i: float(6 + i) for i in sample_players['id']}
        
        result = optimizer.optimize_transfers(
            current_team=current_team,
            available_players=sample_players,
            predicted_points=predicted_points,
            free_transfers=1,
            weeks_ahead=3
        )
        
        assert result.status == 'optimal'
        if result.recommended_transfers:
            for transfer in result.recommended_transfers:
                assert transfer['player_out'] in current_team
                assert transfer['player_in'] not in current_team
                assert transfer['points_gain'] >= 0

# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    def test_prediction_speed(self, benchmark, player_predictor, sample_players):
        """Benchmark prediction speed"""
        features = sample_players[player_predictor.feature_columns].fillna(0)
        
        result = benchmark(player_predictor.predict_ensemble, features)
        
        assert len(result) == len(sample_players)
        # Should complete within reasonable time for real-time usage
    
    def test_optimization_speed(self, benchmark, sample_players):
        """Benchmark optimization speed"""
        from src.models.optimization import FPLOptimizer
        
        optimizer = FPLOptimizer()
        predicted_points = {i: float(5 + i) for i in sample_players['id']}
        
        result = benchmark(optimizer.optimize_team, sample_players, predicted_points)
        
        assert result.status in ['optimal', 'infeasible']
        # Should complete within 5 seconds for real-time usage
```

## Production Deployment

**CRITICAL: Production-ready deployment configuration with monitoring, scaling, and error handling.**

### Docker Configuration:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env.example .env

# Create data directories
RUN mkdir -p data/raw data/processed data/models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.utils.health import check_system_health; exit(0 if check_system_health() else 1)"

# Run application
CMD ["python", "-m", "src.cli.main", "monitor", "start"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  fpl-system:
    build: .
    ports:
      - "8501:8501"  # Streamlit dashboard
      - "8000:8000"  # API endpoint
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/fpl
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=fpl
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Quick Reference

### Adding New Features

1. **Create Pydantic models** in `src/models/data_models.py` for new data structures
2. **Add validation schemas** using Pydantic validators
3. **Implement agent methods** with proper error handling and logging
4. **Add CLI commands** in appropriate command groups
5. **Create dashboard components** for visual interaction
6. **Write comprehensive tests** including integration tests
7. **Update documentation** and type hints

### Development Workflow

```bash
# Daily development workflow
source venv/bin/activate              # Activate environment
python scripts/update_data.py         # Get latest FPL data
python -m pytest tests/               # Run tests
python -m src.cli.main team show      # Test CLI
streamlit run src/dashboard/app.py    # Test dashboard

# Before committing
black src/ tests/                     # Format code
isort src/ tests/                     # Sort imports
flake8 src/ tests/                    # Lint code
mypy src/                             # Type checking
pytest --cov=src                      # Run tests with coverage
```

### Monitoring & Debugging

```bash
# Check system health
python -m src.cli.main data health

# View logs
tail -f logs/fpl_system.log

# Performance profiling  
python -m cProfile -o profile.stats -m src.cli.main team optimize
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Database debugging
python -c "from src.config.database import get_db; print(get_db().execute('SELECT COUNT(*) FROM players').fetchone())"
```

This implementation guide provides a comprehensive foundation for building a production-ready FPL ML system using modern Python practices, Pydantic AI agents, and robust MLOps principles.