# FPL ML System - INITIAL.md

## FEATURE:
- **Pydantic AI FPL Manager** that orchestrates multiple specialized AI agents as tools
- **Data Pipeline Agent** for fetching and processing FPL data from APIs and web sources  
- **ML Prediction Agent** for player points prediction and team optimization
- **Transfer Advisor Agent** for weekly transfer recommendations and reasoning
- **Dashboard Agent** for generating insights and visualizations
- **Notification Agent** for sending alerts and updates
- **CLI interface** to interact with the FPL AI system
- **Streamlit web dashboard** for visual interaction and team management
- **FPL Official API**, **Football Data APIs**, **web scraping** for the data pipeline agent
- **XGBoost/scikit-learn** for ML models, **PuLP** for optimization algorithms

## EXAMPLES:

In the `examples/` folder, there is a README for you to read to understand what the example is all about and also how to structure your own README when you create documentation for the above feature.

- `examples/cli.py` - use this as a template to create the FPL CLI with commands for team optimization, transfer suggestions, player analysis
- `examples/agent/` - read through all of the files here to understand best practices for creating Pydantic AI agents that support different ML models, handling agent dependencies (databases, APIs), and adding tools to the agent (optimization, prediction, data fetching).

Don't copy any of these examples directly, it is for a different project entirely. But use this as inspiration and for best practices for FPL-specific agent architecture.

## DOCUMENTATION:

- Pydantic AI documentation: https://ai.pydantic.dev/
- FPL API documentation: https://fantasy.premierleague.com/api/
- PuLP optimization documentation: https://coin-or.github.io/pulp/
- XGBoost documentation: https://xgboost.readthedocs.io/
- Streamlit documentation: https://docs.streamlit.io/

## OTHER CONSIDERATIONS:

- Include a `.env.example` with FPL_TEAM_ID, API keys, database URLs, and notification settings
- Include README with instructions for setup including:
  - How to get FPL Team ID and configure API access
  - Setting up database (SQLite for development, PostgreSQL for production)
  - Configuring ML model training data sources
  - Setting up notification channels (email/Telegram/Discord)
  - Running the CLI commands and web dashboard
- Include the project structure in the README with clear separation between:
  - Data pipeline components (`src/data/`)
  - ML models and optimization (`src/models/`, `src/optimization/`)
  - AI agents (`src/agents/`)
  - CLI and dashboard interfaces (`src/cli/`, `src/dashboard/`)
  - Configuration and utilities (`src/config/`, `src/utils/`)
- Virtual environment has already been set up with the necessary dependencies including:
  - `pydantic-ai`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `pulp`
  - `streamlit`, `plotly`, `requests`, `beautifulsoup4`
  - `sqlalchemy`, `alembic` for database management
  - `python-dotenv`, `click` for CLI, `pytest` for testing
- Use `python_dotenv` and `load_dotenv()` for environment variables
- Implement proper error handling and logging for production FPL usage
- Include data validation using Pydantic models for FPL data structures
- Add caching mechanisms for API calls to respect rate limits
- Implement backup/restore functionality for FPL team configurations
- Add comprehensive testing suite including backtesting for ML models
- Include monitoring and alerting for system health and FPL performance
- Support for multiple FPL seasons and leagues
- Extensible architecture to add new prediction models and optimization strategies

## AGENT ARCHITECTURE:

### Primary FPL Manager Agent:
- **Tools**: Data Pipeline Agent, ML Prediction Agent, Transfer Advisor Agent, Dashboard Agent, Notification Agent
- **Responsibilities**: Orchestrate weekly FPL workflow, coordinate between agents, handle user requests
- **Models**: Support for multiple LLM providers (OpenAI, Anthropic, local models)

### Data Pipeline Agent:
- **Tools**: FPL API client, web scrapers, database connectors
- **Responsibilities**: Fetch player data, fixture information, injury news, historical statistics
- **Data Sources**: FPL Official API, Understat, FBref, injury news feeds

### ML Prediction Agent:
- **Tools**: Trained ML models (XGBoost, Random Forest, LSTM), feature engineering pipeline
- **Responsibilities**: Predict player points, estimate minutes played, forecast price changes
- **Models**: Separate models for different player positions and time horizons

### Transfer Advisor Agent:
- **Tools**: Optimization algorithms (PuLP), budget calculators, risk assessors
- **Responsibilities**: Recommend transfers, analyze trade-offs, plan chip usage
- **Strategies**: Single transfers, multi-week planning, wildcard optimization

### Dashboard Agent:
- **Tools**: Streamlit components, Plotly charts, data visualization libraries
- **Responsibilities**: Generate interactive dashboards, player comparison tools, performance tracking
- **Features**: Real-time updates, responsive design, export capabilities

### Notification Agent:
- **Tools**: Email clients, messaging APIs (Telegram, Discord), scheduling systems
- **Responsibilities**: Send alerts for price changes, injury news, deadline reminders
- **Channels**: Email, push notifications, instant messaging

## CLI COMMANDS STRUCTURE:

```bash
# Team management
fpl team show                    # Display current team
fpl team optimize               # Get optimal starting XI
fpl team value                  # Show team value and changes

# Transfer analysis  
fpl transfer suggest            # Get transfer recommendations
fpl transfer analyze IN OUT     # Analyze specific transfer
fpl transfer plan --weeks 4     # Multi-week transfer planning

# Player research
fpl player stats PLAYER_NAME    # Show detailed player statistics
fpl player compare P1 P2 P3     # Compare multiple players
fpl player predict --gameweeks 3 # Predict next 3 GW performance

# Captain selection
fpl captain suggest             # Get captain recommendations
fpl captain analyze             # Compare captain options
fpl captain differential        # Find differential captains

# Data management
fpl data update                 # Update all data sources
fpl data backtest --seasons 2   # Backtest models on historical data
fpl data health                 # Check data pipeline health

# Chip strategy
fpl chip plan                   # Show optimal chip usage timeline
fpl chip wildcard              # Plan wildcard team
fpl chip freehit --gameweek 18  # Plan free hit team

# System management  
fpl config show                 # Show current configuration
fpl config set KEY VALUE        # Update configuration
fpl monitor start              # Start monitoring services
```

## EXPECTED PROJECT STRUCTURE:

```
fpl-ml-system/
├── .env.example
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── database.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fpl_manager.py
│   │   ├── data_pipeline.py
│   │   ├── ml_prediction.py
│   │   ├── transfer_advisor.py
│   │   ├── dashboard.py
│   │   └── notification.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_models.py
│   │   ├── ml_models.py
│   │   └── optimization.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetchers.py
│   │   ├── processors.py
│   │   └── validators.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── team.py
│   │   ├── transfer.py
│   │   ├── player.py
│   │   └── utils.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── components/
│   │   └── pages/
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── cache.py
│       └── helpers.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── tests/
│   ├── __init__.py
│   ├── test_agents/
│   ├── test_models/
│   ├── test_data/
│   └── fixtures/
├── notebooks/
│   ├── exploration/
│   ├── modeling/
│   └── analysis/
├── scripts/
│   ├── setup_db.py
│   ├── train_models.py
│   └── deploy.py
└── docs/
    ├── setup.md
    ├── usage.md
    └── api.md
```

## DEVELOPMENT PHASES:

### Phase 1: Foundation (Weeks 1-4)
- Set up project structure and development environment
- Implement basic FPL Manager Agent with CLI interface
- Create Data Pipeline Agent with FPL API integration
- Build core data models and database schema
- Implement basic ML Prediction Agent with simple models

### Phase 2: Core Features (Weeks 5-8)  
- Develop Transfer Advisor Agent with optimization algorithms
- Create Dashboard Agent with Streamlit interface
- Implement comprehensive player analysis and comparison tools
- Add captain selection and chip strategy features
- Build notification system for alerts and updates

### Phase 3: Advanced Features (Weeks 9-12)
- Enhance ML models with advanced algorithms and features
- Implement multi-week planning and season-long strategies
- Add backtesting and performance validation
- Create monitoring and alerting systems
- Polish user experience and add advanced visualizations

### Phase 4: Production (Ongoing)
- Deploy to cloud infrastructure
- Implement continuous integration and deployment
- Add user authentication and multi-user support
- Create mobile-responsive dashboard
- Build community features and sharing capabilities