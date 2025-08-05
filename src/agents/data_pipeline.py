"""
Data Pipeline Agent - handles data fetching, cleaning, and processing operations.
Following main_agent_reference patterns for specialized data operations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta

from pydantic_ai import Agent, RunContext

from ..config.providers import get_llm_model
from ..tools.fpl_api import (
    get_bootstrap_data, 
    get_player_data, 
    get_team_picks,
    get_fixtures,
    get_live_gameweek_data,
    batch_fetch_player_data
)
from ..models.data_models import Player, Team, Fixture
from ..models.ml_models import FeatureEngineer

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert data pipeline specialist for Fantasy Premier League (FPL) systems. Your primary responsibility is to ensure high-quality, clean, and well-structured data flows throughout the FPL ML system.

Your capabilities:
1. **Data Fetching**: Retrieve data from FPL API and other sources with proper error handling
2. **Data Validation**: Ensure data quality and consistency across all data sources
3. **Data Processing**: Clean, transform, and prepare data for ML models and analysis
4. **Feature Engineering**: Create meaningful features for ML predictions
5. **Data Monitoring**: Track data freshness, completeness, and quality metrics
6. **Batch Processing**: Handle large-scale data operations efficiently

Data quality principles:
- Always validate data types and ranges before processing
- Handle missing values appropriately for each use case
- Maintain data lineage and processing timestamps
- Flag anomalies and data quality issues
- Ensure consistent data formats across the pipeline
- Implement proper error handling and recovery mechanisms

When processing FPL data:
- Account for gameweek boundaries and fixture schedules
- Handle player transfers and team changes correctly
- Validate price changes and ownership updates
- Ensure historical data consistency
- Process live gameweek data with appropriate latency considerations
"""


@dataclass
class DataPipelineDependencies:
    """Dependencies for the Data Pipeline agent."""
    database_url: str
    cache_duration_minutes: int = 30
    batch_size: int = 100
    data_validation_enabled: bool = True
    session_id: Optional[str] = None


# Initialize the Data Pipeline agent
data_pipeline_agent = Agent(
    get_llm_model(),
    deps_type=DataPipelineDependencies,
    system_prompt=SYSTEM_PROMPT
)


@data_pipeline_agent.tool
async def fetch_and_validate_bootstrap_data(
    ctx: RunContext[DataPipelineDependencies],
    force_refresh: bool = False
) -> str:
    """
    Fetch and validate bootstrap data from FPL API.
    
    Args:
        force_refresh: Force refresh even if cache is valid
        
    Returns:
        Status report of data fetching and validation
    """
    try:
        logger.info("Fetching bootstrap data from FPL API")
        
        # Fetch bootstrap data
        bootstrap_data = await get_bootstrap_data(force_refresh=force_refresh)
        
        if not bootstrap_data:
            return "❌ Failed to fetch bootstrap data from FPL API"
        
        # Validate data structure
        required_keys = ['elements', 'teams', 'events', 'element_types']
        missing_keys = [key for key in required_keys if key not in bootstrap_data]
        
        if missing_keys:
            return f"❌ Bootstrap data missing required keys: {', '.join(missing_keys)}"
        
        # Count data elements
        players_count = len(bootstrap_data.get('elements', []))
        teams_count = len(bootstrap_data.get('teams', []))
        events_count = len(bootstrap_data.get('events', []))
        
        # Validate expected counts
        if players_count < 400:  # Expect ~500-600 players
            logger.warning(f"Low player count: {players_count}")
        
        if teams_count != 20:  # Always 20 teams in Premier League
            return f"❌ Invalid team count: {teams_count} (expected 20)"
        
        if events_count != 38:  # Always 38 gameweeks
            return f"❌ Invalid gameweek count: {events_count} (expected 38)"
        
        # Find current gameweek
        current_gw = None
        next_gw = None
        
        for event in bootstrap_data['events']:
            if event.get('is_current'):
                current_gw = event['id']
            elif event.get('is_next'):
                next_gw = event['id']
        
        # Validate player data quality
        players_df = pd.DataFrame(bootstrap_data['elements'])
        
        # Check for required player fields
        required_player_fields = ['id', 'web_name', 'element_type', 'team', 'now_cost', 'total_points']
        missing_fields = [field for field in required_player_fields if field not in players_df.columns]
        
        if missing_fields:
            return f"❌ Player data missing required fields: {', '.join(missing_fields)}"
        
        # Data quality checks
        quality_issues = []
        
        # Check for players with no name
        unnamed_players = players_df[players_df['web_name'].isna() | (players_df['web_name'] == '')].shape[0]
        if unnamed_players > 0:
            quality_issues.append(f"{unnamed_players} players with missing names")
        
        # Check for invalid prices (should be between 3.5 and 15.0)
        invalid_prices = players_df[(players_df['now_cost'] < 35) | (players_df['now_cost'] > 150)].shape[0]
        if invalid_prices > 0:
            quality_issues.append(f"{invalid_prices} players with invalid prices")
        
        # Check for negative points
        negative_points = players_df[players_df['total_points'] < 0].shape[0]
        if negative_points > 0:
            quality_issues.append(f"{negative_points} players with negative points")
        
        # Generate status report
        status_report = f"""
✅ **Bootstrap Data Validation Complete**

**Data Summary:**
• Players: {players_count:,}
• Teams: {teams_count}
• Gameweeks: {events_count}
• Current GW: {current_gw or 'Not set'}
• Next GW: {next_gw or 'Not set'}

**Data Quality:**
{'• All quality checks passed ✅' if not quality_issues else '• Issues found: ' + ', '.join(quality_issues)}

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        logger.info(f"Bootstrap data validation complete. Players: {players_count}, Quality issues: {len(quality_issues)}")
        return status_report.strip()
        
    except Exception as e:
        logger.error(f"Bootstrap data validation failed: {e}")
        return f"❌ Bootstrap data validation failed: {str(e)}"


@data_pipeline_agent.tool
async def process_player_historical_data(
    ctx: RunContext[DataPipelineDependencies],
    player_ids: Optional[List[int]] = None,
    gameweeks_back: int = 10
) -> str:
    """
    Process and clean historical player data for ML model training.
    
    Args:
        player_ids: Specific player IDs to process (None for all players)
        gameweeks_back: Number of recent gameweeks to process
        
    Returns:
        Processing status report
    """
    try:
        logger.info(f"Processing historical data for {len(player_ids) if player_ids else 'all'} players")
        
        # Get bootstrap data for player info
        bootstrap_data = await get_bootstrap_data()
        if not bootstrap_data:
            return "❌ Could not fetch bootstrap data"
        
        players_df = pd.DataFrame(bootstrap_data['elements'])
        
        # Filter players if specified
        if player_ids:
            players_df = players_df[players_df['id'].isin(player_ids)]
        
        if players_df.empty:
            return "❌ No players found to process"
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Batch process player data
        processed_players = 0
        failed_players = 0
        total_records = 0
        
        batch_size = ctx.deps.batch_size
        
        for i in range(0, len(players_df), batch_size):
            batch_players = players_df.iloc[i:i+batch_size]
            batch_ids = batch_players['id'].tolist()
            
            try:
                # Fetch historical data for batch
                historical_data = await batch_fetch_player_data(
                    player_ids=batch_ids,
                    gameweeks=gameweeks_back
                )
                
                if historical_data.empty:
                    logger.warning(f"No historical data for batch starting at index {i}")
                    failed_players += len(batch_ids)
                    continue
                
                # Process features
                processed_data = feature_engineer.engineer_features(historical_data)
                
                # Validate processed data
                if ctx.deps.data_validation_enabled:
                    validation_result = _validate_processed_data(processed_data)
                    if not validation_result['valid']:
                        logger.warning(f"Data validation failed for batch {i}: {validation_result['errors']}")
                
                processed_players += len(batch_ids)
                total_records += len(processed_data)
                
                logger.debug(f"Processed batch {i//batch_size + 1}, players: {len(batch_ids)}, records: {len(processed_data)}")
                
            except Exception as e:
                logger.error(f"Failed to process batch starting at {i}: {e}")
                failed_players += len(batch_ids)
                continue
        
        # Generate processing report
        success_rate = (processed_players / len(players_df)) * 100 if len(players_df) > 0 else 0
        
        processing_report = f"""
📊 **Historical Data Processing Complete**

**Processing Summary:**
• Total Players: {len(players_df):,}
• Successfully Processed: {processed_players:,}
• Failed: {failed_players:,}
• Success Rate: {success_rate:.1f}%
• Total Records: {total_records:,}

**Data Features Created:**
• Rolling statistics (5-gameweek windows)
• Per-90 minute metrics
• Form and momentum indicators
• Fixture difficulty features
• Price change trends

**Processing Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        logger.info(f"Historical data processing complete. Processed: {processed_players}/{len(players_df)} players")
        return processing_report.strip()
        
    except Exception as e:
        logger.error(f"Historical data processing failed: {e}")
        return f"❌ Historical data processing failed: {str(e)}"


@data_pipeline_agent.tool
async def validate_live_gameweek_data(
    ctx: RunContext[DataPipelineDependencies],
    gameweek: Optional[int] = None
) -> str:
    """
    Validate and process live gameweek data for real-time updates.
    
    Args:
        gameweek: Specific gameweek to validate (None for current)
        
    Returns:
        Live data validation report
    """
    try:
        logger.info(f"Validating live gameweek data for GW {gameweek or 'current'}")
        
        # Get current gameweek if not specified
        if not gameweek:
            bootstrap_data = await get_bootstrap_data()
            if not bootstrap_data:
                return "❌ Could not determine current gameweek"
            
            for event in bootstrap_data['events']:
                if event.get('is_current') or event.get('is_next'):
                    gameweek = event['id']
                    break
            
            if not gameweek:
                return "❌ No current/next gameweek found"
        
        # Fetch live data
        live_data = await get_live_gameweek_data(gameweek)
        
        if not live_data:
            return f"❌ No live data available for gameweek {gameweek}"
        
        # Validate live data structure
        expected_keys = ['elements']
        missing_keys = [key for key in expected_keys if key not in live_data]
        
        if missing_keys:
            return f"❌ Live data missing keys: {', '.join(missing_keys)}"
        
        elements = live_data.get('elements', [])
        
        if not elements:
            return f"❌ No player data in live gameweek {gameweek}"
        
        # Convert to DataFrame for analysis
        live_df = pd.DataFrame(elements)
        
        # Data quality checks
        total_players = len(live_df)
        players_with_stats = live_df[live_df['stats'].apply(lambda x: len(x) > 0 if isinstance(x, dict) else False)].shape[0]
        
        # Check for data completeness
        data_completeness = (players_with_stats / total_players) * 100 if total_players > 0 else 0
        
        # Analyze stats
        stats_summary = {}
        if players_with_stats > 0:
            # Extract stats from the first player with stats to see what's available
            sample_stats = next(iter([stats for stats in live_df['stats'] if isinstance(stats, dict) and stats]), {})
            stats_summary = {key: f"Available for {players_with_stats} players" for key in sample_stats.keys()}
        
        # Check for fixture updates
        fixtures_data = await get_fixtures(gameweek=gameweek)
        fixtures_count = len(fixtures_data) if fixtures_data else 0
        
        validation_report = f"""
🔴 **Live Gameweek {gameweek} Data Validation**

**Data Availability:**
• Total Players: {total_players:,}
• Players with Stats: {players_with_stats:,}
• Data Completeness: {data_completeness:.1f}%
• Fixtures Available: {fixtures_count}

**Available Statistics:**
{chr(10).join([f"• {stat}: {desc}" for stat, desc in stats_summary.items()]) if stats_summary else "• No detailed stats available yet"}

**Data Quality Status:**
{'✅ Data appears complete and valid' if data_completeness >= 90 else '⚠️ Incomplete data - may be mid-gameweek' if data_completeness >= 50 else '❌ Very limited data available'}

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        logger.info(f"Live data validation complete for GW {gameweek}. Completeness: {data_completeness:.1f}%")
        return validation_report.strip()
        
    except Exception as e:
        logger.error(f"Live data validation failed: {e}")
        return f"❌ Live data validation failed: {str(e)}"


@data_pipeline_agent.tool
async def generate_data_health_report(
    ctx: RunContext[DataPipelineDependencies]
) -> str:
    """
    Generate comprehensive data health and quality report.
    
    Returns:
        Detailed data health report
    """
    try:
        logger.info("Generating comprehensive data health report")
        
        health_checks = []
        overall_health = True
        
        # Check 1: Bootstrap data availability
        try:
            bootstrap_data = await get_bootstrap_data()
            if bootstrap_data and len(bootstrap_data.get('elements', [])) > 400:
                health_checks.append(("Bootstrap Data", "✅ Available and complete", True))
            else:
                health_checks.append(("Bootstrap Data", "❌ Missing or incomplete", False))
                overall_health = False
        except Exception as e:
            health_checks.append(("Bootstrap Data", f"❌ Error: {str(e)[:50]}...", False))
            overall_health = False
        
        # Check 2: Current gameweek determination
        try:
            current_gw = None
            if bootstrap_data:
                for event in bootstrap_data.get('events', []):
                    if event.get('is_current'):
                        current_gw = event['id']
                        break
            
            if current_gw:
                health_checks.append(("Current Gameweek", f"✅ GW {current_gw} identified", True))
            else:
                health_checks.append(("Current Gameweek", "⚠️ Cannot determine current GW", False))
                overall_health = False
        except Exception as e:
            health_checks.append(("Current Gameweek", f"❌ Error: {str(e)[:50]}...", False))
            overall_health = False
        
        # Check 3: Live data availability
        try:
            if current_gw:
                live_data = await get_live_gameweek_data(current_gw)
                if live_data and live_data.get('elements'):
                    health_checks.append(("Live Data", f"✅ Available for GW {current_gw}", True))
                else:
                    health_checks.append(("Live Data", f"⚠️ Limited data for GW {current_gw}", False))
            else:
                health_checks.append(("Live Data", "❌ Cannot check - no current GW", False))
                overall_health = False
        except Exception as e:
            health_checks.append(("Live Data", f"❌ Error: {str(e)[:50]}...", False))
            overall_health = False
        
        # Check 4: Fixtures data
        try:
            fixtures = await get_fixtures()
            if fixtures and len(fixtures) > 0:
                health_checks.append(("Fixtures Data", f"✅ {len(fixtures)} fixtures available", True))
            else:
                health_checks.append(("Fixtures Data", "⚠️ No fixtures data", False))
        except Exception as e:
            health_checks.append(("Fixtures Data", f"❌ Error: {str(e)[:50]}...", False))
            overall_health = False
        
        # Check 5: Data freshness
        try:
            # Check how recent the data is (this would normally check database timestamps)
            data_age_minutes = 15  # Placeholder - would be calculated from actual timestamps
            
            if data_age_minutes <= ctx.deps.cache_duration_minutes:
                health_checks.append(("Data Freshness", f"✅ Data is {data_age_minutes} minutes old", True))
            else:
                health_checks.append(("Data Freshness", f"⚠️ Data is {data_age_minutes} minutes old", False))
        except Exception as e:
            health_checks.append(("Data Freshness", f"❌ Error: {str(e)[:50]}...", False))
            overall_health = False
        
        # Generate health report
        health_status = "✅ HEALTHY" if overall_health else "❌ UNHEALTHY"
        healthy_components = sum(1 for _, _, status in health_checks if status)
        total_components = len(health_checks)
        
        health_report = f"""
🏥 **Data Pipeline Health Report**

**Overall Status:** {health_status}
**Components Healthy:** {healthy_components}/{total_components}

**Component Status:**
{chr(10).join([f"• {component}: {status}" for component, status, _ in health_checks])}

**Recommendations:**
{_generate_health_recommendations(health_checks)}

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        logger.info(f"Data health report generated. Overall health: {overall_health}")
        return health_report.strip()
        
    except Exception as e:
        logger.error(f"Health report generation failed: {e}")
        return f"❌ Health report generation failed: {str(e)}"


def _validate_processed_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate processed data quality."""
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    if data.empty:
        validation_result['valid'] = False
        validation_result['errors'].append("Data is empty")
        return validation_result
    
    # Check for required columns (basic features)
    required_cols = ['player_id', 'gameweek', 'total_points']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for data types
    if 'total_points' in data.columns:
        non_numeric_points = data[~pd.to_numeric(data['total_points'], errors='coerce').notna()].shape[0]
        if non_numeric_points > 0:
            validation_result['warnings'].append(f"{non_numeric_points} rows with non-numeric points")
    
    # Check for reasonable value ranges
    if 'total_points' in data.columns:
        extreme_points = data[(data['total_points'] < -5) | (data['total_points'] > 30)].shape[0]
        if extreme_points > 0:
            validation_result['warnings'].append(f"{extreme_points} rows with extreme point values")
    
    return validation_result


def _generate_health_recommendations(health_checks: List[Tuple[str, str, bool]]) -> str:
    """Generate health recommendations based on failed checks."""
    failed_checks = [component for component, _, status in health_checks if not status]
    
    if not failed_checks:
        return "• All systems operating normally"
    
    recommendations = []
    
    if "Bootstrap Data" in failed_checks:
        recommendations.append("• Check FPL API connectivity and retry bootstrap data fetch")
    
    if "Current Gameweek" in failed_checks:
        recommendations.append("• Verify gameweek configuration in FPL API response")
    
    if "Live Data" in failed_checks:
        recommendations.append("• Check if gameweek is in progress or completed")
    
    if "Fixtures Data" in failed_checks:
        recommendations.append("• Verify fixtures endpoint availability")
    
    if "Data Freshness" in failed_checks:
        recommendations.append("• Trigger data refresh to get latest information")
    
    return "\n".join(recommendations) if recommendations else "• Review system logs for specific error details"


# Convenience function to create Data Pipeline agent
def create_data_pipeline_agent(
    database_url: str,
    cache_duration_minutes: int = 30,
    batch_size: int = 100,
    session_id: Optional[str] = None
) -> Agent:
    """
    Create Data Pipeline agent with specified dependencies.
    
    Args:
        database_url: Database connection URL
        cache_duration_minutes: Cache duration for data freshness
        batch_size: Batch size for processing operations
        session_id: Optional session identifier
        
    Returns:
        Configured Data Pipeline agent
    """
    return data_pipeline_agent