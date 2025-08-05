"""
Pure FPL API tool functions for agent integration.
Following main_agent_reference/tools.py patterns with proper error handling.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
from ..models.data_models import Player, Team, Fixture, Event

logger = logging.getLogger(__name__)

# FPL API base configuration
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.1  # 100ms between requests to respect rate limits


async def get_bootstrap_data() -> Dict[str, Any]:
    """
    Pure function to get FPL bootstrap-static data.
    
    Returns:
        Dictionary containing players, teams, events, and game settings
        
    Raises:
        ValueError: If API request fails or returns invalid data
        Exception: If network or parsing errors occur
    """
    url = f"{FPL_BASE_URL}/bootstrap-static/"
    
    logger.info("Fetching FPL bootstrap data")
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.get(url)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("Rate limited by FPL API, waiting...")
                await asyncio.sleep(RATE_LIMIT_DELAY * 10)  # Wait longer for rate limit
                response = await client.get(url)
            
            # Handle other HTTP errors
            if response.status_code != 200:
                raise ValueError(f"FPL API returned {response.status_code}: {response.text}")
            
            data = response.json()
            
            # Validate response structure
            required_keys = ['elements', 'teams', 'events', 'game_settings']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"FPL API response missing keys: {missing_keys}")
            
            logger.info(f"Successfully fetched {len(data['elements'])} players, "
                       f"{len(data['teams'])} teams, {len(data['events'])} events")
            
            return data
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching FPL bootstrap data: {e}")
            raise Exception(f"Failed to connect to FPL API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching FPL bootstrap data: {e}")
            raise


async def get_player_data(player_id: int) -> Dict[str, Any]:
    """
    Pure function to get detailed player data including history.
    
    Args:
        player_id: FPL player ID
        
    Returns:
        Dictionary with player details and history
        
    Raises:
        ValueError: If player_id is invalid or API returns error
        Exception: If network or parsing errors occur
    """
    if not player_id or player_id <= 0:
        raise ValueError("Player ID must be a positive integer")
    
    url = f"{FPL_BASE_URL}/element-summary/{player_id}/"
    
    logger.info(f"Fetching detailed data for player {player_id}")
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            # Add small delay to respect rate limits
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
            response = await client.get(url)
            
            if response.status_code == 404:
                raise ValueError(f"Player {player_id} not found")
            elif response.status_code == 429:
                logger.warning(f"Rate limited for player {player_id}, retrying...")
                await asyncio.sleep(RATE_LIMIT_DELAY * 5)
                response = await client.get(url)
            elif response.status_code != 200:
                raise ValueError(f"FPL API returned {response.status_code} for player {player_id}")
            
            data = response.json()
            
            # Validate response structure
            required_keys = ['fixtures', 'history', 'history_past']
            missing_keys = [key for key in data if key not in required_keys]
            
            logger.info(f"Successfully fetched data for player {player_id}: "
                       f"{len(data.get('history', []))} gameweek records")
            
            return data
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching player {player_id} data: {e}")
            raise Exception(f"Failed to connect to FPL API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching player {player_id} data: {e}")
            raise


async def get_team_picks(team_id: int, gameweek: int) -> Dict[str, Any]:
    """
    Pure function to get team picks for a specific gameweek.
    
    Args:
        team_id: FPL manager team ID
        gameweek: Gameweek number (1-38)
        
    Returns:
        Dictionary with team picks and transfer data
        
    Raises:
        ValueError: If team_id or gameweek are invalid
        Exception: If network or parsing errors occur
    """
    if not team_id or team_id <= 0:
        raise ValueError("Team ID must be a positive integer")
    if not (1 <= gameweek <= 38):
        raise ValueError("Gameweek must be between 1 and 38")
    
    url = f"{FPL_BASE_URL}/entry/{team_id}/event/{gameweek}/picks/"
    
    logger.info(f"Fetching team {team_id} picks for gameweek {gameweek}")
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
            response = await client.get(url)
            
            if response.status_code == 404:
                raise ValueError(f"Team {team_id} not found or gameweek {gameweek} not accessible")
            elif response.status_code == 429:
                logger.warning(f"Rate limited for team {team_id}, retrying...")
                await asyncio.sleep(RATE_LIMIT_DELAY * 5)
                response = await client.get(url)
            elif response.status_code != 200:
                raise ValueError(f"FPL API returned {response.status_code} for team {team_id}")
            
            data = response.json()
            
            # Validate response structure
            if 'picks' not in data:
                raise ValueError("Invalid team picks response - missing 'picks' field")
            
            logger.info(f"Successfully fetched {len(data['picks'])} picks for team {team_id}")
            
            return data
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching team {team_id} picks: {e}")
            raise Exception(f"Failed to connect to FPL API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching team {team_id} picks: {e}")
            raise


async def get_fixtures() -> List[Dict[str, Any]]:
    """
    Pure function to get all FPL fixtures.
    
    Returns:
        List of fixture dictionaries
        
    Raises:
        ValueError: If API request fails
        Exception: If network or parsing errors occur
    """
    url = f"{FPL_BASE_URL}/fixtures/"
    
    logger.info("Fetching FPL fixtures")
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
            response = await client.get(url)
            
            if response.status_code == 429:
                logger.warning("Rate limited fetching fixtures, retrying...")
                await asyncio.sleep(RATE_LIMIT_DELAY * 5)
                response = await client.get(url)
            elif response.status_code != 200:
                raise ValueError(f"FPL API returned {response.status_code} for fixtures")
            
            fixtures = response.json()
            
            if not isinstance(fixtures, list):
                raise ValueError("Invalid fixtures response - expected list")
            
            logger.info(f"Successfully fetched {len(fixtures)} fixtures")
            
            return fixtures
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching fixtures: {e}")
            raise Exception(f"Failed to connect to FPL API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            raise


async def get_live_gameweek_data(gameweek: int) -> Dict[str, Any]:
    """
    Pure function to get live gameweek data with player performances.
    
    Args:
        gameweek: Gameweek number (1-38)
        
    Returns:
        Dictionary with live gameweek data
        
    Raises:
        ValueError: If gameweek is invalid or API returns error
        Exception: If network or parsing errors occur
    """
    if not (1 <= gameweek <= 38):
        raise ValueError("Gameweek must be between 1 and 38")
    
    url = f"{FPL_BASE_URL}/event/{gameweek}/live/"
    
    logger.info(f"Fetching live data for gameweek {gameweek}")
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
            response = await client.get(url)
            
            if response.status_code == 404:
                raise ValueError(f"Gameweek {gameweek} live data not found")
            elif response.status_code == 429:
                logger.warning(f"Rate limited for gameweek {gameweek} live data, retrying...")
                await asyncio.sleep(RATE_LIMIT_DELAY * 5)
                response = await client.get(url)
            elif response.status_code != 200:
                raise ValueError(f"FPL API returned {response.status_code} for gameweek {gameweek}")
            
            data = response.json()
            
            # Validate response structure
            if 'elements' not in data:
                raise ValueError("Invalid live data response - missing 'elements' field")
            
            logger.info(f"Successfully fetched live data for gameweek {gameweek}: "
                       f"{len(data['elements'])} player performances")
            
            return data
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching gameweek {gameweek} live data: {e}")
            raise Exception(f"Failed to connect to FPL API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching gameweek {gameweek} live data: {e}")
            raise


async def get_manager_history(team_id: int) -> Dict[str, Any]:
    """
    Pure function to get manager's historical performance.
    
    Args:
        team_id: FPL manager team ID
        
    Returns:
        Dictionary with manager history and performance data
        
    Raises:
        ValueError: If team_id is invalid or API returns error
        Exception: If network or parsing errors occur
    """
    if not team_id or team_id <= 0:
        raise ValueError("Team ID must be a positive integer")
    
    url = f"{FPL_BASE_URL}/entry/{team_id}/history/"
    
    logger.info(f"Fetching history for manager {team_id}")
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
            response = await client.get(url)
            
            if response.status_code == 404:
                raise ValueError(f"Manager {team_id} not found")
            elif response.status_code == 429:
                logger.warning(f"Rate limited for manager {team_id} history, retrying...")
                await asyncio.sleep(RATE_LIMIT_DELAY * 5)
                response = await client.get(url)
            elif response.status_code != 200:
                raise ValueError(f"FPL API returned {response.status_code} for manager {team_id}")
            
            data = response.json()
            
            # Validate response structure
            required_keys = ['current', 'past', 'chips']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                logger.warning(f"Manager history missing some keys: {missing_keys}")
            
            logger.info(f"Successfully fetched history for manager {team_id}: "
                       f"{len(data.get('current', []))} current season records")
            
            return data
            
        except httpx.RequestError as e:
            logger.error(f"Network error fetching manager {team_id} history: {e}")
            raise Exception(f"Failed to connect to FPL API: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching manager {team_id} history: {e}")
            raise


async def batch_fetch_player_data(player_ids: List[int], max_concurrent: int = 5) -> Dict[int, Dict[str, Any]]:
    """
    Pure function to batch fetch multiple players' data with concurrency control.
    
    Args:
        player_ids: List of FPL player IDs
        max_concurrent: Maximum concurrent requests
        
    Returns:
        Dictionary mapping player_id to player data
        
    Raises:
        ValueError: If player_ids is empty or contains invalid IDs
        Exception: If network errors occur
    """
    if not player_ids:
        raise ValueError("Player IDs list cannot be empty")
    
    invalid_ids = [pid for pid in player_ids if not isinstance(pid, int) or pid <= 0]
    if invalid_ids:
        raise ValueError(f"Invalid player IDs: {invalid_ids}")
    
    logger.info(f"Batch fetching data for {len(player_ids)} players")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    
    async def fetch_single_player(player_id: int) -> None:
        async with semaphore:
            try:
                data = await get_player_data(player_id)
                results[player_id] = data
            except Exception as e:
                logger.error(f"Failed to fetch data for player {player_id}: {e}")
                results[player_id] = {"error": str(e)}
    
    # Execute all requests concurrently with semaphore limiting
    tasks = [fetch_single_player(pid) for pid in player_ids]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_fetches = len([r for r in results.values() if "error" not in r])
    logger.info(f"Successfully fetched data for {successful_fetches}/{len(player_ids)} players")
    
    return results