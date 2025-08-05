"""
FPL API client with authentication, caching, and web scraping capabilities.
Implements higher-level data fetching with error handling and rate limiting.
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import httpx
import pandas as pd
from bs4 import BeautifulSoup

from ..config.settings import settings
from ..tools.fpl_api import (
    get_bootstrap_data,
    get_player_data, 
    get_team_picks,
    get_fixtures,
    get_live_gameweek_data,
    get_manager_history,
)
from ..models.data_models import Player, Team, Fixture, Event

logger = logging.getLogger(__name__)


class FPLDataFetcher:
    """
    Comprehensive FPL data fetcher with authentication, caching, and web scraping.
    """
    
    def __init__(self):
        self.base_url = settings.fpl_api_base_url
        self.cache_dir = Path("data/raw/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = settings.cache_ttl
        self.session = None
        self._authenticated = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        if not settings.enable_caching:
            return False
        
        # Check if cache is within TTL
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        return cache_age < self.cache_ttl
    
    def _load_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(cache_key)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded data from cache: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def _save_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        if not settings.enable_caching:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    async def authenticate(self, email: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Authenticate with FPL for accessing private endpoints.
        
        Args:
            email: FPL account email (uses settings if not provided)
            password: FPL account password (uses settings if not provided)
            
        Returns:
            True if authentication successful
        """
        if not self.session:
            raise RuntimeError("Session not initialized - use async context manager")
        
        email = email or settings.fpl_email
        password = password or settings.fpl_password
        
        if not email or not password:
            logger.warning("FPL authentication credentials not provided - using public endpoints only")
            return False
        
        login_url = "https://users.premierleague.com/accounts/login/"
        
        try:
            # Get login page to extract CSRF token
            response = await self.session.get(login_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
            
            # Perform login
            login_data = {
                "login": email,
                "password": password,
                "csrfmiddlewaretoken": csrf_token,
            }
            
            response = await self.session.post(
                login_url,
                data=login_data,
                headers={
                    'Referer': login_url,
                    'User-Agent': 'Mozilla/5.0 (compatible; FPL-ML-System/1.0)',
                }
            )
            
            if response.status_code == 200 and "login" not in response.url.path:
                self._authenticated = True
                logger.info("Successfully authenticated with FPL")
                return True
            else:
                logger.error("FPL authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"FPL authentication error: {e}")
            return False
    
    async def get_all_players(self, use_cache: bool = True) -> List[Player]:
        """
        Get all FPL players with comprehensive data.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            List of Player objects
        """
        cache_key = "all_players"
        
        if use_cache:
            cached_data = self._load_cache(cache_key)
            if cached_data:
                return [Player(**player_data) for player_data in cached_data]
        
        try:
            # Get bootstrap data
            bootstrap = await get_bootstrap_data()
            players_data = bootstrap['elements']
            
            # Convert to Player objects
            players = []
            for player_data in players_data:
                try:
                    player = Player(**player_data)
                    players.append(player)
                except Exception as e:
                    logger.warning(f"Failed to parse player {player_data.get('id', 'unknown')}: {e}")
            
            # Cache the data
            if use_cache:
                self._save_cache(cache_key, [p.dict() for p in players])
            
            logger.info(f"Fetched {len(players)} players")
            return players
            
        except Exception as e:
            logger.error(f"Failed to fetch all players: {e}")
            raise
    
    async def get_all_teams(self, use_cache: bool = True) -> List[Team]:
        """
        Get all FPL teams with strength ratings.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            List of Team objects
        """
        cache_key = "all_teams"
        
        if use_cache:
            cached_data = self._load_cache(cache_key)
            if cached_data:
                return [Team(**team_data) for team_data in cached_data]
        
        try:
            # Get bootstrap data
            bootstrap = await get_bootstrap_data()
            teams_data = bootstrap['teams']
            
            # Convert to Team objects
            teams = []
            for team_data in teams_data:
                try:
                    team = Team(**team_data)
                    teams.append(team)
                except Exception as e:
                    logger.warning(f"Failed to parse team {team_data.get('id', 'unknown')}: {e}")
            
            # Cache the data
            if use_cache:
                self._save_cache(cache_key, [t.dict() for t in teams])
            
            logger.info(f"Fetched {len(teams)} teams")
            return teams
            
        except Exception as e:
            logger.error(f"Failed to fetch all teams: {e}")
            raise
    
    async def get_player_history(self, player_id: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get detailed player history including fixtures and past performance.
        
        Args:
            player_id: FPL player ID
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with player history data
        """
        cache_key = f"player_history_{player_id}"
        
        if use_cache:
            cached_data = self._load_cache(cache_key)
            if cached_data:
                return cached_data
        
        try:
            player_data = await get_player_data(player_id)
            
            # Cache the data
            if use_cache:
                self._save_cache(cache_key, player_data)
            
            return player_data
            
        except Exception as e:
            logger.error(f"Failed to fetch player {player_id} history: {e}")
            raise
    
    async def get_current_gameweek(self) -> Optional[Event]:
        """
        Get current gameweek information.
        
        Returns:
            Current Event object or None if not found
        """
        try:
            bootstrap = await get_bootstrap_data()
            events = bootstrap['events']
            
            # Find current gameweek
            current_event = None
            for event_data in events:
                if event_data.get('is_current', False):
                    current_event = Event(**event_data)
                    break
            
            if not current_event:
                # If no current event, find next one
                for event_data in events:
                    if event_data.get('is_next', False):
                        current_event = Event(**event_data)
                        break
            
            return current_event
            
        except Exception as e:
            logger.error(f"Failed to get current gameweek: {e}")
            return None
    
    async def get_injury_news(self) -> List[Dict[str, Any]]:
        """
        Scrape injury news from external sources.
        
        Returns:
            List of injury news dictionaries
        """
        injury_news = []
        
        try:
            # Scrape from FPL website news section
            news_url = "https://www.premierleague.com/news"
            
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            response = await self.session.get(news_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract injury-related news (this is a simplified example)
            news_items = soup.find_all('article', class_='newsListing')
            
            for item in news_items[:10]:  # Limit to recent news
                title_elem = item.find('h3')
                link_elem = item.find('a')
                
                if title_elem and link_elem:
                    title = title_elem.get_text().strip()
                    link = link_elem.get('href', '')
                    
                    # Simple keyword matching for injury news
                    injury_keywords = ['injury', 'injured', 'doubt', 'fitness', 'ruled out', 'sidelined']
                    if any(keyword in title.lower() for keyword in injury_keywords):
                        injury_news.append({
                            'title': title,
                            'link': link,
                            'source': 'Premier League',
                            'scraped_at': datetime.now().isoformat(),
                        })
            
            logger.info(f"Scraped {len(injury_news)} injury news items")
            return injury_news
            
        except Exception as e:
            logger.error(f"Failed to scrape injury news: {e}")
            return []
    
    async def get_team_performance_data(self, team_id: int, gameweeks: int = 10) -> pd.DataFrame:
        """
        Get team performance data for analysis.
        
        Args:
            team_id: FPL manager team ID
            gameweeks: Number of recent gameweeks to fetch
            
        Returns:
            DataFrame with team performance data
        """
        try:
            history_data = await get_manager_history(team_id)
            current_season = history_data.get('current', [])
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(current_season)
            
            if len(df) > gameweeks:
                df = df.tail(gameweeks)
            
            # Add derived metrics
            if not df.empty:
                df['points_per_gameweek'] = df['points']
                df['rank_change'] = df['overall_rank'].diff()
                df['points_on_bench'] = df['points_on_bench']
                df['total_transfers'] = df['event_transfers']
            
            logger.info(f"Fetched performance data for team {team_id}: {len(df)} gameweeks")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get team {team_id} performance data: {e}")
            return pd.DataFrame()
    
    async def get_price_change_data(self) -> Dict[int, Dict[str, float]]:
        """
        Get player price change probabilities (would require external service).
        
        Returns:
            Dictionary mapping player_id to price change probabilities
        """
        # This is a placeholder - in reality, you'd integrate with services like
        # FPL Review, Fantasy Football Fix, or build your own price prediction model
        
        try:
            players = await self.get_all_players(use_cache=True)
            price_changes = {}
            
            for player in players:
                # Simple heuristic based on transfers and ownership
                transfers_in = getattr(player, 'transfers_in_event', 0)
                transfers_out = getattr(player, 'transfers_out_event', 0)
                net_transfers = transfers_in - transfers_out
                
                # Basic probability calculation (would be much more sophisticated in reality)
                if net_transfers > 50000:
                    rise_prob = 0.7
                    fall_prob = 0.1
                elif net_transfers < -50000:
                    rise_prob = 0.1
                    fall_prob = 0.7
                else:
                    rise_prob = 0.3
                    fall_prob = 0.3
                
                price_changes[player.id] = {
                    'rise_probability': rise_prob,
                    'fall_probability': fall_prob,
                    'net_transfers': net_transfers,
                }
            
            logger.info(f"Generated price change data for {len(price_changes)} players")
            return price_changes
            
        except Exception as e:
            logger.error(f"Failed to get price change data: {e}")
            return {}
    
    async def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            for cache_file in cache_files:
                cache_file.unlink()
            logger.info(f"Cleared {len(cache_files)} cache files")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            cache_info = {
                'total_files': len(cache_files),
                'total_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
                'files': []
            }
            
            for cache_file in cache_files:
                stat = cache_file.stat()
                cache_info['files'].append({
                    'name': cache_file.stem,
                    'size_kb': stat.st_size / 1024,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'age_seconds': datetime.now().timestamp() - stat.st_mtime,
                })
            
            return cache_info
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {'error': str(e)}