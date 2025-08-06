"""
Unit tests for data fetchers (data/fetchers.py).
Target: 90%+ coverage with comprehensive FPL API client testing.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timedelta
from pathlib import Path
import httpx
import pandas as pd

from src.data.fetchers import FPLDataFetcher


@pytest.mark.unit
class TestFPLDataFetcher:
    """Comprehensive unit tests for FPLDataFetcher."""
    
    def test_fpl_data_fetcher_initialization(self):
        """Test FPLDataFetcher initialization."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://fantasy.premierleague.com/api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                fetcher = FPLDataFetcher()
                
                assert fetcher.base_url == "https://fantasy.premierleague.com/api"
                assert fetcher.cache_ttl == 900
                assert fetcher.session is None
                assert fetcher._authenticated is False
                
                # Check cache directory creation
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test FPLDataFetcher async context manager."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                with patch('httpx.AsyncClient') as mock_async_client:
                    mock_client = AsyncMock()
                    mock_async_client.return_value = mock_client
                    
                    async with FPLDataFetcher() as fetcher:
                        assert fetcher.session is not None
                        mock_async_client.assert_called_with(timeout=30.0)
                    
                    # Should close session on exit
                    mock_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager_exception(self):
        """Test async context manager handles exceptions properly."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                with patch('httpx.AsyncClient') as mock_async_client:
                    mock_client = AsyncMock()
                    mock_async_client.return_value = mock_client
                    
                    try:
                        async with FPLDataFetcher() as fetcher:
                            raise ValueError("Test exception")
                    except ValueError:
                        pass  # Expected
                    
                    # Should still close session on exception
                    mock_client.aclose.assert_called_once()
    
    def test_cache_file_path_generation(self):
        """Test cache file path generation."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                # Test cache path generation
                cache_path = fetcher._get_cache_path("bootstrap_data")
                expected_path = fetcher.cache_dir / "bootstrap_data.json"
                assert cache_path == expected_path
                
                # Test with complex endpoint
                cache_path = fetcher._get_cache_path("player_123_history")
                expected_path = fetcher.cache_dir / "player_123_history.json"
                assert cache_path == expected_path
    
    def test_cache_validation_fresh(self):
        """Test cache validation for fresh cache."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900  # 15 minutes
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                # Mock file that exists and is fresh
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                
                # File modified 10 minutes ago (fresh)
                recent_time = datetime.now() - timedelta(minutes=10)
                mock_path.stat.return_value.st_mtime = recent_time.timestamp()
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    is_fresh = fetcher._is_cache_fresh("test_endpoint")
                    assert is_fresh is True
    
    def test_cache_validation_stale(self):
        """Test cache validation for stale cache."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900  # 15 minutes
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                # Mock file that exists but is stale
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                
                # File modified 20 minutes ago (stale)
                old_time = datetime.now() - timedelta(minutes=20)
                mock_path.stat.return_value.st_mtime = old_time.timestamp()
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    is_fresh = fetcher._is_cache_fresh("test_endpoint")
                    assert is_fresh is False
    
    def test_cache_validation_missing(self):
        """Test cache validation for missing cache file."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                # Mock file that doesn't exist
                mock_path = MagicMock()
                mock_path.exists.return_value = False
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    is_fresh = fetcher._is_cache_fresh("test_endpoint")
                    assert is_fresh is False
    
    def test_load_cached_data_success(self):
        """Test loading cached data successfully."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                test_data = {"test": "data", "number": 123}
                mock_path = MagicMock()
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    with patch('builtins.open', mock_open(read_data=json.dumps(test_data))):
                        loaded_data = fetcher._load_cached_data("test_endpoint")
                        assert loaded_data == test_data
    
    def test_load_cached_data_json_error(self):
        """Test loading cached data with JSON decode error."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                mock_path = MagicMock()
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    with patch('builtins.open', mock_open(read_data="invalid json")):
                        with patch('src.data.fetchers.logger') as mock_logger:
                            loaded_data = fetcher._load_cached_data("test_endpoint")
                            assert loaded_data is None
                            mock_logger.warning.assert_called()
    
    def test_load_cached_data_file_not_found(self):
        """Test loading cached data when file doesn't exist."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                mock_path = MagicMock()
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    with patch('builtins.open', side_effect=FileNotFoundError()):
                        loaded_data = fetcher._load_cached_data("test_endpoint")
                        assert loaded_data is None
    
    def test_save_cached_data(self):
        """Test saving data to cache."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                test_data = {"test": "data", "number": 123}
                mock_path = MagicMock()
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    with patch('builtins.open', mock_open()) as mock_file:
                        fetcher._save_cached_data("test_endpoint", test_data)
                        
                        # Check that file was written with correct data
                        mock_file.assert_called_with(mock_path, 'w')
                        written_data = ''.join([
                            call.args[0] for call in mock_file().write.call_args_list
                        ])
                        assert json.loads(written_data) == test_data
    
    def test_save_cached_data_error(self):
        """Test saving data to cache with write error."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                test_data = {"test": "data"}
                mock_path = MagicMock()
                
                with patch.object(fetcher, '_get_cache_path', return_value=mock_path):
                    with patch('builtins.open', side_effect=IOError("Write error")):
                        with patch('src.data.fetchers.logger') as mock_logger:
                            fetcher._save_cached_data("test_endpoint", test_data)
                            mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_fetch_with_cache_hit(self):
        """Test fetch with cache hit (returns cached data)."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                cached_data = {"cached": True, "data": "test"}
                
                # Mock cache methods
                with patch.object(fetcher, '_is_cache_fresh', return_value=True):
                    with patch.object(fetcher, '_load_cached_data', return_value=cached_data):
                        with patch('src.data.fetchers.get_bootstrap_data') as mock_api:
                            result = await fetcher.fetch_bootstrap_data()
                            
                            assert result == cached_data
                            # Should not call API since cache is fresh
                            mock_api.assert_not_called()
    
    @pytest.mark.asyncio 
    async def test_fetch_with_cache_miss(self):
        """Test fetch with cache miss (fetches from API)."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                api_data = {"fresh": True, "data": "from_api"}
                
                # Mock cache methods (cache miss)
                with patch.object(fetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(fetcher, '_save_cached_data') as mock_save:
                        with patch('src.data.fetchers.get_bootstrap_data', return_value=api_data):
                            result = await fetcher.fetch_bootstrap_data()
                            
                            assert result == api_data
                            # Should save to cache
                            mock_save.assert_called_with('bootstrap_data', api_data)
    
    @pytest.mark.asyncio
    async def test_fetch_bootstrap_data(self):
        """Test fetching bootstrap data."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                mock_data = {
                    "elements": [{"id": 1, "web_name": "Player 1"}],
                    "teams": [{"id": 1, "name": "Team 1"}],
                    "events": [{"id": 1, "name": "GW 1"}]
                }
                
                with patch.object(fetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(fetcher, '_save_cached_data'):
                        with patch('src.data.fetchers.get_bootstrap_data', return_value=mock_data):
                            result = await fetcher.fetch_bootstrap_data()
                            assert result == mock_data
    
    @pytest.mark.asyncio
    async def test_fetch_bootstrap_data_with_force_refresh(self):
        """Test fetching bootstrap data with force refresh."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                api_data = {"fresh": True, "data": "forced_refresh"}
                
                # Even if cache is fresh, should bypass with force_refresh=True
                with patch.object(fetcher, '_is_cache_fresh', return_value=True):
                    with patch.object(fetcher, '_save_cached_data') as mock_save:
                        with patch('src.data.fetchers.get_bootstrap_data', return_value=api_data):
                            result = await fetcher.fetch_bootstrap_data(force_refresh=True)
                            
                            assert result == api_data
                            # Should save to cache
                            mock_save.assert_called_with('bootstrap_data', api_data)
    
    @pytest.mark.asyncio
    async def test_fetch_player_data(self):
        """Test fetching player data."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                player_id = 123
                mock_data = {
                    "history": [{"total_points": 10, "round": 1}],
                    "fixtures": [{"event": 20, "is_home": True}]
                }
                
                with patch.object(fetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(fetcher, '_save_cached_data'):
                        with patch('src.data.fetchers.get_player_data', return_value=mock_data):
                            result = await fetcher.fetch_player_data(player_id)
                            assert result == mock_data
    
    @pytest.mark.asyncio
    async def test_fetch_team_picks(self):
        """Test fetching team picks."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                team_id = 456
                gameweek = 20
                mock_data = {
                    "picks": [{"element": 1, "position": 1, "is_captain": True}],
                    "entry_history": {"event": 20, "points": 67}
                }
                
                with patch.object(fetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(fetcher, '_save_cached_data'):
                        with patch('src.data.fetchers.get_team_picks', return_value=mock_data):
                            result = await fetcher.fetch_team_picks(team_id, gameweek)
                            assert result == mock_data
    
    @pytest.mark.asyncio
    async def test_fetch_fixtures(self):
        """Test fetching fixtures."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                mock_data = [
                    {"id": 1, "team_h": 1, "team_a": 2, "event": 20},
                    {"id": 2, "team_h": 3, "team_a": 4, "event": 20}
                ]
                
                with patch.object(fetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(fetcher, '_save_cached_data'):
                        with patch('src.data.fetchers.get_fixtures', return_value=mock_data):
                            result = await fetcher.fetch_fixtures()
                            assert result == mock_data
    
    @pytest.mark.asyncio
    async def test_fetch_live_gameweek_data(self):
        """Test fetching live gameweek data."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                gameweek = 20
                mock_data = {
                    "elements": [
                        {"id": 1, "stats": {"total_points": 8}},
                        {"id": 2, "stats": {"total_points": 12}}
                    ]
                }
                
                with patch.object(fetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(fetcher, '_save_cached_data'):
                        with patch('src.data.fetchers.get_live_gameweek_data', return_value=mock_data):
                            result = await fetcher.fetch_live_gameweek_data(gameweek)
                            assert result == mock_data
    
    @pytest.mark.asyncio
    async def test_fetch_manager_history(self):
        """Test fetching manager history."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                manager_id = 789
                mock_data = {
                    "current": [{"event": 1, "points": 65, "total_points": 65}],
                    "past": [{"season_name": "2022/23", "total_points": 2150}]
                }
                
                with patch.object(fetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(fetcher, '_save_cached_data'):
                        with patch('src.data.fetchers.get_manager_history', return_value=mock_data):
                            result = await fetcher.fetch_manager_history(manager_id)
                            assert result == mock_data
    
    def test_parse_players_from_bootstrap(self):
        """Test parsing players from bootstrap data."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                bootstrap_data = {
                    "elements": [
                        {
                            "id": 1,
                            "first_name": "Mohamed",
                            "second_name": "Salah",
                            "web_name": "Salah",
                            "element_type": 3,
                            "team": 1,
                            "now_cost": 130,
                            "total_points": 187
                        },
                        {
                            "id": 2,
                            "first_name": "Harry",
                            "second_name": "Kane", 
                            "web_name": "Kane",
                            "element_type": 4,
                            "team": 2,
                            "now_cost": 110,
                            "total_points": 156
                        }
                    ]
                }
                
                players = fetcher.parse_players_from_bootstrap(bootstrap_data)
                
                assert len(players) == 2
                assert all(isinstance(player, Player) for player in players)
                assert players[0].web_name == "Salah"
                assert players[0].now_cost == 130
                assert players[1].web_name == "Kane"
                assert players[1].now_cost == 110
    
    def test_parse_teams_from_bootstrap(self):
        """Test parsing teams from bootstrap data."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                bootstrap_data = {
                    "teams": [
                        {
                            "id": 1,
                            "name": "Liverpool",
                            "short_name": "LIV",
                            "strength": 5
                        },
                        {
                            "id": 2,
                            "name": "Manchester City",
                            "short_name": "MCI", 
                            "strength": 5
                        }
                    ]
                }
                
                teams = fetcher.parse_teams_from_bootstrap(bootstrap_data)
                
                assert len(teams) == 2
                assert all(isinstance(team, Team) for team in teams)
                assert teams[0].name == "Liverpool"
                assert teams[0].short_name == "LIV"
                assert teams[1].name == "Manchester City"
                assert teams[1].short_name == "MCI"
    
    def test_parse_empty_data(self):
        """Test parsing empty or invalid data."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                # Test empty bootstrap data
                empty_data = {"elements": [], "teams": []}
                players = fetcher.parse_players_from_bootstrap(empty_data)
                teams = fetcher.parse_teams_from_bootstrap(empty_data)
                
                assert players == []
                assert teams == []
                
                # Test missing keys
                invalid_data = {}
                with patch('src.data.fetchers.logger') as mock_logger:
                    players = fetcher.parse_players_from_bootstrap(invalid_data)
                    teams = fetcher.parse_teams_from_bootstrap(invalid_data)
                    
                    assert players == []
                    assert teams == []
                    mock_logger.error.assert_called()


@pytest.mark.unit
class TestFPLDataFetcherIntegration:
    """Integration tests for FPLDataFetcher with external dependencies."""
    
    @pytest.mark.asyncio
    async def test_full_data_fetch_workflow(self):
        """Test complete data fetching workflow."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                bootstrap_data = {
                    "elements": [
                        {
                            "id": 1, "first_name": "Test", "second_name": "Player",
                            "web_name": "Player", "element_type": 3, "team": 1,
                            "now_cost": 80, "total_points": 100
                        }
                    ],
                    "teams": [
                        {"id": 1, "name": "Test FC", "short_name": "TST", "strength": 3}
                    ],
                    "events": [
                        {"id": 20, "name": "Gameweek 20", "is_current": True}
                    ]
                }
                
                # Mock all the cache operations to force API calls
                with patch.object(FPLDataFetcher, '_is_cache_fresh', return_value=False):
                    with patch.object(FPLDataFetcher, '_save_cached_data'):
                        with patch('src.data.fetchers.get_bootstrap_data', return_value=bootstrap_data):
                            async with FPLDataFetcher() as fetcher:
                                result = await fetcher.fetch_bootstrap_data()
                                
                                assert result == bootstrap_data
                                
                                # Test parsing
                                players = fetcher.parse_players_from_bootstrap(result)
                                teams = fetcher.parse_teams_from_bootstrap(result)
                                
                                assert len(players) == 1
                                assert len(teams) == 1
                                assert players[0].web_name == "Player"
                                assert teams[0].name == "Test FC"
    
    @pytest.mark.asyncio
    async def test_error_handling_network_issues(self):
        """Test error handling for network issues."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                # Mock network error
                with patch.object(FPLDataFetcher, '_is_cache_fresh', return_value=False):
                    with patch('src.data.fetchers.get_bootstrap_data', 
                              side_effect=httpx.TimeoutException("Network timeout")):
                        with patch('src.data.fetchers.logger') as mock_logger:
                            async with FPLDataFetcher() as fetcher:
                                with pytest.raises(httpx.TimeoutException):
                                    await fetcher.fetch_bootstrap_data()
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access doesn't cause issues."""
        with patch('src.data.fetchers.settings') as mock_settings:
            mock_settings.fpl_api_base_url = "https://test.api"
            mock_settings.cache_ttl = 900
            
            with patch('pathlib.Path.mkdir'):
                fetcher = FPLDataFetcher()
                
                # Simulate concurrent cache operations
                test_data_1 = {"endpoint": "1", "data": "test1"}
                test_data_2 = {"endpoint": "2", "data": "test2"}
                
                with patch('builtins.open', mock_open()):
                    # These should not interfere with each other
                    fetcher._save_cached_data("endpoint_1", test_data_1)
                    fetcher._save_cached_data("endpoint_2", test_data_2)
                    
                    # No exception should be raised