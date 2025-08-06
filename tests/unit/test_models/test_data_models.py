"""
Unit tests for Pydantic data models (data_models.py).
Target: 90%+ coverage with comprehensive validation testing.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models.data_models import (
    ElementType, ChipType, Player, Team, Fixture
)


@pytest.mark.unit
class TestElementType:
    """Unit tests for ElementType enum."""
    
    def test_element_type_values(self):
        """Test ElementType enum has correct values."""
        assert ElementType.GOALKEEPER == 1
        assert ElementType.DEFENDER == 2
        assert ElementType.MIDFIELDER == 3
        assert ElementType.FORWARD == 4
    
    def test_element_type_is_int_enum(self):
        """Test ElementType inherits from int for FPL API compatibility."""
        assert isinstance(ElementType.GOALKEEPER, int)
        assert ElementType.GOALKEEPER == 1
        assert int(ElementType.MIDFIELDER) == 3


@pytest.mark.unit
class TestChipType:
    """Unit tests for ChipType enum."""
    
    def test_chip_type_values(self):
        """Test ChipType enum has correct string values."""
        assert ChipType.WILDCARD == "wildcard"
        assert ChipType.FREE_HIT == "freehit"
        assert ChipType.BENCH_BOOST == "bboost"
        assert ChipType.TRIPLE_CAPTAIN == "3xc"
    
    def test_chip_type_is_string_enum(self):
        """Test ChipType inherits from str for API compatibility."""
        assert isinstance(ChipType.WILDCARD, str)
        assert str(ChipType.FREE_HIT) == "freehit"


@pytest.mark.unit
class TestPlayer:
    """Comprehensive unit tests for Player model."""
    
    def test_player_minimal_valid_data(self):
        """Test Player creation with minimal valid data."""
        player_data = {
            'id': 1,
            'first_name': 'Mohamed',
            'second_name': 'Salah',
            'web_name': 'Salah',
            'element_type': ElementType.FORWARD,
            'team': 1,
            'now_cost': 130  # £13.0M
        }
        
        player = Player(**player_data)
        
        assert player.id == 1
        assert player.first_name == 'Mohamed'
        assert player.second_name == 'Salah'
        assert player.web_name == 'Salah'
        assert player.element_type == ElementType.FORWARD
        assert player.team == 1
        assert player.now_cost == 130
        
        # Test default values
        assert player.total_points == 0
        assert player.points_per_game == 0
        assert player.form == "0.0"
        assert player.minutes == 0
        assert player.goals_scored == 0
        assert player.assists == 0
        assert player.clean_sheets == 0
        assert player.saves == 0
        assert player.bonus == 0
        assert player.bps == 0
        assert player.news == ""
        assert player.chance_of_playing_this_round is None
    
    def test_player_complete_data(self):
        """Test Player creation with complete data."""
        player_data = {
            'id': 2,
            'first_name': 'Harry',
            'second_name': 'Kane',
            'web_name': 'Kane',
            'element_type': ElementType.FORWARD,
            'team': 2,
            'now_cost': 110,
            'cost_change_event': 1,
            'cost_change_start': 5,
            'total_points': 187,
            'points_per_game': 6.2,
            'form': "8.5",
            'minutes': 2430,
            'goals_scored': 18,
            'assists': 7,
            'clean_sheets': 0,
            'goals_conceded': 0,
            'saves': 0,
            'bonus': 24,
            'bps': 456,
            'influence': "98.4",
            'creativity': "67.2",
            'threat': "156.8",
            'ict_index': "322.4",
            'expected_goals': "16.83",
            'expected_assists': "5.42",
            'expected_goal_involvements': "22.25",
            'expected_goals_conceded': "0.0",
            'selected_by_percent': "28.7",
            'transfers_in_event': 45123,
            'transfers_out_event': 12456,
            'chance_of_playing_this_round': 100,
            'chance_of_playing_next_round': 100,
            'news': 'No issues reported',
            'news_added': datetime(2024, 1, 15, 10, 30)
        }
        
        player = Player(**player_data)
        
        assert player.total_points == 187
        assert player.goals_scored == 18
        assert player.assists == 7
        assert player.influence == "98.4"
        assert player.selected_by_percent == "28.7"
        assert player.chance_of_playing_this_round == 100
        assert player.news == 'No issues reported'
    
    def test_player_price_validation_valid_range(self):
        """Test player price validation for valid ranges."""
        # Test minimum valid price (£3.0M = 30)
        player_data = {
            'id': 1, 'first_name': 'Test', 'second_name': 'Player',
            'web_name': 'Test', 'element_type': 1, 'team': 1,
            'now_cost': 30  # Minimum valid price
        }
        player = Player(**player_data)
        assert player.now_cost == 30
        
        # Test maximum valid price (£15.0M = 150)
        player_data['now_cost'] = 150  # Maximum valid price
        player = Player(**player_data)
        assert player.now_cost == 150
    
    def test_player_price_validation_invalid_range(self):
        """Test player price validation fails for invalid ranges."""
        player_data = {
            'id': 1, 'first_name': 'Test', 'second_name': 'Player',
            'web_name': 'Test', 'element_type': 1, 'team': 1,
            'now_cost': 29  # Below minimum
        }
        
        with pytest.raises(ValidationError, match="Player price must be between £3.0M and £15.0M"):
            Player(**player_data)
        
        player_data['now_cost'] = 151  # Above maximum
        with pytest.raises(ValidationError, match="Player price must be between £3.0M and £15.0M"):
            Player(**player_data)
    
    def test_player_negative_values_validation(self):
        """Test validation fails for negative values where not allowed."""
        player_data = {
            'id': 1, 'first_name': 'Test', 'second_name': 'Player',
            'web_name': 'Test', 'element_type': 1, 'team': 1,
            'now_cost': 100, 'total_points': -5
        }
        
        with pytest.raises(ValidationError):
            Player(**player_data)
        
        player_data['total_points'] = 0
        player_data['minutes'] = -10
        
        with pytest.raises(ValidationError):
            Player(**player_data)
    
    def test_player_chance_of_playing_validation(self):
        """Test chance of playing validation (0-100)."""
        player_data = {
            'id': 1, 'first_name': 'Test', 'second_name': 'Player',
            'web_name': 'Test', 'element_type': 1, 'team': 1,
            'now_cost': 100, 'chance_of_playing_this_round': 101
        }
        
        with pytest.raises(ValidationError):
            Player(**player_data)
        
        player_data['chance_of_playing_this_round'] = -1
        with pytest.raises(ValidationError):
            Player(**player_data)
        
        # Valid range should work
        player_data['chance_of_playing_this_round'] = 75
        player = Player(**player_data)
        assert player.chance_of_playing_this_round == 75
    
    def test_player_price_millions_property(self):
        """Test price_millions property calculation."""
        player_data = {
            'id': 1, 'first_name': 'Test', 'second_name': 'Player',
            'web_name': 'Test', 'element_type': 1, 'team': 1,
            'now_cost': 130  # £13.0M
        }
        
        player = Player(**player_data)
        assert player.price_millions == 13.0
        
        player_data['now_cost'] = 85  # £8.5M
        player = Player(**player_data)
        assert player.price_millions == 8.5
    
    def test_player_position_name_property(self):
        """Test position_name property returns correct strings."""
        player_data = {
            'id': 1, 'first_name': 'Test', 'second_name': 'Player',
            'web_name': 'Test', 'team': 1, 'now_cost': 100
        }
        
        # Test all positions
        player_data['element_type'] = ElementType.GOALKEEPER
        player = Player(**player_data)
        assert player.position_name == "GK"
        
        player_data['element_type'] = ElementType.DEFENDER
        player = Player(**player_data)
        assert player.position_name == "DEF"
        
        player_data['element_type'] = ElementType.MIDFIELDER
        player = Player(**player_data)
        assert player.position_name == "MID"
        
        player_data['element_type'] = ElementType.FORWARD
        player = Player(**player_data)
        assert player.position_name == "FWD"
    
    def test_player_required_fields_missing(self):
        """Test validation fails when required fields are missing."""
        # Missing id
        with pytest.raises(ValidationError):
            Player(first_name='Test', second_name='Player', web_name='Test', 
                  element_type=1, team=1, now_cost=100)
        
        # Missing element_type
        with pytest.raises(ValidationError):
            Player(id=1, first_name='Test', second_name='Player', web_name='Test', 
                  team=1, now_cost=100)
        
        # Missing team
        with pytest.raises(ValidationError):
            Player(id=1, first_name='Test', second_name='Player', web_name='Test', 
                  element_type=1, now_cost=100)
    
    def test_player_element_type_conversion(self):
        """Test ElementType accepts both int and enum values."""
        player_data = {
            'id': 1, 'first_name': 'Test', 'second_name': 'Player',
            'web_name': 'Test', 'team': 1, 'now_cost': 100
        }
        
        # Test with int value
        player_data['element_type'] = 3
        player = Player(**player_data)
        assert player.element_type == ElementType.MIDFIELDER
        assert player.position_name == "MID"
        
        # Test with enum value
        player_data['element_type'] = ElementType.FORWARD
        player = Player(**player_data)
        assert player.element_type == ElementType.FORWARD
        assert player.position_name == "FWD"


@pytest.mark.unit
class TestTeam:
    """Comprehensive unit tests for Team model."""
    
    def test_team_minimal_valid_data(self):
        """Test Team creation with minimal valid data."""
        team_data = {
            'id': 1,
            'name': 'Arsenal',
            'short_name': 'ARS'
        }
        
        team = Team(**team_data)
        
        assert team.id == 1
        assert team.name == 'Arsenal'
        assert team.short_name == 'ARS'
        
        # Test default values
        assert team.played == 0
        assert team.wins == 0
        assert team.draws == 0
        assert team.losses == 0
        assert team.points == 0
        assert team.position == 1
        assert team.strength == 3
        assert team.strength_overall_home == 3
        assert team.strength_overall_away == 3
    
    def test_team_complete_data(self):
        """Test Team creation with complete data."""
        team_data = {
            'id': 1,
            'name': 'Manchester City',
            'short_name': 'MCI',
            'played': 20,
            'wins': 15,
            'draws': 3,
            'losses': 2,
            'points': 48,
            'position': 2,
            'strength': 5,
            'strength_overall_home': 5,
            'strength_overall_away': 4,
            'strength_attack_home': 5,
            'strength_attack_away': 4,
            'strength_defence_home': 4,
            'strength_defence_away': 3
        }
        
        team = Team(**team_data)
        
        assert team.played == 20
        assert team.wins == 15
        assert team.points == 48
        assert team.position == 2
        assert team.strength == 5
        assert team.strength_attack_home == 5
        assert team.strength_defence_away == 3
    
    def test_team_strength_validation_valid_range(self):
        """Test team strength validation for valid range (1-5)."""
        team_data = {
            'id': 1, 'name': 'Test Team', 'short_name': 'TST',
            'strength': 1  # Minimum valid
        }
        team = Team(**team_data)
        assert team.strength == 1
        
        team_data['strength'] = 5  # Maximum valid
        team = Team(**team_data)
        assert team.strength == 5
    
    def test_team_strength_validation_invalid_range(self):
        """Test team strength validation fails for invalid range."""
        team_data = {
            'id': 1, 'name': 'Test Team', 'short_name': 'TST',
            'strength': 0  # Below minimum
        }
        
        with pytest.raises(ValidationError):
            Team(**team_data)
        
        team_data['strength'] = 6  # Above maximum
        with pytest.raises(ValidationError):
            Team(**team_data)
    
    def test_team_position_validation(self):
        """Test team position validation (1-20)."""
        team_data = {
            'id': 1, 'name': 'Test Team', 'short_name': 'TST',
            'position': 0  # Invalid position
        }
        
        with pytest.raises(ValidationError):
            Team(**team_data)
        
        team_data['position'] = 21  # Invalid position
        with pytest.raises(ValidationError):
            Team(**team_data)
        
        # Valid positions
        team_data['position'] = 1
        team = Team(**team_data)
        assert team.position == 1
        
        team_data['position'] = 20
        team = Team(**team_data)
        assert team.position == 20
    
    def test_team_negative_values_validation(self):
        """Test validation fails for negative values where not allowed."""
        team_data = {
            'id': 1, 'name': 'Test Team', 'short_name': 'TST',
            'played': -1
        }
        
        with pytest.raises(ValidationError):
            Team(**team_data)
        
        team_data['played'] = 0
        team_data['wins'] = -1
        with pytest.raises(ValidationError):
            Team(**team_data)
    
    def test_team_required_fields_missing(self):
        """Test validation fails when required fields are missing."""
        # Missing id
        with pytest.raises(ValidationError):
            Team(name='Test Team', short_name='TST')
        
        # Missing name
        with pytest.raises(ValidationError):
            Team(id=1, short_name='TST')
        
        # Missing short_name
        with pytest.raises(ValidationError):
            Team(id=1, name='Test Team')


@pytest.mark.unit
class TestFixture:
    """Comprehensive unit tests for Fixture model."""
    
    def test_fixture_minimal_valid_data(self):
        """Test Fixture creation with minimal valid data."""
        fixture_data = {
            'id': 1,
            'code': 123456,
            'team_h': 1,
            'team_a': 2
        }
        
        fixture = Fixture(**fixture_data)
        
        assert fixture.id == 1
        assert fixture.code == 123456
        assert fixture.team_h == 1
        assert fixture.team_a == 2
        
        # Test default values
        assert fixture.event is None
        assert fixture.kickoff_time is None
        assert fixture.finished is False
        assert fixture.started is False
        assert fixture.provisional_start_time is False
        assert fixture.team_h_score is None
        assert fixture.team_a_score is None
    
    def test_fixture_complete_data(self):
        """Test Fixture creation with complete data."""
        kickoff_time = datetime(2024, 1, 15, 15, 0)
        
        fixture_data = {
            'id': 1,
            'code': 123456,
            'event': 20,
            'team_h': 1,
            'team_a': 2,
            'kickoff_time': kickoff_time,
            'finished': True,
            'started': True,
            'provisional_start_time': False,
            'team_h_score': 2,
            'team_a_score': 1
        }
        
        fixture = Fixture(**fixture_data)
        
        assert fixture.event == 20
        assert fixture.kickoff_time == kickoff_time
        assert fixture.finished is True
        assert fixture.started is True
        assert fixture.team_h_score == 2
        assert fixture.team_a_score == 1
    
    def test_fixture_score_validation(self):
        """Test fixture score validation (non-negative)."""
        fixture_data = {
            'id': 1, 'code': 123456, 'team_h': 1, 'team_a': 2,
            'team_h_score': -1
        }
        
        with pytest.raises(ValidationError):
            Fixture(**fixture_data)
        
        fixture_data['team_h_score'] = 0
        fixture_data['team_a_score'] = -1
        
        with pytest.raises(ValidationError):
            Fixture(**fixture_data)
        
        # Valid scores
        fixture_data['team_h_score'] = 3
        fixture_data['team_a_score'] = 0
        fixture = Fixture(**fixture_data)
        assert fixture.team_h_score == 3
        assert fixture.team_a_score == 0
    
    def test_fixture_required_fields_missing(self):
        """Test validation fails when required fields are missing."""
        # Missing id
        with pytest.raises(ValidationError):
            Fixture(code=123456, team_h=1, team_a=2)
        
        # Missing code
        with pytest.raises(ValidationError):
            Fixture(id=1, team_h=1, team_a=2)
        
        # Missing team_h
        with pytest.raises(ValidationError):
            Fixture(id=1, code=123456, team_a=2)
        
        # Missing team_a
        with pytest.raises(ValidationError):
            Fixture(id=1, code=123456, team_h=1)
    
    def test_fixture_datetime_handling(self):
        """Test fixture handles datetime objects correctly."""
        kickoff_time = datetime(2024, 2, 10, 17, 30)
        
        fixture_data = {
            'id': 1, 'code': 123456, 'team_h': 1, 'team_a': 2,
            'kickoff_time': kickoff_time
        }
        
        fixture = Fixture(**fixture_data)
        assert fixture.kickoff_time == kickoff_time
        assert isinstance(fixture.kickoff_time, datetime)
    
    def test_fixture_boolean_fields(self):
        """Test fixture boolean fields work correctly."""
        fixture_data = {
            'id': 1, 'code': 123456, 'team_h': 1, 'team_a': 2,
            'finished': True,
            'started': True,
            'provisional_start_time': True
        }
        
        fixture = Fixture(**fixture_data)
        assert fixture.finished is True
        assert fixture.started is True
        assert fixture.provisional_start_time is True
        
        # Test with False values
        fixture_data.update({
            'finished': False,
            'started': False,
            'provisional_start_time': False
        })
        
        fixture = Fixture(**fixture_data)
        assert fixture.finished is False
        assert fixture.started is False
        assert fixture.provisional_start_time is False


@pytest.mark.unit
class TestModelIntegration:
    """Integration tests for model interactions."""
    
    def test_player_with_team_element_type(self):
        """Test Player model works with Team and ElementType."""
        # Create a team
        team = Team(id=1, name='Liverpool', short_name='LIV')
        
        # Create a player for that team
        player = Player(
            id=1, first_name='Mohamed', second_name='Salah', web_name='Salah',
            element_type=ElementType.FORWARD, team=team.id, now_cost=130
        )
        
        assert player.team == team.id
        assert player.element_type == ElementType.FORWARD
        assert player.position_name == "FWD"
    
    def test_fixture_with_teams(self):
        """Test Fixture model works with team IDs."""
        team1 = Team(id=1, name='Arsenal', short_name='ARS')
        team2 = Team(id=2, name='Chelsea', short_name='CHE')
        
        fixture = Fixture(
            id=1, code=123456, team_h=team1.id, team_a=team2.id,
            event=20, finished=True, team_h_score=2, team_a_score=1
        )
        
        assert fixture.team_h == team1.id
        assert fixture.team_a == team2.id
        assert fixture.finished is True
        assert fixture.team_h_score == 2
        assert fixture.team_a_score == 1
    
    def test_model_serialization(self):
        """Test models can be serialized and deserialized."""
        # Create a player
        player = Player(
            id=1, first_name='Test', second_name='Player', web_name='Test',
            element_type=ElementType.MIDFIELDER, team=1, now_cost=80,
            total_points=120, goals_scored=5, assists=8
        )
        
        # Serialize to dict
        player_dict = player.model_dump()
        assert isinstance(player_dict, dict)
        assert player_dict['id'] == 1
        assert player_dict['total_points'] == 120
        assert player_dict['element_type'] == 3  # MIDFIELDER enum value
        
        # Recreate from dict
        new_player = Player(**player_dict)
        assert new_player.id == player.id
        assert new_player.total_points == player.total_points
        assert new_player.element_type == player.element_type