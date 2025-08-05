"""
Comprehensive Pydantic data models for FPL structures.
Based on FPL API bootstrap-static endpoint analysis.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ElementType(int, Enum):
    """Player position types in FPL."""
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4


class ChipType(str, Enum):
    """Available chip types in FPL."""
    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"


class Player(BaseModel):
    """Core player model based on FPL API elements structure."""
    
    # Basic information
    id: int = Field(..., description="Unique player ID")
    first_name: str = Field(..., description="Player first name")
    second_name: str = Field(..., description="Player surname")
    web_name: str = Field(..., description="Short display name")
    element_type: ElementType = Field(..., description="Player position (1=GK, 2=DEF, 3=MID, 4=FWD)")
    team: int = Field(..., description="Team ID")
    
    # Financial data
    now_cost: int = Field(..., description="Current price in 0.1M units (e.g., 100 = £10.0M)")
    cost_change_event: int = Field(0, description="Price change this gameweek")
    cost_change_start: int = Field(0, description="Price change from season start")
    
    # Performance metrics
    total_points: int = Field(0, ge=0, description="Total points this season")
    points_per_game: float = Field(0, ge=0, description="Average points per game")
    form: str = Field("0.0", description="Form rating over last 5 games")
    minutes: int = Field(0, ge=0, description="Total minutes played")
    
    # Basic stats
    goals_scored: int = Field(0, ge=0, description="Goals scored this season")
    assists: int = Field(0, ge=0, description="Assists this season")
    clean_sheets: int = Field(0, ge=0, description="Clean sheets this season")
    goals_conceded: int = Field(0, ge=0, description="Goals conceded this season")
    saves: int = Field(0, ge=0, description="Saves made (GK only)")
    
    # Bonus system
    bonus: int = Field(0, ge=0, description="Bonus points earned")
    bps: int = Field(0, ge=0, description="Bonus points system score")
    
    # Advanced metrics (ICT Index)
    influence: str = Field("0.0", description="Influence rating")
    creativity: str = Field("0.0", description="Creativity rating")
    threat: str = Field("0.0", description="Threat rating")
    ict_index: str = Field("0.0", description="ICT Index combined rating")
    
    # Expected stats
    expected_goals: str = Field("0.0", description="Expected goals (xG)")
    expected_assists: str = Field("0.0", description="Expected assists (xA)")
    expected_goal_involvements: str = Field("0.0", description="Expected goal involvements")
    expected_goals_conceded: str = Field("0.0", description="Expected goals conceded")
    
    # Usage statistics
    selected_by_percent: str = Field("0.0", description="Ownership percentage")
    transfers_in_event: int = Field(0, description="Transfers in this gameweek")
    transfers_out_event: int = Field(0, description="Transfers out this gameweek")
    
    # Status information
    chance_of_playing_this_round: Optional[int] = Field(None, ge=0, le=100, description="Injury/availability chance")
    chance_of_playing_next_round: Optional[int] = Field(None, ge=0, le=100, description="Next gameweek availability")
    news: str = Field("", description="Latest player news")
    news_added: Optional[datetime] = Field(None, description="When news was added")
    
    @validator('now_cost')
    def validate_price(cls, v):
        """Ensure price is in valid FPL range."""
        if not (30 <= v <= 150):  # £3.0M to £15.0M in 0.1M units
            raise ValueError('Player price must be between £3.0M and £15.0M')
        return v
    
    @property
    def price_millions(self) -> float:
        """Get price in millions (e.g., 100 -> 10.0)."""
        return self.now_cost / 10.0
    
    @property
    def position_name(self) -> str:
        """Get position name as string."""
        position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        return position_map.get(self.element_type, "UNK")


class Team(BaseModel):
    """FPL team model based on API teams structure."""
    
    id: int = Field(..., description="Unique team ID")
    name: str = Field(..., description="Full team name")
    short_name: str = Field(..., description="3-letter team code")
    
    # League position and performance
    played: int = Field(0, ge=0, description="Games played")
    wins: int = Field(0, ge=0, description="Games won")
    draws: int = Field(0, ge=0, description="Games drawn")
    losses: int = Field(0, ge=0, description="Games lost")
    points: int = Field(0, ge=0, description="League points")
    position: int = Field(1, ge=1, le=20, description="League position")
    
    # Strength ratings (1-5 scale)
    strength: int = Field(3, ge=1, le=5, description="Overall team strength")
    strength_overall_home: int = Field(3, ge=1, le=5, description="Home strength overall")
    strength_overall_away: int = Field(3, ge=1, le=5, description="Away strength overall")
    strength_attack_home: int = Field(3, ge=1, le=5, description="Home attack strength")
    strength_attack_away: int = Field(3, ge=1, le=5, description="Away attack strength")
    strength_defence_home: int = Field(3, ge=1, le=5, description="Home defence strength")
    strength_defence_away: int = Field(3, ge=1, le=5, description="Away defence strength")


class Fixture(BaseModel):
    """FPL fixture model based on API fixtures structure."""
    
    id: int = Field(..., description="Unique fixture ID")
    code: int = Field(..., description="Fixture code")
    event: Optional[int] = Field(None, description="Gameweek number")
    
    # Teams
    team_h: int = Field(..., description="Home team ID")
    team_a: int = Field(..., description="Away team ID")
    
    # Timing
    kickoff_time: Optional[datetime] = Field(None, description="Kickoff time")
    
    # Status
    finished: bool = Field(False, description="Whether fixture is finished")
    started: bool = Field(False, description="Whether fixture has started")
    provisional_start_time: bool = Field(False, description="Whether start time is provisional")
    
    # Results (when finished)
    team_h_score: Optional[int] = Field(None, ge=0, description="Home team score")
    team_a_score: Optional[int] = Field(None, ge=0, description="Away team score")
    
    # Difficulty ratings (1-5 scale)
    team_h_difficulty: int = Field(3, ge=1, le=5, description="Difficulty for home team")
    team_a_difficulty: int = Field(3, ge=1, le=5, description="Difficulty for away team")
    
    # Performance stats (when available)
    stats: List[Dict[str, Any]] = Field(default_factory=list, description="Match statistics")


class Event(BaseModel):
    """FPL gameweek/event model based on API events structure."""
    
    id: int = Field(..., description="Gameweek number")
    name: str = Field(..., description="Gameweek name (e.g., 'Gameweek 1')")
    
    # Timing
    deadline_time: datetime = Field(..., description="Transfer deadline")
    deadline_time_epoch: int = Field(..., description="Deadline as Unix timestamp")
    
    # Status
    is_previous: bool = Field(False, description="Is previous gameweek")
    is_current: bool = Field(False, description="Is current gameweek")
    is_next: bool = Field(False, description="Is next gameweek")
    finished: bool = Field(False, description="Is gameweek finished")
    data_checked: bool = Field(False, description="Is data verified")
    
    # Statistics
    average_entry_score: Optional[int] = Field(None, description="Average points scored")
    highest_score: Optional[int] = Field(None, description="Highest individual score")
    highest_scoring_entry: Optional[int] = Field(None, description="Team ID with highest score")
    most_selected: Optional[int] = Field(None, description="Most selected player ID")
    most_transferred_in: Optional[int] = Field(None, description="Most transferred in player ID")
    most_captained: Optional[int] = Field(None, description="Most captained player ID")
    most_vice_captained: Optional[int] = Field(None, description="Most vice captained player ID")
    
    # Transfer information
    transfers_made: Optional[int] = Field(None, description="Total transfers made")
    chip_plays: List[Dict[str, Any]] = Field(default_factory=list, description="Chip usage stats")


class PlayerPrediction(BaseModel):
    """ML prediction result for a player."""
    
    player_id: int = Field(..., description="Player ID")
    gameweeks_ahead: int = Field(..., ge=1, le=8, description="Number of gameweeks predicted")
    
    # Predictions
    expected_points: List[float] = Field(..., description="Expected points for each gameweek")
    confidence_intervals: List[Dict[str, float]] = Field(..., description="95% confidence intervals")
    
    # Model metadata
    model_used: str = Field(..., description="Primary model used (XGBoost, LSTM, etc.)")
    prediction_date: datetime = Field(default_factory=datetime.now, description="When prediction was made")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    
    @validator('expected_points')
    def validate_expected_points_length(cls, v, values):
        if 'gameweeks_ahead' in values and len(v) != values['gameweeks_ahead']:
            raise ValueError('Expected points length must match gameweeks_ahead')
        return v


class TransferRecommendation(BaseModel):
    """Transfer recommendation from optimization agent."""
    
    player_out: Player = Field(..., description="Player to transfer out")
    player_in: Player = Field(..., description="Player to transfer in")
    
    # Recommendation metrics
    expected_points_gain: float = Field(..., description="Expected points gain over analysis period")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in recommendation (0-1)")
    cost_change: float = Field(..., description="Price difference (positive = more expensive)")
    
    # Analysis context
    reasoning: List[str] = Field(..., min_items=1, description="Reasons for this transfer")
    weeks_analyzed: int = Field(..., ge=1, description="Number of weeks analysis covers")
    risk_level: str = Field(..., pattern="^(Low|Medium|High)$", description="Risk assessment")
    
    # Supporting data
    fixture_analysis: Dict[str, Any] = Field(default_factory=dict, description="Fixture difficulty analysis")
    form_analysis: Dict[str, Any] = Field(default_factory=dict, description="Recent form comparison")


class OptimizedTeam(BaseModel):
    """Team optimization result from PuLP solver."""
    
    status: str = Field(..., pattern="^(optimal|infeasible|timeout|error)$", description="Optimization status")
    
    # Results (when successful)
    selected_players: Optional[List[int]] = None
    total_cost: Optional[float] = None
    predicted_points: Optional[float] = None
    remaining_budget: Optional[float] = None
    
    # Team structure
    formation: Optional[str] = None  # e.g., "3-4-3"
    captain_suggestion: Optional[int] = None
    vice_captain_suggestion: Optional[int] = None
    
    # Transfer-specific results
    recommended_transfers: Optional[List[TransferRecommendation]] = None
    transfer_cost: Optional[int] = None  # Points cost for transfers
    expected_points_gain: Optional[float] = None
    
    # Performance metrics
    optimization_time: Optional[float] = None  # Seconds taken
    iterations: Optional[int] = None  # Solver iterations
    
    # Error information (when status != optimal)
    error_message: Optional[str] = None
    constraints_violated: Optional[List[str]] = None
    
    @validator('total_cost')
    def validate_total_cost(cls, v):
        if v is not None and v > 100.0:
            raise ValueError('Total cost cannot exceed £100.0M')
        return v
    
    @validator('selected_players')
    def validate_team_size(cls, v):
        if v is not None and len(v) != 15:
            raise ValueError('Team must have exactly 15 players')
        return v


class ChipStrategy(BaseModel):
    """Chip usage strategy recommendation."""
    
    chip_name: ChipType = Field(..., description="Type of chip")
    optimal_gameweek: int = Field(..., ge=1, le=38, description="Recommended gameweek to play chip")
    expected_points_gain: float = Field(..., ge=0, description="Expected additional points")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in recommendation")
    
    # Reasoning and requirements
    reasoning: List[str] = Field(..., min_items=1, description="Why this gameweek is optimal")
    requirements: List[str] = Field(default_factory=list, description="Prerequisites for optimal usage")
    
    # Timing considerations
    earliest_recommended: int = Field(..., ge=1, le=38, description="Earliest viable gameweek")
    latest_recommended: int = Field(..., ge=1, le=38, description="Latest viable gameweek")
    
    # Supporting analysis
    fixture_congestion: Dict[str, float] = Field(default_factory=dict, description="Fixture difficulty by team")
    double_gameweeks: List[int] = Field(default_factory=list, description="Double gameweek numbers")
    
    @validator('latest_recommended')
    def validate_timing_order(cls, v, values):
        if 'earliest_recommended' in values and v < values['earliest_recommended']:
            raise ValueError('Latest recommended must be >= earliest recommended')
        return v


class SystemHealth(BaseModel):
    """System health monitoring model."""
    
    is_healthy: bool = Field(..., description="Overall system health status")
    last_check: datetime = Field(default_factory=datetime.now, description="Last health check time")
    
    # Component health
    components: List['ComponentHealth'] = Field(..., min_items=1, description="Individual component status")
    
    # Performance metrics
    response_time_ms: Optional[float] = Field(None, description="Average response time in milliseconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    disk_usage_percent: Optional[float] = Field(None, description="Disk usage percentage")
    
    @property
    def unhealthy_components(self) -> List['ComponentHealth']:
        """Get list of unhealthy components."""
        return [c for c in self.components if not c.is_healthy]


class ComponentHealth(BaseModel):
    """Individual component health status."""
    
    name: str = Field(..., description="Component name")
    is_healthy: bool = Field(..., description="Component health status")
    status: str = Field(..., description="Status description")
    last_updated: datetime = Field(..., description="Last update time")
    
    # Issues and metrics
    issues: List[str] = Field(default_factory=list, description="Current issues")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Component-specific metrics")
    
    # Performance data
    response_time_ms: Optional[float] = Field(None, description="Component response time")
    error_rate: Optional[float] = Field(None, description="Error rate (0-1)")
    uptime_percent: Optional[float] = Field(None, description="Uptime percentage")


# Enable forward references for circular dependencies
SystemHealth.model_rebuild()
OptimizedTeam.model_rebuild()