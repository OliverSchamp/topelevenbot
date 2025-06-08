"""
Interface definitions using Pydantic models for the Top Eleven Bot
"""

from typing import Optional, Tuple, List
from pydantic import BaseModel, Field
from datetime import datetime

class ScreenRegion(BaseModel):
    """Region on the screen defined by coordinates"""
    x1: int = Field(..., description="Left coordinate")
    y1: int = Field(..., description="Top coordinate")
    x2: int = Field(..., description="Right coordinate")
    y2: int = Field(..., description="Bottom coordinate")

class TemplateMatch(BaseModel):
    """Result of template matching on screen"""
    center_x: Optional[int] = Field(None, description="X coordinate of match center")
    center_y: Optional[int] = Field(None, description="Y coordinate of match center")
    top_left_x: Optional[int] = Field(None, description="X coordinate of top-left corner")
    top_left_y: Optional[int] = Field(None, description="Y coordinate of top-left corner")
    width: Optional[int] = Field(None, description="Width of matched template")
    height: Optional[int] = Field(None, description="Height of matched template")
    confidence: float = Field(..., description="Confidence score of the match")

class PlayerDetails(BaseModel):
    """Details of a player in the game"""
    name: str
    age: int
    value: float
    quality: int
    positions: List[Optional[str]]
    playstyle: Optional[str]

class BotStatus(BaseModel):
    """Current status of the bot"""
    mode: Optional[str]
    team_name: str
    is_running: bool

class TrainingProgress(BaseModel):
    """Training progress and condition information"""
    progress: Optional[int] = Field(None, description="Current training progress percentage")
    greens_budget: Optional[int] = Field(None, description="Available greens for condition restoration")

class PlayerAttributes(BaseModel):
    """Complete record of a player's attributes from auction"""
    timestamp: datetime = Field(default_factory=datetime.now, description="When the player was evaluated")
    name: Optional[str] = Field(None, description="Player's name")
    age: Optional[int] = Field(None, description="Player's age")
    value: Optional[float] = Field(None, description="Player's value in millions")
    quality: Optional[int] = Field(None, description="Player's quality rating")
    positions: List[Optional[str]] = Field(default_factory=list, description="List of player's positions")
    playstyle: Optional[str] = Field(None, description="Player's playstyle")
    expected_value: Optional[float] = Field(None, description="Expected value based on quality/age")
    comparison_result: Optional[str] = Field(None, description="Result of value comparison")
    reason_rejected: Optional[str] = Field(None, description="Reason if player was rejected")
    was_bid_placed: bool = Field(False, description="Whether a bid was placed on this player")
    bid_amount: Optional[float] = Field(None, description="Amount of bid placed on this player")

class AuctionPageResult(BaseModel):
    """Result of processing an auction page, including player data and new Y position"""
    player_attributes: Optional[PlayerAttributes] = Field(None, description="Player attributes if found")
    new_y_position: Optional[int] = Field(None, description="New Y position for next scan, if auction reset") 