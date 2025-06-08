"""
Interface definitions using Pydantic models for the Top Eleven Bot
"""

from typing import Optional, Tuple
from pydantic import BaseModel, Field

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
    positions: list[Optional[str]]
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