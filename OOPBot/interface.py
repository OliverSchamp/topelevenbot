"""
Interface definitions using Pydantic models for the Top Eleven Bot
"""

from typing import Optional, Tuple, List
from pydantic import BaseModel, Field
from datetime import datetime
import os
import csv

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

class BidDetails(BaseModel):
    """Details about a bid attempt including budgets and amounts"""
    starting_bid_tokens: Optional[float] = Field(None, description="Initial token bid amount")
    current_bid_tokens: Optional[float] = Field(None, description="Final/current token bid amount")
    starting_bid_money: Optional[float] = Field(None, description="Initial money bid amount")
    current_bid_money: Optional[float] = Field(None, description="Final/current money bid amount")
    token_budget: Optional[float] = Field(None, description="Available token budget at time of bid")
    money_budget: Optional[float] = Field(None, description="Available money budget at time of bid")

class PlayerAttributes(BaseModel):
    """Player attributes and auction details"""
    timestamp: datetime = Field(default_factory=datetime.now, description="When the player was evaluated")
    name: Optional[str] = Field(None, description="Player name")
    age: Optional[int] = Field(None, description="Player age")
    quality: Optional[int] = Field(None, description="Player quality percentage")
    value: Optional[float] = Field(None, description="Player value in millions")
    expected_value: Optional[float] = Field(None, description="Expected value from fast trainers sheet")
    positions: List[str] = Field(default_factory=list, description="Player positions")
    playstyle: Optional[str] = Field(None, description="Player playstyle")
    comparison_result: Optional[str] = Field(None, description="Result of value comparison")
    reason_rejected: Optional[str] = Field(None, description="Reason for rejecting the player")
    was_bid_placed: bool = Field(False, description="Whether a bid was placed")
    bid_amount: Optional[float] = Field(None, description="Amount of tokens bid")
    bid_details: Optional[BidDetails] = Field(None, description="Detailed information about the bid attempt")
