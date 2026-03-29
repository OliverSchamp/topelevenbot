"""
Interface definitions using Pydantic models for the Top Eleven Bot
"""

from typing import Optional, Tuple, List
from pydantic import BaseModel, Field
from datetime import datetime
import os
import csv
from rapidfuzz.distance import Levenshtein


class PlayerInfo(BaseModel):
    name: Optional[str] = None
    roles: List[str] = []
    age: Optional[str] = None
    qlty: Optional[str] = None
    value: Optional[str] = None
    pstyl: Optional[bool] = None
    spec: Optional[str] = None
    # roles_unfiltered: Optional[List[str]] = None
    # y_coords: Optional[Tuple[float, float]]

    # def get_y_coord(self) -> float:
    #     return (self.y_coord_range[0] + self.y_coord_range[1]) / 2


    # def get_y_coord_range(self) -> Tuple[float, float]:
    #     return self.y_coords[1] - self.y_coords[0]

    # def split_roles(self):
    #     # split by spacebar to convert ["ST LW"] into ["ST", "LW"] or ["MR", "MC ML"] into ["MR", "MC", "ML"]
    #     split_roles = []
    #     for role in self.roles:
    #         split_roles.extend(role.split())
    #     self.roles = split_roles

    VALID_ROLES: List[str] = [
    "GK",   # Goalkeeper
    "DC",   # Center Back
    "DL",   # Left Back
    "DR",   # Right Back
    "DMC",  # Defensive Midfielder
    "AMC",  # Attacking Midfielder
    "AML",  # Left Winger 
    "AMR",  # Right Winger
    "MC",   # Center Midfielder
    "ML",   # Left Midfielder
    "MR",   # Right Midfielder
    "ST"    # Striker
    ]


    def clean_roles(self):
        # iterate through the roles, and match the strings to the valid position most similar. use levenshtein distance to determine similarity. if the most similar valid position is above a certain threshold, assign that position, otherwise discard the role
        cleaned_roles = []
        for role in self.roles:
            # remove any characters present in role that are not found in VALID_ROLES
            for char in role:
                if not any(char in valid_role for valid_role in self.VALID_ROLES):
                    role = role.replace(char, "")
            best_match = None
            best_distance = float('inf')
            for valid_role in self.VALID_ROLES:
                distance = Levenshtein.distance(role, valid_role)
                if distance < best_distance:
                    best_distance = distance
                    best_match = valid_role
            if best_distance < 3: # threshold of 2 edits
                cleaned_roles.append(best_match)

    # def cleanup_qlty_value(self):
    #     # only take the first 2 characters of the qlty string
    #     if self.qlty is not None:
    #         self.qlty = self.qlty[:2]

    # def split_roles(self):
    #     # extract all valid roles from a list of strings. first join the roles into one string. then remove any characters not present in any valid positions. then attempt to construct the remaining string from a combination of concatenated valid positions
    #     self.roles_unfiltered = self.roles.copy()
    #     roles_string = "".join(self.roles)
    #     for char in roles_string:
    #         if not any(char in role for role in self.VALID_ROLES):
    #             roles_string = roles_string.replace(char, "")
    #     extracted_roles = []
    #     for role in self.VALID_ROLES:
    #         if role in roles_string:
    #             extracted_roles.append(role)
    #             roles_string = roles_string.replace(role, "")
    #     self.roles = extracted_roles
    #     if len(self.roles) == 0:
    #         self.roles = self.roles_unfiltered
    
    def to_string(self):
        return f"Name: {self.name}, Roles: {self.roles}, Age: {self.age}, Quality: {self.qlty}, Value: {self.value}, Playstyle: {self.pstyl}, Special: {self.spec}"

    def is_valid(self):
        return self.name is not None and self.age is not None and self.qlty is not None and self.value is not None and len(self.roles) > 0

class ScreenRegion(BaseModel):
    """Region on the screen defined by coordinates"""
    x1: int = Field(..., description="Left coordinate")
    y1: int = Field(..., description="Top coordinate")
    x2: int = Field(..., description="Right coordinate")
    y2: int = Field(..., description="Bottom coordinate")
    conf: float = Field(..., description="Model Confidence")

    def wh_ratio(self):
        return (self.x2-self.x1)/(self.y2-self.y1)

class TemplateMatch(BaseModel):
    """Result of template matching on screen"""
    center_x: Optional[int] = Field(None, description="X coordinate of match center")
    center_y: Optional[int] = Field(None, description="Y coordinate of match center")
    top_left_x: Optional[int] = Field(None, description="X coordinate of top-left corner")
    top_left_y: Optional[int] = Field(None, description="Y coordinate of top-left corner")
    width: Optional[int] = Field(None, description="Width of matched template")
    height: Optional[int] = Field(None, description="Height of matched template")
    confidence: float = Field(..., description="Confidence score of the match")

    def get_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.top_left_x, self.top_left_y, self.top_left_x + self.width, self.top_left_y + self.height)

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
