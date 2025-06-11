from typing import List, Optional
from pydantic import BaseModel, Field

class PersonalContext(BaseModel):
    values: List[str] = Field(default=[], description="List of personal values")
    challenges: List[str] = Field(default=[], description="List of current challenges")
    mood: Optional[str] = Field(default="neutral", description="Current emotional state")
    goals: Optional[List[str]] = Field(default=[], description="List of personal goals")

    class Config:
        json_schema_extra = {
            "example": {
                "values": ["growth", "authenticity", "compassion"],
                "challenges": ["work-life balance", "self-doubt"],
                "mood": "contemplative",
                "goals": ["self-improvement", "meaningful work"]
            }
        }