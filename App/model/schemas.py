from typing import List, Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    name: str
    
class PersonalContext(BaseModel):
    values: List[str] = Field(default=[], description="List of personal values")
    challenges: List[str] = Field(default=[], description="List of current challenges")
    goals: Optional[List[str]] = Field(default=[], description="List of personal goals")
    feelings: Optional[List[str]] = Field(default=[], description="How the user is feeling today")
    alignment_moment: Optional[str] = Field(default="", description="Description of a moment when user felt aligned with values")
    misalignment_moment: Optional[str] = Field(default="", description="Description of a moment of misalignment")
    greater_alignment: Optional[str] = Field(default="", description="What greater alignment would look like")
    reflections: Optional[str] = Field(default="", description="Free reflections")

    class Config:
        json_schema_extra = {
            "example": {
                "values": ["growth", "authenticity", "compassion"],
                "challenges": ["work-life balance", "self-doubt"],
                "goals": ["self-improvement", "meaningful work"],
                "feelings": ["hopeful", "anxious"],
                "alignment_moment": "When I chose to pursue a project aligned with my values.",
                "misalignment_moment": "When I accepted a task that conflicted with my personal beliefs.",
                "greater_alignment": "Consistently making choices that reflect my true self.",
                "reflections": "I need to focus on what truly matters to me and not get sidetracked by external expectations."
            }
        }