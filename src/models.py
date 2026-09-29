"""
Data models for AI-Powered Resource Allocation System.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Resource(BaseModel):
    id: str
    name: str
    type: str
    category: str = "Physical"  # Physical or Human
    experience_years: int
    cost_inr: float
    available: bool
    water_capacity_lpd: Optional[float] = None
    chemical_compatibility: List[str] = Field(default_factory=list)
    specifications: Optional[str] = ""
    power_rating_kw: Optional[float] = None
    location: Optional[str] = "Main Site"
    next_available_date: Optional[str] = "Immediate"


class ParsedRequirement(BaseModel):
    raw_query: str
    resource_type: str = "Any"
    min_experience_years: Optional[int] = None
    max_budget_inr: Optional[float] = None
    water_usage_lpd: Optional[float] = None
    chemical_processes: List[str] = Field(default_factory=list)
    quantity_required: int = 1
    urgency: str = "Normal"  # Normal, Immediate, Scheduled
    is_ambiguous: bool = False
    is_unrelated: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    assumptions_made: List[str] = Field(default_factory=list)
    clarification_questions: List[str] = Field(default_factory=list)


class DimensionGap(BaseModel):
    dimension: str
    required_value: Any
    actual_value: Any
    gap_description: str
    severity: str = "Minor"  # Critical, Major, Minor, None
    is_satisfied: bool = True


class GapReport(BaseModel):
    has_gaps: bool
    critical_blockers: List[str] = Field(default_factory=list)
    dimension_gaps: List[DimensionGap] = Field(default_factory=list)
    remediation_suggestions: List[str] = Field(default_factory=list)


class CandidateEvaluation(BaseModel):
    resource: Resource
    overall_score: float  # 0.0 to 100.0
    is_perfect_match: bool
    is_feasible: bool
    gap_report: GapReport
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    justification_points: List[str] = Field(default_factory=list)
    shortfall_reasons: List[str] = Field(default_factory=list)
    rejection_summary: str = "Satisfies all constraints"


class AllocationResponse(BaseModel):
    requirement: ParsedRequirement
    recommended_resource: Optional[Resource] = None
    alternative_resources: List[Resource] = Field(default_factory=list)
    top_evaluation: Optional[CandidateEvaluation] = None
    all_evaluations: List[CandidateEvaluation] = Field(default_factory=list)
    overall_status: str  # "MATCH_FOUND", "PARTIAL_MATCH_WITH_GAPS", "NO_FEASIBLE_RESOURCE", "AMBIGUOUS_QUERY"
    natural_language_explanation: str
    gap_analysis_summary: str
    non_selection_report: str = ""
