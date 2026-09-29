"""
Resource Matching and Multi-Criteria Scoring Engine.
Evaluates candidate resources, calculates compatibility scores, and ranks recommendations.
"""
import json
from typing import List, Dict, Any, Optional
from .models import Resource, ParsedRequirement, CandidateEvaluation, AllocationResponse
from .gap_analyzer import GapAnalyzer


class ResourceMatcher:
    """
    Evaluates, scores, and ranks candidate resources against structured business requirements.
    """

    def __init__(self, resources: Optional[List[Resource]] = None, dataset_path: Optional[str] = None):
        self.gap_analyzer = GapAnalyzer()
        if resources is not None:
            self.resources = resources
        elif dataset_path is not None:
            self.resources = self._load_from_file(dataset_path)
        else:
            self.resources = []

    def _load_from_file(self, filepath: str) -> List[Resource]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Resource(**item) for item in data]

    def evaluate_resource(self, req: ParsedRequirement, res: Resource) -> CandidateEvaluation:
        gap_report = self.gap_analyzer.analyze(req, res)

        score_breakdown: Dict[str, float] = {}
        justification_points: List[str] = []

        shortfall_reasons: List[str] = []

        # 1. Type Match Weight (30 pts)
        if req.resource_type == "Any" or req.resource_type.lower() == res.type.lower():
            score_breakdown["type_score"] = 30.0
            justification_points.append(f"Meets requested resource category ({res.type})")
        else:
            score_breakdown["type_score"] = 0.0
            shortfall_reasons.append(f"Category mismatch (Requested '{req.resource_type}', candidate is '{res.type}')")

        # 2. Availability Weight (20 pts)
        if res.available:
            score_breakdown["availability_score"] = 20.0
            justification_points.append("Resource is currently available for immediate deployment")
        else:
            score_breakdown["availability_score"] = 5.0  # partial credit if scheduled
            next_avail = res.next_available_date or "Unknown date"
            shortfall_reasons.append(f"Currently unavailable (In use / maintenance until {next_avail})")

        # 3. Experience Match Weight (25 pts)
        if req.min_experience_years is not None:
            if res.experience_years >= req.min_experience_years:
                score_breakdown["experience_score"] = 25.0
                surplus = res.experience_years - req.min_experience_years
                if surplus > 0:
                    justification_points.append(
                        f"Experience exceeds requirement: {res.experience_years} years vs {req.min_experience_years} required (+{surplus} yrs)"
                    )
                else:
                    justification_points.append(f"Meets minimum experience threshold of {req.min_experience_years} years")
            else:
                deficit = req.min_experience_years - res.experience_years
                ratio = max(0.0, res.experience_years / req.min_experience_years)
                score_breakdown["experience_score"] = round(25.0 * ratio, 1)
                shortfall_reasons.append(f"Experience deficit of {deficit} yr(s) (Has {res.experience_years} yrs vs {req.min_experience_years} yrs req)")
        else:
            score_breakdown["experience_score"] = 25.0  # no constraint specified

        # 4. Budget Match Weight (15 pts)
        if req.max_budget_inr is not None:
            if res.cost_inr <= req.max_budget_inr:
                score_breakdown["budget_score"] = 15.0
                savings = req.max_budget_inr - res.cost_inr
                justification_points.append(
                    f"Cost of Rs. {res.cost_inr:,.2f} is within budget of Rs. {req.max_budget_inr:,.2f} (Savings: Rs. {savings:,.2f})"
                )
            else:
                overrun = res.cost_inr - req.max_budget_inr
                overrun_ratio = overrun / req.max_budget_inr
                score_breakdown["budget_score"] = max(0.0, round(15.0 * (1.0 - min(1.0, overrun_ratio)), 1))
                shortfall_reasons.append(f"Budget overrun by Rs. {overrun:,.0f} (+{overrun_ratio*100:.1f}%)")
        else:
            score_breakdown["budget_score"] = 15.0

        # 5. Technical Specification & Water Match (10 pts)
        tech_score = 10.0
        if req.water_usage_lpd is not None and res.water_capacity_lpd is not None:
            if res.water_capacity_lpd >= req.water_usage_lpd:
                justification_points.append(
                    f"Water throughput capacity ({res.water_capacity_lpd:,.0f} L/day) handles required load ({req.water_usage_lpd:,.0f} L/day)"
                )
            else:
                tech_score -= 5.0
                water_deficit = req.water_usage_lpd - res.water_capacity_lpd
                shortfall_reasons.append(f"Capacity deficit of {water_deficit:,.0f} L/day (Has {res.water_capacity_lpd:,.0f} vs {req.water_usage_lpd:,.0f} req)")

        if req.chemical_processes:
            matched = [
                c for c in req.chemical_processes
                if any(c.lower() in rc.lower() or rc.lower() in c.lower() for rc in res.chemical_compatibility)
            ]
            unmatched = [c for c in req.chemical_processes if c not in matched]
            if matched:
                justification_points.append(f"Certified for required chemical processes: {', '.join(matched)}")
            if unmatched:
                tech_score -= 5.0
                shortfall_reasons.append(f"Incompatible / unverified for: {', '.join(unmatched)}")

        score_breakdown["tech_spec_score"] = max(0.0, tech_score)

        overall_score = sum(score_breakdown.values())
        is_perfect = not gap_report.has_gaps and (score_breakdown.get("type_score", 0) > 0)
        is_feasible = len(gap_report.critical_blockers) == 0

        rejection_summary = "; ".join(shortfall_reasons) if shortfall_reasons else "Meets requirements (Lower rank / cost trade-off)"

        return CandidateEvaluation(
            resource=res,
            overall_score=round(overall_score, 1),
            is_perfect_match=is_perfect,
            is_feasible=is_feasible,
            gap_report=gap_report,
            score_breakdown=score_breakdown,
            justification_points=justification_points,
            shortfall_reasons=shortfall_reasons,
            rejection_summary=rejection_summary
        )

    def match(self, req: ParsedRequirement) -> List[CandidateEvaluation]:
        if req.is_unrelated:
            return []

        evaluations: List[CandidateEvaluation] = []
        for res in self.resources:
            # Filter out completely unrelated types if specific type requested
            if req.resource_type != "Any" and req.resource_type.lower() != res.type.lower():
                continue
            evaluation = self.evaluate_resource(req, res)
            evaluations.append(evaluation)

        # If strict type filter returned nothing, evaluate all resources
        if not evaluations:
            for res in self.resources:
                evaluation = self.evaluate_resource(req, res)
                evaluations.append(evaluation)

        # Sort primarily by feasibility, then by overall score descending, then by cost ascending
        evaluations.sort(
            key=lambda e: (
                1 if e.is_feasible else 0,
                1 if e.resource.available else 0,
                e.overall_score,
                -e.resource.cost_inr
            ),
            reverse=True
        )

        return evaluations
