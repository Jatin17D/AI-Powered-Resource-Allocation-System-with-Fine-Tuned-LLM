"""
Natural Language Explanation and Justification Generator for Resource Allocation.
Produces clear executive justifications, trade-off breakdowns, and gap analysis narratives.
"""
from typing import List, Optional
from .models import AllocationResponse, CandidateEvaluation, ParsedRequirement, Resource


class Explainer:
    """
    Synthesizes recommendation justifications and gap analysis narratives.
    """

    def explain(
        self,
        req: ParsedRequirement,
        evaluations: List[CandidateEvaluation]
    ) -> AllocationResponse:
        if req.is_unrelated:
            explanation = (
                f"The input query '{req.raw_query}' does not appear to be related to industrial resource allocation. "
                "This system specializes in allocating industrial equipment (Boilers, Refrigeration Chiller units, "
                "Water Treatment plants) and technical personnel (Chemical Process Consultants, Thermal Engineers). "
                "Please provide an industrial equipment, personnel, budget, or engineering requirement."
            )
            return AllocationResponse(
                requirement=req,
                recommended_resource=None,
                alternative_resources=[],
                top_evaluation=None,
                all_evaluations=[],
                overall_status="OUT_OF_DOMAIN",
                natural_language_explanation=explanation,
                gap_analysis_summary="Domain mismatch: No valid industrial resource parameters detected.",
                non_selection_report=""
            )

        if not evaluations:
            explanation = (
                "No resources currently exist in the database that match your search criteria. "
                "Please consider broadening your query or updating resource catalog inventory."
            )
            return AllocationResponse(
                requirement=req,
                recommended_resource=None,
                alternative_resources=[],
                top_evaluation=None,
                all_evaluations=[],
                overall_status="NO_FEASIBLE_RESOURCE",
                natural_language_explanation=explanation,
                gap_analysis_summary="No candidate resources evaluated.",
                non_selection_report=""
            )

        top_eval = evaluations[0]
        top_res = top_eval.resource
        alternatives = [e.resource for e in evaluations[1:4]]

        # Determine overall status
        if req.is_ambiguous and not req.raw_query:
            status = "AMBIGUOUS_QUERY"
        elif top_eval.is_perfect_match:
            status = "MATCH_FOUND"
        elif top_eval.is_feasible:
            status = "PARTIAL_MATCH_WITH_GAPS"
        else:
            status = "NO_FEASIBLE_RESOURCE"

        # 1. Build Natural Language Justification
        explanation_lines = []
        if top_eval.is_perfect_match:
            explanation_lines.append(
                f"{top_res.name} (ID: {top_res.id}) is strongly recommended as a perfect fit for your requirements."
            )
        elif top_eval.is_feasible:
            explanation_lines.append(
                f"{top_res.name} (ID: {top_res.id}) is recommended as the closest available match (Match Score: {top_eval.overall_score}/100)."
            )
        else:
            explanation_lines.append(
                f"No resource completely satisfies all constraints. The nearest candidate is {top_res.name} (ID: {top_res.id}), but significant gaps exist."
            )

        explanation_lines.append("\nWhy this resource was selected:")
        for pt in top_eval.justification_points:
            explanation_lines.append(f"  * {pt}")

        # Note assumptions if any
        if req.assumptions_made:
            explanation_lines.append("\nOperational Assumptions Applied:")
            for assumption in req.assumptions_made:
                explanation_lines.append(f"  * {assumption}")

        # Note clarification questions if ambiguous
        if req.clarification_questions:
            explanation_lines.append("\nClarification Questions for Precision:")
            for q in req.clarification_questions:
                explanation_lines.append(f"  ? {q}")

        # 2. Build Gap Analysis Summary
        gap_lines = []
        gap_report = top_eval.gap_report

        if not gap_report.has_gaps:
            gap_lines.append("No gaps identified. All operational and financial constraints are satisfied.")
        else:
            gap_lines.append("Identified Gaps & Trade-offs:")
            for d_gap in gap_report.dimension_gaps:
                if not d_gap.is_satisfied:
                    gap_lines.append(f"  * [{d_gap.severity.upper()} GAP] {d_gap.dimension}: {d_gap.gap_description}")

            if gap_report.remediation_suggestions:
                gap_lines.append("\nRecommended Remediation Strategies:")
                for rem in gap_report.remediation_suggestions:
                    gap_lines.append(f"  -> {rem}")

        # 3. Build Non-Selection & Alternative Candidate Explanations
        non_selected_evals = [e for e in evaluations if e.resource.id != top_res.id]
        non_selection_lines = []
        if non_selected_evals:
            non_selection_lines.append("Analysis of Why Other Candidates Were Not Allocated:")
            for ev in non_selected_evals:
                r = ev.resource
                reasons = ev.shortfall_reasons if ev.shortfall_reasons else ["Lower overall score / ranking preference compared to top recommendation"]
                non_selection_lines.append(f"\n  • {r.name} (ID: {r.id}, Cost: Rs. {r.cost_inr:,.0f}, Score: {ev.overall_score}/100):")
                for reason in reasons:
                    non_selection_lines.append(f"    - {reason}")
        else:
            non_selection_lines.append("No other candidate resources in inventory to evaluate.")

        return AllocationResponse(
            requirement=req,
            recommended_resource=top_res,
            alternative_resources=alternatives,
            top_evaluation=top_eval,
            all_evaluations=evaluations,
            overall_status=status,
            natural_language_explanation="\n".join(explanation_lines),
            gap_analysis_summary="\n".join(gap_lines),
            non_selection_report="\n".join(non_selection_lines)
        )
