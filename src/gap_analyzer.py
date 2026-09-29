"""
Dimension-by-Dimension Gap Analysis Engine for Resource Allocation.
Detects missing capabilities, budget deficits, timeline conflicts, and provides remediation strategies.
"""
from typing import List
from .models import Resource, ParsedRequirement, DimensionGap, GapReport


class GapAnalyzer:
    """
    Evaluates disparities between requirements and candidate resource attributes.
    """

    def analyze(self, req: ParsedRequirement, res: Resource) -> GapReport:
        gaps: List[DimensionGap] = []
        critical_blockers: List[str] = []
        remediations: List[str] = []

        # 1. Resource Type Check
        if req.resource_type != "Any" and req.resource_type.lower() != res.type.lower():
            gaps.append(DimensionGap(
                dimension="Resource Type",
                required_value=req.resource_type,
                actual_value=res.type,
                gap_description=f"Resource type mismatch: Required '{req.resource_type}', but '{res.name}' is a '{res.type}'.",
                severity="Critical",
                is_satisfied=False
            ))
            critical_blockers.append(f"Type mismatch ({res.type} != {req.resource_type})")

        # 2. Availability Check
        if not res.available:
            next_date = res.next_available_date or "Unknown"
            gaps.append(DimensionGap(
                dimension="Availability",
                required_value="Immediate / Available",
                actual_value="Unavailable (In Use / Maintenance)",
                gap_description=f"Resource is currently unavailable. Next scheduled availability: {next_date}.",
                severity="Major" if req.urgency == "Immediate" else "Minor",
                is_satisfied=False
            ))
            if req.urgency == "Immediate":
                critical_blockers.append("Resource unavailable for immediate requirement")
                remediations.append(f"Reschedule project deployment to {next_date} or select an immediately available alternative.")
            else:
                remediations.append(f"Book resource in advance for {next_date}.")
        else:
            gaps.append(DimensionGap(
                dimension="Availability",
                required_value="Available",
                actual_value="Available",
                gap_description="Resource is currently online and ready for deployment.",
                severity="None",
                is_satisfied=True
            ))

        # 3. Experience Gap Analysis
        if req.min_experience_years is not None:
            diff = res.experience_years - req.min_experience_years
            if diff < 0:
                deficit = abs(diff)
                severity = "Critical" if deficit >= 4 else "Major" if deficit >= 2 else "Minor"
                gaps.append(DimensionGap(
                    dimension="Experience",
                    required_value=f"{req.min_experience_years} years",
                    actual_value=f"{res.experience_years} years",
                    gap_description=f"Experience deficit of {deficit} year(s) (Required: {req.min_experience_years} yrs, Resource: {res.experience_years} yrs).",
                    severity=severity,
                    is_satisfied=False
                ))
                remediations.append(
                    f"Accept {res.experience_years} years experience or pair equipment with a senior certified operator."
                )
            else:
                surplus = diff
                gaps.append(DimensionGap(
                    dimension="Experience",
                    required_value=f"{req.min_experience_years} years",
                    actual_value=f"{res.experience_years} years (+{surplus} yrs surplus)",
                    gap_description=f"Experience requirement fully satisfied with {surplus} years surplus.",
                    severity="None",
                    is_satisfied=True
                ))

        # 4. Budget Gap Analysis
        if req.max_budget_inr is not None:
            overrun = res.cost_inr - req.max_budget_inr
            if overrun > 0:
                severity = "Critical" if overrun > 25000 else "Major" if overrun > 10000 else "Minor"
                gaps.append(DimensionGap(
                    dimension="Budget / Cost",
                    required_value=f"Rs. {req.max_budget_inr:,.2f}",
                    actual_value=f"Rs. {res.cost_inr:,.2f}",
                    gap_description=f"Budget exceeded by Rs. {overrun:,.2f} ({((res.cost_inr/req.max_budget_inr)-1)*100:.1f}% overrun).",
                    severity=severity,
                    is_satisfied=False
                ))
                remediations.append(f"Increase project budget ceiling by Rs. {overrun:,.2f} to accommodate this resource.")
            else:
                savings = abs(overrun)
                gaps.append(DimensionGap(
                    dimension="Budget / Cost",
                    required_value=f"Rs. {req.max_budget_inr:,.2f}",
                    actual_value=f"Rs. {res.cost_inr:,.2f} (Savings: Rs. {savings:,.2f})",
                    gap_description=f"Cost is within budget, generating Rs. {savings:,.2f} in operational savings.",
                    severity="None",
                    is_satisfied=True
                ))

        # 5. Water Capacity Gap Analysis
        if req.water_usage_lpd is not None and res.water_capacity_lpd is not None:
            if res.water_capacity_lpd < req.water_usage_lpd:
                deficit_lpd = req.water_usage_lpd - res.water_capacity_lpd
                gaps.append(DimensionGap(
                    dimension="Water Capacity",
                    required_value=f"{req.water_usage_lpd:,.0f} L/day",
                    actual_value=f"{res.water_capacity_lpd:,.0f} L/day",
                    gap_description=f"Water throughput capacity deficit of {deficit_lpd:,.0f} L/day.",
                    severity="Major",
                    is_satisfied=False
                ))
                remediations.append(f"Deploy an auxiliary water booster/treatment unit (e.g., W001) to supply the extra {deficit_lpd:,.0f} L/day.")
            else:
                gaps.append(DimensionGap(
                    dimension="Water Capacity",
                    required_value=f"{req.water_usage_lpd:,.0f} L/day",
                    actual_value=f"{res.water_capacity_lpd:,.0f} L/day",
                    gap_description="Water throughput capacity meets and exceeds required daily volume.",
                    severity="None",
                    is_satisfied=True
                ))

        # 6. Chemical Compatibility Gap Analysis
        if req.chemical_processes:
            matched_chems = []
            unmatched_chems = []
            res_chems_lower = [c.lower() for c in res.chemical_compatibility]

            for chem in req.chemical_processes:
                chem_l = chem.lower()
                if any(chem_l in rc or rc in chem_l for rc in res_chems_lower):
                    matched_chems.append(chem)
                else:
                    unmatched_chems.append(chem)

            if unmatched_chems:
                gaps.append(DimensionGap(
                    dimension="Chemical Compatibility",
                    required_value=", ".join(req.chemical_processes),
                    actual_value=", ".join(res.chemical_compatibility) or "None explicitly listed",
                    gap_description=f"Unverified compatibility for process media: {', '.join(unmatched_chems)}.",
                    severity="Minor" if matched_chems else "Major",
                    is_satisfied=False
                ))
                remediations.append(f"Conduct metallurgy / coating audit or add neutralizer before feeding {', '.join(unmatched_chems)}.")
            else:
                gaps.append(DimensionGap(
                    dimension="Chemical Compatibility",
                    required_value=", ".join(req.chemical_processes),
                    actual_value=", ".join(res.chemical_compatibility),
                    gap_description="Full compatibility confirmed for all designated chemical processes.",
                    severity="None",
                    is_satisfied=True
                ))

        has_gaps = any(not g.is_satisfied for g in gaps)

        return GapReport(
            has_gaps=has_gaps,
            critical_blockers=critical_blockers,
            dimension_gaps=gaps,
            remediation_suggestions=remediations
        )
