#!/usr/bin/env python3
"""
================================================================================
          AI-POWERED RESOURCE ALLOCATION SYSTEM
================================================================================
An intelligent system that:
1. Gathers all required business and technical specifications interactively from the user.
2. Converts natural language inputs into structured constraints.
3. Makes an optimal AI decision on resource allocation with multi-criteria scoring.
4. Explains WHY the resource was selected with itemized justifications.
5. Performs rigorous dimension-by-dimension gap analysis and remediation.

Selected Technical Approach: Hugging Face Pre-trained LLM with LoRA Fine-Tuning
Author: Chirag (Assignment Submission)
================================================================================
"""

import os
import sys
import json
import argparse
from typing import List, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import Resource, ParsedRequirement, CandidateEvaluation, AllocationResponse
from src.parser import RequirementParser, HuggingFaceLLMParser
from src.matcher import ResourceMatcher
from src.gap_analyzer import GapAnalyzer
from src.explainer import Explainer


class ResourceAllocationSystem:
    """
    Unified Orchestrator for the AI Resource Allocation Pipeline.
    Supports interactive requirement intake, Hugging Face LLM parsing,
    and deterministic constraint evaluation.
    """

    def __init__(self, dataset_path: Optional[str] = None, use_hf: bool = False, hf_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources.json")
        self.dataset_path = dataset_path
        
        hf_parser = None
        if use_hf:
            hf_parser = HuggingFaceLLMParser(model_name_or_path=hf_model_name)
            hf_parser.initialize()

        self.parser = RequirementParser(hf_parser=hf_parser)
        self.matcher = ResourceMatcher(dataset_path=self.dataset_path)
        self.explainer = Explainer()

    def process_query(self, natural_language_query: str) -> AllocationResponse:
        """
        Executes end-to-end allocation pipeline on a single query string.
        """
        req: ParsedRequirement = self.parser.parse(natural_language_query)
        evaluations: List[CandidateEvaluation] = self.matcher.match(req)
        response: AllocationResponse = self.explainer.explain(req, evaluations)
        return response

    def process_requirement(self, req: ParsedRequirement) -> AllocationResponse:
        """
        Evaluates an existing ParsedRequirement object and produces recommendations.
        """
        evaluations: List[CandidateEvaluation] = self.matcher.match(req)
        response: AllocationResponse = self.explainer.explain(req, evaluations)
        return response

    def display_response(self, response: AllocationResponse):
        """
        Formats and displays the allocation results cleanly in terminal.
        """
        req = response.requirement
        top_eval = response.top_evaluation
        top_res = response.recommended_resource

        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass

        if response.overall_status == "OUT_OF_DOMAIN":
            print("\n" + "=" * 70)
            print("       [!] OUT-OF-DOMAIN / UNRELATED QUERY DETECTED")
            print("=" * 70)
            print(f"Aggregated Query : \"{req.raw_query}\"")
            print(f"\n{response.natural_language_explanation}")
            print("\nScope of this Allocation System:")
            print("  * Physical Equipment : Boilers, Refrigerants/Chillers, Water Treatment Units")
            print("  * Human Resources    : Chemical Process Consultants, Thermal Engineers")
            print("\nExample Valid Queries:")
            print("  * 'I need a boiler with 5 years experience and budget under Rs. 60,000'")
            print("  * 'Need an ammonia chiller immediately for cooling'")
            print("  * 'Looking for a chemical consultant with 8 years experience'")
            print("=" * 70 + "\n")
            return

        print("\n" + "=" * 70)
        print("           STAGE 2: AI DECISION & RESOURCE ALLOCATION REPORT")
        print("=" * 70)

        # 1. Final Structured Requirements
        print("\n[1] CONFIRMED REQUIREMENT SPECIFICATION PROFILE")
        print("-" * 70)
        print(f"Aggregated Query : \"{req.raw_query}\"")
        spec_dict = {
            "Resource Type": req.resource_type,
            "Min Experience Required": f"{req.min_experience_years} years" if req.min_experience_years is not None else "Any / Not constrained",
            "Max Budget Ceiling": f"Rs. {req.max_budget_inr:,.2f}" if req.max_budget_inr is not None else "No upper limit specified",
            "Water Usage Throughput": f"{req.water_usage_lpd:,.0f} L/day" if req.water_usage_lpd is not None else "Not specified",
            "Chemical Processes": req.chemical_processes if req.chemical_processes else ["General / All compatible"],
            "Quantity": req.quantity_required,
            "Urgency / Timeline": req.urgency
        }
        for k, v in spec_dict.items():
            print(f"  * {k:<26}: {v}")

        if req.assumptions_made:
            print("\nOperational Assumptions Applied:")
            for a in req.assumptions_made:
                print(f"  * {a}")

        # 2. Recommended Resource
        print("\n[2] TOP RECOMMENDED RESOURCE ALLOCATION")
        print("-" * 70)
        if top_res and top_eval:
            avail_status = "Available (Ready for deployment)" if top_res.available else f"Unavailable (Scheduled: {top_res.next_available_date})"
            print(f"Recommended Resource : {top_res.name} (ID: {top_res.id})")
            print(f"Category / Type      : {top_res.category} / {top_res.type}")
            print(f"Experience           : {top_res.experience_years} years")
            print(f"Cost                 : Rs. {top_res.cost_inr:,.2f}")
            print(f"Availability         : {avail_status}")
            print(f"Location             : {top_res.location}")
            print(f"Compatibility        : {', '.join(top_res.chemical_compatibility)}")
            print(f"Overall Match Score  : {top_eval.overall_score}/100.0 (Status: {response.overall_status})")
        else:
            print("No suitable resource found in database.")

        # 3. Justification (Why this resource?)
        print("\n[3] WHY THIS RESOURCE WAS SELECTED (ALLOCATION JUSTIFICATION)")
        print("-" * 70)
        if top_eval and top_eval.justification_points:
            for pt in top_eval.justification_points:
                print(f"  [OK] {pt}")
        else:
            print("  - Insufficient match criteria satisfied.")

        # 4. Non-Selection Explanations (Why other candidates were not chosen)
        if response.non_selection_report:
            print("\n[4] WHY OTHER CANDIDATES WERE NOT ALLOCATED (NON-SELECTION ANALYSIS)")
            print("-" * 70)
            print(response.non_selection_report)

        # 5. Gap Analysis for top choice
        print("\n[5] GAP ANALYSIS & REMEDIATION (TOP CHOICE)")
        print("-" * 70)
        print(response.gap_analysis_summary)

        # 6. Candidate Ranking Table
        print("\n[6] ALL EVALUATED CANDIDATES IN INVENTORY")
        print("-" * 70)
        header = f"{'ID':<6} | {'Name':<28} | {'Exp':<5} | {'Cost (INR)':<11} | {'Avail':<6} | {'Score':<5} | {'Key Factor / Shortfall'}"
        print(header)
        print("-" * 95)
        for ev in response.all_evaluations:
            r = ev.resource
            avail_str = "Yes" if r.available else "No"
            is_top = (r.id == top_res.id) if top_res else False
            status_tag = "[SELECTED]" if is_top else ev.rejection_summary
            print(f"{r.id:<6} | {r.name[:28]:<28} | {r.experience_years:>2}yr | Rs.{r.cost_inr:>7,.0f} | {avail_str:<6} | {ev.overall_score:>5.1f} | {status_tag}")
        print("=" * 70 + "\n")


def display_profile_status(req: ParsedRequirement):
    """
    Displays the current gathered specification profile and pending items.
    """
    print("\n" + "=" * 70)
    print("           STAGE 1: REQUIREMENT GATHERING & INTAKE PROFILE")
    print("=" * 70)
    
    type_display = req.resource_type if req.resource_type != "Any" else "[PENDING / NOT SPECIFIED]"
    exp_display = f"{req.min_experience_years} years" if req.min_experience_years is not None else "[PENDING / NOT SPECIFIED]"
    budget_display = f"Rs. {req.max_budget_inr:,.2f}" if req.max_budget_inr is not None else "[PENDING / NOT SPECIFIED]"
    water_display = f"{req.water_usage_lpd:,.0f} L/day" if req.water_usage_lpd is not None else "[PENDING / NOT SPECIFIED]"
    chem_display = ", ".join(req.chemical_processes) if req.chemical_processes else "[PENDING / NOT SPECIFIED]"

    print(f"  1. Resource Type       : {type_display}")
    print(f"  2. Experience Required : {exp_display}")
    print(f"  3. Budget Ceiling      : {budget_display}")
    if req.resource_type in ["Boiler", "Water Treatment", "Any"]:
        print(f"  4. Water Throughput    : {water_display}")
    if req.resource_type in ["Boiler", "Refrigerant", "Water Treatment", "Consultant", "Any"]:
        print(f"  5. Chemical Processes  : {chem_display}")
    print(f"  6. Quantity Required   : {req.quantity_required}")
    print(f"  7. Urgency / Timeline  : {req.urgency}")
    print("=" * 70)


def gather_requirements_interactive(system: ResourceAllocationSystem, initial_text: Optional[str] = None) -> Optional[ParsedRequirement]:
    """
    Interactive multi-turn requirement gathering intake wizard.
    Gathers all missing specifications from the user before executing the AI allocation decision.
    """
    if initial_text:
        req = system.parser.parse(initial_text)
    else:
        req = ParsedRequirement(raw_query="")

    display_profile_status(req)

    # 1. Ask for Resource Type if missing
    if req.resource_type == "Any":
        print("\n[?] What category of resource do you require?")
        print("    Examples: 'Boiler', 'Refrigerant', 'Water Treatment', 'Consultant', 'Engineer'")
        user_val = input("    Resource Type >> ").strip()
        if user_val.lower() in ["exit", "quit", "q"]:
            return None
        if user_val:
            req = system.parser.merge_requirements(req, user_val)
            display_profile_status(req)

    # 2. Ask for Experience if missing
    if req.min_experience_years is None:
        print("\n[?] What is the minimum operational experience or vintage required (in years)?")
        print("    (e.g., '5 years', '8 yrs', or press Enter / type 'skip' for any experience)")
        user_val = input("    Minimum Experience >> ").strip()
        if user_val.lower() in ["exit", "quit", "q"]:
            return None
        if user_val and user_val.lower() not in ["skip", "none", "no"]:
            req = system.parser.merge_requirements(req, f"{user_val} years experience")
            display_profile_status(req)

    # 3. Ask for Budget Ceiling if missing
    if req.max_budget_inr is None:
        print("\n[?] What is your maximum budget ceiling in INR?")
        print("    (e.g., 'Rs. 60,000', '100000', '45k', or press Enter / type 'skip' if flexible)")
        user_val = input("    Budget Ceiling >> ").strip()
        if user_val.lower() in ["exit", "quit", "q"]:
            return None
        if user_val and user_val.lower() not in ["skip", "none", "no"]:
            req = system.parser.merge_requirements(req, f"budget {user_val}")
            display_profile_status(req)

    # 4. Ask for Water Throughput / Capacity if applicable and missing
    if req.water_usage_lpd is None and req.resource_type in ["Boiler", "Water Treatment"]:
        print(f"\n[?] What is your daily water throughput / capacity requirement for this {req.resource_type}?")
        print("    (e.g., '5000 liters/day', '8000 L/day', or press Enter / type 'skip')")
        user_val = input("    Water Throughput >> ").strip()
        if user_val.lower() in ["exit", "quit", "q"]:
            return None
        if user_val and user_val.lower() not in ["skip", "none", "no"]:
            req = system.parser.merge_requirements(req, f"{user_val} water usage")
            display_profile_status(req)

    # 5. Ask for Chemical Processes / Media if missing
    if not req.chemical_processes and req.resource_type in ["Boiler", "Refrigerant", "Water Treatment", "Consultant"]:
        print("\n[?] What chemical processes or media will this equipment/consultant handle?")
        print("    Examples: 'Acids', 'Organic Solvents', 'Ammonia', 'Polymers', 'Petrochemicals'")
        print("    (Type chemicals or press Enter / type 'skip' for general processing)")
        user_val = input("    Chemical Processes >> ").strip()
        if user_val.lower() in ["exit", "quit", "q"]:
            return None
        if user_val and user_val.lower() not in ["skip", "none", "no"]:
            req = system.parser.merge_requirements(req, f"handles {user_val}")
            display_profile_status(req)

    # Summary confirmation
    print("\n[AI Assistant] All specifications successfully gathered!")
    return req


def interactive_mode(system: ResourceAllocationSystem):
    """
    Main interactive console supporting Guided Intake and AI Decision.
    """
    print("\n" + "=" * 70)
    print("          AI-POWERED RESOURCE ALLOCATION - INTERACTIVE CLI")
    print("=" * 70)
    print("Mode: Interactive Guided Requirement Intake & Decision")
    print("Commands: 'demo' (run benchmarks), 'exit' (quit)\n")

    while True:
        try:
            print("\nEnter your initial requirement query in plain English:")
            print("(Example: 'I need a boiler with 20 years experience and budget below 100000')")
            print("Or press [Enter] to start step-by-step guided intake.")
            
            user_input = input("\nEnter requirement >> ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Resource Allocation Assistant. Goodbye!")
                break

            if user_input.lower() == "demo":
                run_demo_scenarios(system)
                continue

            # Stage 1: Gather all requirements from user
            final_req = gather_requirements_interactive(system, initial_text=user_input if user_input else None)
            if final_req is None:
                break

            # Stage 2: Execute AI Decision & Allocation
            print("\n[AI Decision Engine] Evaluating candidate resources against gathered constraints...")
            response = system.process_requirement(final_req)
            system.display_response(response)

            # Post-decision action prompt
            next_action = input("\nWould you like to allocate another resource? (y/n, default y): ").strip().lower()
            if next_action in ["n", "no", "exit", "quit", "q"]:
                print("Thank you for using the AI Resource Allocation System. Goodbye!")
                break

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def run_demo_scenarios(system: ResourceAllocationSystem):
    """
    Executes sample test scenarios showcasing diverse capabilities.
    """
    scenarios = [
        (
            "Scenario 1: Standard Requirement (Exact Match)",
            "I need resources for a chemical processing project that uses 5000 liters of water per day. The project requires a boiler with at least 5 years of experience and the budget is Rs. 60,000."
        ),
        (
            "Scenario 2: Strict Constraint with Gap Analysis (Budget & Experience Gap)",
            "I need an industrial boiler with at least 10 years experience and maximum budget of Rs. 50,000."
        ),
        (
            "Scenario 3: Ambiguous / Incomplete Query (Handling Missing Data & Assumptions)",
            "I need a boiler for my chemical project."
        ),
        (
            "Scenario 4: Urgent Cooling Unit (Ammonia Refrigerant Match)",
            "Urgent requirement: Need industrial refrigerant chiller for ammonia cooling, budget Rs. 35,000, immediately available."
        ),
        (
            "Scenario 5: Consulting / Human Resource Allocation",
            "We need an experienced chemical process consultant with 8+ years experience in polymer synthesis, budget up to Rs. 80,000."
        )
    ]

    print("\n" + "#" * 70)
    print("       RUNNING AUTOMATED TEST SCENARIOS FOR EVALUATION")
    print("#" * 70)

    for title, query in scenarios:
        print(f"\n>>> Running: {title}")
        response = system.process_query(query)
        system.display_response(response)


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Resource Allocation System (Hugging Face / Semantic Matcher)")
    parser.add_argument("--demo", "-d", action="store_true", help="Run automated evaluation benchmark scenarios")
    parser.add_argument("--query", "-q", type=str, help="Process a single natural language query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive guided intake interface")
    parser.add_argument("--use-hf", action="store_true", help="Enable Hugging Face pre-trained/fine-tuned model for extraction")
    parser.add_argument("--hf-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face model ID or checkpoint directory")

    args = parser.parse_args()
    system = ResourceAllocationSystem(use_hf=args.use_hf, hf_model_name=args.hf_model)

    if args.demo:
        run_demo_scenarios(system)
    elif args.query:
        response = system.process_query(args.query)
        system.display_response(response)
    elif args.interactive:
        interactive_mode(system)
    else:
        # Default behavior: Show interactive menu
        print("\nWelcome to the AI-Powered Resource Allocation System!")
        print("Selected Technical Approach: Hugging Face Pre-trained LLM + Deterministic Constraints")
        print("1. Start Interactive Requirement Gathering & Allocation")
        print("2. Run Automated Test Scenarios (Demo)")
        print("3. Exit")

        try:
            choice = input("\nSelect option [1/2/3] (default 1): ").strip()
            if choice == "2":
                run_demo_scenarios(system)
            elif choice == "3":
                print("Goodbye!")
            else:
                interactive_mode(system)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")


if __name__ == "__main__":
    main()
