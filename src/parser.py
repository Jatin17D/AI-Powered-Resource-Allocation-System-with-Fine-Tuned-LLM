"""
Natural Language Requirement Parser for Resource Allocation.
Converts unstructured business queries into structured requirements,
handles ambiguity, missing parameters, and explicit assumptions.
Includes first-class Hugging Face LLM integration and interactive multi-turn specification merging.
"""
import os
import re
import json
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from .models import ParsedRequirement

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)


class HuggingFaceLLMParser:
    """
    Hugging Face LLM Parser for requirement extraction.
    Supports both:
    1. Hugging Face Serverless Inference API (via HF_TOKEN) - Instant, lightweight, no local GPU needed.
    2. Local Hugging Face Transformers pipeline (via PyTorch/Transformers/LoRA checkpoints).
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.model_name_or_path = model_name_or_path or os.getenv("HF_MODEL_NAME") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = device
        self.inference_client = None
        self.local_pipeline = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """
        Attempts to initialize Hugging Face Inference API client or local pipeline.
        """
        # 1. Try Hugging Face Inference Client if token is available
        if self.hf_token and self.hf_token.strip() and not self.hf_token.startswith("your_"):
            try:
                from huggingface_hub import InferenceClient
                self.inference_client = InferenceClient(model=self.model_name_or_path, token=self.hf_token)
                self._is_initialized = True
                logger.info(f"Initialized Hugging Face Inference API client for model: {self.model_name_or_path}")
                return True
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to initialize HF InferenceClient: {e}")

        # 2. Try Local Transformers / LoRA Pipeline
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Check if this is a fine-tuned LoRA adapter directory
            adapter_config_path = os.path.join(self.model_name_or_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                from peft import PeftModel, PeftConfig
                peft_config = PeftConfig.from_pretrained(self.model_name_or_path)
                base_model_name = peft_config.base_model_name_or_path
                logger.info(f"Loading base model '{base_model_name}' and fine-tuned LoRA adapter from '{self.model_name_or_path}'")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=self.hf_token)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    token=self.hf_token,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                model = PeftModel.from_pretrained(base_model, self.model_name_or_path)
            else:
                logger.info(f"Loading local Hugging Face model '{self.model_name_or_path}' on device: {self.device}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, token=self.hf_token)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    token=self.hf_token,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            self._is_initialized = True
            return True
        except Exception as e:
            logger.warning(f"Could not initialize local Hugging Face model ({e}). Using deterministic semantic fallback.")
            self._is_initialized = False
            return False

    def extract_json(self, natural_query: str) -> Optional[Dict[str, Any]]:
        """
        Infers JSON requirement slots from query using Hugging Face.
        """
        if not self._is_initialized:
            if not self.initialize():
                return None

        prompt = (
            f"You are an expert industrial resource allocation assistant. "
            f"Extract technical specifications from the user request and return ONLY a JSON object with keys: "
            f"resource_type (string), min_experience_years (integer or null), max_budget_inr (number or null), "
            f"water_usage_lpd (number or null), chemical_processes (list of strings), quantity_required (integer), urgency (string).\n\n"
            f"User Request: \"{natural_query}\"\n\n"
            f"JSON Response:"
        )

        # Method A: Hugging Face Inference Client (Cloud Serverless API)
        if self.inference_client:
            try:
                chat_response = self.inference_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert industrial resource allocation parser. "
                                "Extract technical and business requirements from user queries into a JSON object. "
                                "Return ONLY raw JSON with keys: resource_type (string, e.g. Boiler, Refrigerant, Water Treatment, Consultant, Engineer), "
                                "min_experience_years (integer or null), max_budget_inr (float or null), "
                                "water_usage_lpd (float or null), chemical_processes (list of strings), quantity_required (integer), urgency (string)."
                            )
                        },
                        {
                            "role": "user",
                            "content": natural_query
                        }
                    ],
                    max_tokens=256,
                    temperature=0.1
                )
                raw_text = chat_response.choices[0].message.content.strip()
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            except Exception as e:
                logger.warning(f"HF Chat Inference API failed ({e}). Trying fallback...")

        # Method B: Local Transformers / LoRA Pipeline
        if self.local_pipeline:
            try:
                outputs = self.local_pipeline(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False
                )
                generated_text = outputs[0]["generated_text"][len(prompt):].strip()
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            except Exception as e:
                logger.warning(f"Local HF model generation failed: {e}")

        return None


class RequirementParser:
    """
    Parses natural language business requirements into structured specifications.
    Supports Hugging Face LLMs, multi-turn requirement merging, and robust rule-based semantic extraction.
    """

    KNOWN_RESOURCE_TYPES = {
        "boiler": "Boiler",
        "steam generator": "Boiler",
        "fire-tube": "Boiler",
        "water-tube": "Boiler",
        "refrigerant": "Refrigerant",
        "chiller": "Refrigerant",
        "cooling unit": "Refrigerant",
        "cryogenic": "Refrigerant",
        "deep freeze": "Refrigerant",
        "water treatment": "Water Treatment",
        "ro plant": "Water Treatment",
        "demineralization": "Water Treatment",
        "filtration": "Water Treatment",
        "consultant": "Consultant",
        "chemical consultant": "Consultant",
        "process consultant": "Consultant",
        "engineer": "Engineer",
        "thermal engineer": "Engineer",
        "specialist": "Specialist",
        "technician": "Engineer",
    }

    KNOWN_CHEMICAL_KEYWORDS = {
        "acid": "Acids",
        "acids": "Acids",
        "solvent": "Solvents",
        "solvents": "Solvents",
        "organic solvents": "Organic Solvents",
        "polymer": "Polymers",
        "polymers": "Polymers",
        "petrochemical": "Petrochemicals",
        "petrochemicals": "Petrochemicals",
        "ammonia": "Ammonia",
        "caustic": "Caustics",
        "caustics": "Caustics",
        "hydrocarbon": "Hydrocarbons",
        "hydrocarbons": "Hydrocarbons",
        "brine": "Brine",
        "chemical processing": "General Chemical Processing",
        "food-grade": "Food-Grade Chemicals",
        "cryogenic": "Cryogenic Liquefaction",
        "steam systems": "Steam Systems"
    }

    OUT_OF_DOMAIN_PATTERNS = [
        r'\bweather\b', r'\bforecast\b', r'\btemperature\b', r'\brain\b', r'\bsnow\b',
        r'\bjoke\b', r'\brecipe\b', r'\bcook\b', r'\bbake\b', r'\bmovie\b', r'\bsong\b',
        r'\bpoem\b', r'\bwho is\b', r'\bwho won\b', r'\bcricket\b', r'\bcricketer\b',
        r'\bsport\b', r'\bsports\b', r'\bfootball\b', r'\bsoccer\b', r'\bplayer\b',
        r'\bcelebrity\b', r'\bactor\b', r'\bactress\b', r'\bpolitics\b', r'\bpresident\b',
        r'\bprime minister\b', r'\bcapital of\b', r'\bhow are you\b', r'\bwhat is your name\b',
        r'\btell me a story\b', r'\bstock market\b', r'\bbitcoin\b', r'\bcrypto\b',
        r'\btranslate\b', r'\bgold price\b', r'\bnews\b', r'\bhoroscope\b', r'\bastrology\b'
    ]

    DOMAIN_KEYWORDS = [
        "resource", "equipment", "machine", "hire", "allocate", "procure", "worker",
        "staff", "personnel", "engineer", "consultant", "unit", "plant", "boiler",
        "chiller", "refrigerant", "water", "chemical", "capacity", "budget", "cost",
        "experience", "lpd", "liters", "litres", "inr", "rs", "steam", "cooling", "treatment"
    ]

    def _is_out_of_domain(self, text: str, resource_type: str, has_specs: bool) -> bool:
        if not text or not text.strip():
            return False
        lower = text.lower().strip()
        # 1. Direct match for known out-of-domain conversational queries
        for pattern in self.OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, lower):
                return True

        # 2. General conversational questions without technical domain context
        conversational_starts = [
            "who is ", "who are ", "who was ", "tell me about ", "what is the capital",
            "how to make ", "how to play ", "lyrics of ", "sing a ", "write a poem",
            "tell me a joke", "best cricketer", "best player", "best team"
        ]
        if any(lower.startswith(prefix) for prefix in conversational_starts):
            return True

        # 3. If no resource type matched and no technical specs extracted
        if resource_type == "Any" and not has_specs:
            has_domain_term = any(term in lower for term in self.DOMAIN_KEYWORDS)
            if not has_domain_term:
                return True
        return False

    def __init__(self, hf_parser: Optional[HuggingFaceLLMParser] = None):
        self.hf_parser = hf_parser

    def parse(self, query: str) -> ParsedRequirement:
        """
        Parse a natural language query into a ParsedRequirement object.
        """
        if not query or not query.strip():
            return ParsedRequirement(
                raw_query="",
                is_ambiguous=True,
                missing_fields=["resource_type", "max_budget_inr", "min_experience_years"],
                clarification_questions=[
                    "What type of resource do you require (e.g. Boiler, Refrigerant, Water Treatment, Consultant)?",
                    "What is your allocated maximum budget in INR?",
                    "What are your minimum experience or operational requirements?"
                ]
            )

        # 1. If Hugging Face LLM parser is provided, try LLM inference first
        if self.hf_parser:
            try:
                hf_data = self.hf_parser.extract_json(query)
                if hf_data and isinstance(hf_data, dict):
                    return self._build_from_dict(query, hf_data)
            except Exception as e:
                logger.warning(f"HF Parser error ({e}), falling back to deterministic parser.")

        # 2. Rule-based semantic parsing (Robust, Instant & Deterministic)
        return self._parse_semantic_rules(query)

    def merge_requirements(self, base_req: ParsedRequirement, update_query: str) -> ParsedRequirement:
        """
        Merges new user-supplied specifications into an existing ParsedRequirement.
        Allows interactive multi-turn query modification and refinement.
        """
        new_parsed = self.parse(update_query)

        # Merge fields (override if newly provided)
        resource_type = new_parsed.resource_type if new_parsed.resource_type != "Any" else base_req.resource_type
        min_experience = new_parsed.min_experience_years if new_parsed.min_experience_years is not None else base_req.min_experience_years
        max_budget = new_parsed.max_budget_inr if new_parsed.max_budget_inr is not None else base_req.max_budget_inr
        water_usage = new_parsed.water_usage_lpd if new_parsed.water_usage_lpd is not None else base_req.water_usage_lpd
        
        # Merge chemical processes
        chems = list(dict.fromkeys(base_req.chemical_processes + new_parsed.chemical_processes))
        if "General Chemical Processing" in chems and len(chems) > 1:
            chems.remove("General Chemical Processing")

        quantity = new_parsed.quantity_required if new_parsed.quantity_required > 1 else base_req.quantity_required
        urgency = new_parsed.urgency if new_parsed.urgency != "Normal" else base_req.urgency

        merged_raw = f"{base_req.raw_query} | Refinement: {update_query}" if base_req.raw_query else update_query

        # Recalculate missing fields and clarification questions
        missing_fields = []
        assumptions_made = []
        clarification_questions = []

        if resource_type == "Any":
            missing_fields.append("resource_type")
            clarification_questions.append("What type of resource do you need (e.g. Boiler, Refrigerant, Water Treatment, Consultant)?")
            assumptions_made.append("Defaulting to physical industrial equipment search across all categories.")

        if min_experience is None:
            missing_fields.append("min_experience_years")
            clarification_questions.append("What is the minimum operating experience or vintage required (in years)?")
            assumptions_made.append("No minimum experience constraint specified; all operational levels considered.")

        if max_budget is None:
            missing_fields.append("max_budget_inr")
            clarification_questions.append("What is your maximum target budget ceiling in INR?")
            assumptions_made.append("No budget ceiling specified; prioritizing the most cost-effective match.")

        if water_usage is None and resource_type in ["Boiler", "Water Treatment"]:
            missing_fields.append("water_usage_lpd")
            clarification_questions.append(f"What is your required daily water throughput / capacity for this {resource_type} (in Liters/day)?")
            assumptions_made.append("Water throughput not specified; assuming standard equipment capacity.")

        if not chems and resource_type in ["Boiler", "Refrigerant", "Water Treatment", "Consultant"]:
            missing_fields.append("chemical_processes")
            clarification_questions.append("What chemical processes or fluids will be involved (e.g., Acids, Solvents, Polymers, Ammonia)?")
            assumptions_made.append("Chemical process compatibility not specified; searching across all compatible units.")

        is_ambiguous = len(missing_fields) >= 2 or resource_type == "Any"

        return ParsedRequirement(
            raw_query=merged_raw,
            resource_type=resource_type,
            min_experience_years=min_experience,
            max_budget_inr=max_budget,
            water_usage_lpd=water_usage,
            chemical_processes=chems,
            quantity_required=quantity,
            urgency=urgency,
            is_ambiguous=is_ambiguous,
            missing_fields=missing_fields,
            assumptions_made=assumptions_made,
            clarification_questions=clarification_questions
        )

    def _build_from_dict(self, raw_query: str, data: Dict[str, Any]) -> ParsedRequirement:
        r_type = data.get("resource_type", "Any")
        if r_type:
            r_type = r_type.title()

        min_exp = data.get("min_experience_years") or data.get("minimum_experience")
        max_budget = data.get("max_budget_inr") or data.get("maximum_cost") or data.get("budget")
        water_usage = data.get("water_usage_lpd") or data.get("water_usage")
        chems_raw = data.get("chemical_processes") or []
        if isinstance(chems_raw, str):
            chems = [chems_raw]
        elif isinstance(chems_raw, list):
            chems = [str(c) for c in chems_raw if c]
        else:
            chems = []
        qty = data.get("quantity_required", 1)
        urgency = data.get("urgency", "Normal") or "Normal"

        missing = []
        assumptions = []
        clarifications = []

        if not min_exp:
            missing.append("min_experience_years")
            clarifications.append("What is the minimum operating experience required (in years)?")
            assumptions.append("No minimum experience constraint specified; all operational levels considered.")
        if not max_budget:
            missing.append("max_budget_inr")
            clarifications.append("What is your maximum target budget in INR?")
            assumptions.append("No budget ceiling specified; prioritizing the most cost-effective match.")
        if not water_usage and r_type in ["Boiler", "Water Treatment"]:
            missing.append("water_usage_lpd")
            clarifications.append("What is the daily water usage or throughput (in Liters/day)?")
            assumptions.append("Water throughput assumed at standard capacity.")
        if not chems:
            missing.append("chemical_processes")
            clarifications.append("What chemical processes or media will this resource handle?")

        has_specs = (
            min_exp is not None or
            max_budget is not None or
            water_usage is not None or
            len(chems) > 0
        )
        is_unrelated = self._is_out_of_domain(raw_query, r_type, has_specs)
        is_ambiguous = len(missing) >= 2 or is_unrelated

        return ParsedRequirement(
            raw_query=raw_query,
            resource_type=r_type if r_type in ["Boiler", "Refrigerant", "Water Treatment", "Consultant", "Engineer"] else "Any",
            min_experience_years=int(min_exp) if min_exp is not None else None,
            max_budget_inr=float(max_budget) if max_budget is not None else None,
            water_usage_lpd=float(water_usage) if water_usage is not None else None,
            chemical_processes=[c.title() for c in chems],
            quantity_required=int(qty) if qty else 1,
            urgency=urgency,
            is_ambiguous=is_ambiguous,
            is_unrelated=is_unrelated,
            missing_fields=missing,
            assumptions_made=assumptions,
            clarification_questions=clarifications
        )

    def _parse_semantic_rules(self, text: str) -> ParsedRequirement:
        clean_text = text.strip()
        lower_text = clean_text.lower()

        # 1. Extract Resource Type
        resource_type = "Any"
        for keyword, standard_type in self.KNOWN_RESOURCE_TYPES.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                resource_type = standard_type
                break

        # 2. Extract Water Usage (e.g., 5000 liters, 5000 L/day, 8000lpd, 5k liters)
        water_usage_lpd = None
        water_patterns = [
            r'(\d+[\d,]*)\s*(?:liters?|litres?|l|lpd|liters/day|l/day)\s*(?:of\s+water)?',
            r'(?:water\s+usage|water\s+consumption|capacity|uses)\s*(?:of\s*)?(\d+[\d,]*)\s*(?:liters?|l|lpd)?',
        ]
        for pattern in water_patterns:
            match = re.search(pattern, lower_text)
            if match:
                val_str = match.group(1).replace(",", "")
                try:
                    water_usage_lpd = float(val_str)
                    break
                except ValueError:
                    pass

        # 3. Extract Experience (e.g. 5 years, at least 5 years, 8+ yrs, 5 yrs of experience)
        min_experience_years = None
        exp_patterns = [
            r'(?:at\s+least|minimum|min|over|above|>=)?\s*(\d+)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s*(?:of)?\s*experience|\s*exp)?',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
            r'experience\s*(?:of|at least|>=|:)?\s*(\d+)\s*(?:years?|yrs?)?',
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, lower_text)
            if match:
                try:
                    min_experience_years = int(match.group(1))
                    break
                except ValueError:
                    pass

        # 4. Extract Budget / Cost (e.g. Rs. 60,000, Rs 60000, INR 55,000, ₹60,000, budget is 60000, below 60k, max 50000)
        max_budget_inr = None
        # Replace shorthand like 60k -> 60000
        normalized_text = re.sub(r'(\d+)\s*k\b', lambda m: str(int(m.group(1)) * 1000), lower_text)

        budget_patterns = [
            r'(?:rs\.?|inr|₹)\s*(\d+[\d,]*)',
            r'budget\s*(?:is|of|under|below|max|maximum|up to)?\s*(?:rs\.?|inr|₹)?\s*(\d+[\d,]*)',
            r'(?:cost|price)\s*(?:under|below|less than|max|within|<=)?\s*(?:rs\.?|inr|₹)?\s*(\d+[\d,]*)',
            r'(?:max|maximum|under|below|within)\s*(?:rs\.?|inr|₹)?\s*(\d+[\d,]*)',
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, normalized_text)
            if match:
                val_str = match.group(1).replace(",", "")
                try:
                    max_budget_inr = float(val_str)
                    break
                except ValueError:
                    pass

        # 5. Extract Chemical Processes
        chemical_processes = []
        for kw, canonical in self.KNOWN_CHEMICAL_KEYWORDS.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', lower_text):
                chemical_processes.append(canonical)
        # Deduplicate
        chemical_processes = list(dict.fromkeys(chemical_processes))
        if not chemical_processes and "chemical" in lower_text:
            chemical_processes.append("General Chemical Processing")

        # 6. Extract Quantity
        quantity = 1
        qty_match = re.search(r'\b(\d+)\s*(?:boilers?|refrigerants?|chillers?|units?|engineers?|consultants?)\b', lower_text)
        if qty_match:
            try:
                quantity = int(qty_match.group(1))
            except ValueError:
                quantity = 1

        # 7. Extract Urgency
        urgency = "Normal"
        if any(term in lower_text for term in ["immediate", "urgent", "asap", "right now", "today"]):
            urgency = "Immediate"
        elif any(term in lower_text for term in ["next month", "scheduled", "later", "future"]):
            urgency = "Scheduled"

        # 8. Detect Ambiguity & Missing Parameters
        missing_fields = []
        assumptions_made = []
        clarification_questions = []

        if resource_type == "Any":
            missing_fields.append("resource_type")
            clarification_questions.append("What category of resource do you need (e.g. Boiler, Refrigerant, Water Treatment, Consultant)?")
            assumptions_made.append("Defaulting to physical industrial equipment search across all categories.")

        if min_experience_years is None:
            missing_fields.append("min_experience_years")
            clarification_questions.append("What is the minimum operating experience or vintage required (in years)?")
            assumptions_made.append("No minimum experience constraint specified; all operational experience levels considered.")

        if max_budget_inr is None:
            missing_fields.append("max_budget_inr")
            clarification_questions.append("What is your maximum target budget ceiling in INR?")
            assumptions_made.append("No upper budget ceiling provided; prioritizing the most cost-effective feasible match.")

        if water_usage_lpd is None and resource_type in ["Boiler", "Water Treatment"]:
            missing_fields.append("water_usage_lpd")
            clarification_questions.append(f"What is your required daily water throughput / capacity for this {resource_type} (in Liters/day)?")
            assumptions_made.append("Water throughput requirements not explicitly stated; assuming standard system capacity.")

        if not chemical_processes and resource_type in ["Boiler", "Refrigerant", "Water Treatment", "Consultant"]:
            missing_fields.append("chemical_processes")
            clarification_questions.append("What chemical processes or fluids will be involved (e.g., Acids, Solvents, Polymers, Ammonia)?")
            assumptions_made.append("Chemical processes not specified; assuming general industrial processing.")

        has_specs = (
            min_experience_years is not None or
            max_budget_inr is not None or
            water_usage_lpd is not None or
            len(chemical_processes) > 0
        )
        is_unrelated = self._is_out_of_domain(clean_text, resource_type, has_specs)
        is_ambiguous = len(missing_fields) >= 2 or resource_type == "Any" or is_unrelated

        return ParsedRequirement(
            raw_query=clean_text,
            resource_type=resource_type,
            min_experience_years=min_experience_years,
            max_budget_inr=max_budget_inr,
            water_usage_lpd=water_usage_lpd,
            chemical_processes=chemical_processes,
            quantity_required=quantity,
            urgency=urgency,
            is_ambiguous=is_ambiguous,
            is_unrelated=is_unrelated,
            missing_fields=missing_fields,
            assumptions_made=assumptions_made,
            clarification_questions=clarification_questions
        )
