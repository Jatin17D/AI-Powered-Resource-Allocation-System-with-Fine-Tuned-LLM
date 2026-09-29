"""
================================================================================
  HUGGING FACE FINE-TUNING PIPELINE & DATASET GENERATOR FOR RESOURCE ALLOCATION
================================================================================
This script implements the Hugging Face pre-trained LLM fine-tuning workflow:
1. Generates 150+ synthetic prompt-completion pairs in JSONL format.
2. Tokenizes and prepares PyTorch datasets for instruction tuning.
3. Configures LoRA (Low-Rank Adaptation) using PEFT.
4. Executes a robust, native PyTorch training loop compatible with Windows & Python 3.12.

Selected Implementation Approach: Hugging Face Transformers & PEFT/LoRA.
"""
import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any


RESOURCE_TEMPLATES = [
    {
        "type": "Boiler",
        "phrasings": [
            "I need resources for a {chem} project that uses {water} liters of water per day. The project requires a boiler with at least {exp} years of experience and the budget is Rs. {budget}.",
            "Looking for an industrial boiler for {chem} applications. Water requirement is {water} L/day, minimum {exp} years experience, cost under Rs. {budget}.",
            "Require a heavy-duty steam boiler handling {chem}. Budget is {budget} INR, min experience {exp} years.",
            "Need a boiler for my {chem} plant with {exp} years experience and budget of {budget}.",
            "We urgently require {qty} boilers capable of {water} lpd water throughput for {chem} synthesis under Rs. {budget}."
        ],
        "chemicals": ["chemical processing", "organic solvents", "acid distillation", "polymer production", "petrochemical refining"],
        "water_range": [3000, 5000, 8000, 12000, 15000],
        "exp_range": [3, 5, 8, 10, 12],
        "budget_range": [40000, 50000, 60000, 75000, 90000]
    },
    {
        "type": "Refrigerant",
        "phrasings": [
            "Urgent requirement: Need industrial refrigerant chiller for {chem} cooling, budget Rs. {budget}, immediately available.",
            "Looking for a closed-loop refrigeration unit for {chem} storage. Minimum {exp} years operational life, max budget Rs. {budget}.",
            "Need eco-friendly chiller unit for {chem} with at least {exp} years experience, budget {budget} INR.",
            "Require refrigeration equipment for {chem}, low cost under Rs. {budget}, available immediately."
        ],
        "chemicals": ["ammonia cooling", "glycol refrigeration", "cryogenic storage", "solvent condensation", "food-grade chemical chilling"],
        "water_range": [2500, 4000, 6000, 8000],
        "exp_range": [2, 4, 6, 8],
        "budget_range": [25000, 32000, 45000, 65000]
    },
    {
        "type": "Water Treatment",
        "phrasings": [
            "Need water treatment RO unit handling {water} L/day for {chem} effluent processing, budget Rs. {budget}.",
            "Looking for industrial filtration & demineralization unit with capacity {water} lpd, cost under {budget} INR.",
            "Require demineralized water plant for {chem}, water usage {water} liters/day, budget {budget}."
        ],
        "chemicals": ["acid effluent", "brine treatment", "industrial wastewater", "boiler feed water"],
        "water_range": [5000, 10000, 15000, 20000],
        "exp_range": [3, 5, 7],
        "budget_range": [30000, 40000, 55000]
    },
    {
        "type": "Consultant",
        "phrasings": [
            "We need an experienced chemical process consultant with {exp}+ years experience in {chem}, budget up to Rs. {budget}.",
            "Looking for a lead industrial safety and process consultant for {chem}, at least {exp} years experience, budget {budget}.",
            "Require chartered chemical engineering consultant for {chem} audit with minimum {exp} years experience."
        ],
        "chemicals": ["petrochemical synthesis", "polymer engineering", "acid plant safety", "thermal efficiency"],
        "water_range": [None],
        "exp_range": [7, 10, 12, 15],
        "budget_range": [60000, 75000, 90000, 120000]
    },
    {
        "type": "Engineer",
        "phrasings": [
            "Need certified thermal systems boiler engineer with {exp} years experience for {chem} maintenance, budget Rs. {budget}.",
            "Looking for refrigeration specialist engineer with {exp} years experience for {chem} systems, cost under {budget} INR."
        ],
        "chemicals": ["steam systems", "ammonia chilling", "combustion tuning", "plant operations"],
        "water_range": [None],
        "exp_range": [4, 6, 8],
        "budget_range": [40000, 50000, 60000]
    }
]

AMBIGUOUS_TEMPLATES = [
    ("I need a boiler for my chemical project.", "Boiler", ["min_experience_years", "max_budget_inr", "water_usage_lpd"]),
    ("Need refrigeration equipment, low cost, available immediately.", "Refrigerant", ["min_experience_years", "max_budget_inr"]),
    ("Looking for industrial equipment for water recycling.", "Water Treatment", ["max_budget_inr", "water_usage_lpd"]),
    ("Require an expert consultant for plant troubleshooting.", "Consultant", ["min_experience_years", "max_budget_inr"]),
    ("I need some resources for a new project next month.", "Any", ["resource_type", "min_experience_years", "max_budget_inr"])
]


def generate_synthetic_dataset(num_samples: int = 150) -> List[Dict[str, Any]]:
    """
    Generates a rich, diversified dataset of prompt-completion pairs.
    """
    dataset = []

    # 1. Add Ambiguous / Incomplete cases
    for prompt_text, r_type, missing in AMBIGUOUS_TEMPLATES:
        dataset.append({
            "prompt": prompt_text,
            "completion": {
                "resource_type": r_type,
                "min_experience_years": None,
                "max_budget_inr": None,
                "water_usage_lpd": None,
                "chemical_processes": ["General Processing"] if r_type != "Any" else [],
                "quantity_required": 1,
                "urgency": "Immediate" if "immediately" in prompt_text else "Normal",
                "is_ambiguous": True,
                "missing_fields": missing,
                "assumptions_made": [f"Defaulting assumptions for {field}" for field in missing]
            }
        })

    # 2. Add Parametric Synthesized cases
    for _ in range(num_samples):
        group = random.choice(RESOURCE_TEMPLATES)
        template = random.choice(group["phrasings"])
        chem = random.choice(group["chemicals"])
        water = random.choice(group["water_range"])
        exp = random.choice(group["exp_range"])
        budget = random.choice(group["budget_range"])
        qty = random.choice([1, 1, 1, 2])

        prompt = template.format(
            chem=chem,
            water=f"{water:,}" if water else "5,000",
            exp=exp,
            budget=f"{budget:,}",
            qty=qty
        )

        urgency = "Immediate" if "urgent" in prompt.lower() or "immediately" in prompt.lower() else "Normal"

        dataset.append({
            "prompt": prompt,
            "completion": {
                "resource_type": group["type"],
                "min_experience_years": exp if "{exp}" in template else None,
                "max_budget_inr": float(budget) if "{budget}" in template else None,
                "water_usage_lpd": float(water) if water and "{water}" in template else None,
                "chemical_processes": [chem.title()],
                "quantity_required": qty if "{qty}" in template else 1,
                "urgency": urgency,
                "is_ambiguous": False,
                "missing_fields": [],
                "assumptions_made": []
            }
        })

    return dataset


def save_dataset_to_jsonl(filepath: str, num_samples: int = 150):
    dataset = generate_synthetic_dataset(num_samples)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"[HuggingFace Pipeline] Successfully saved {len(dataset)} training samples to: {filepath}")


def train_huggingface_lora(
    dataset_file: str = "training_data.jsonl",
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "./fine_tuned_allocator",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4
):
    """
    Executes Hugging Face PEFT LoRA fine-tuning using a stable native PyTorch loop.
    Avoids multi-threading/lock bugs on Windows and Python 3.12.
    """
    print("\n" + "=" * 70)
    print("       HUGGING FACE PEFT / LoRA FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"Target Base Model : {base_model_name}")
    print(f"Training Dataset  : {dataset_file}")
    print(f"Output Checkpoint : {output_dir}")

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
        from peft import LoraConfig, get_peft_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Computation Device: {device.type.upper()}")

        # 1. Read JSONL directly
        print("\n1. Loading and parsing JSONL dataset...")
        raw_samples = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_samples.append(json.loads(line))
        print(f"   Loaded {len(raw_samples)} training pairs.")

        # 2. Load Tokenizer & Base Model (Loads from local cache if already downloaded)
        print(f"\n2. Loading tokenizer and model '{base_model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        if device.type != "cuda":
            model.to(device)
        model.config.use_cache = False

        # 3. Configure LoRA
        print("\n3. Attaching LoRA adapter (PEFT)...")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 4. Create PyTorch Dataset
        class InstructionDataset(Dataset):
            def __init__(self, samples, tok, max_len=256):
                self.inputs = []
                for s in samples:
                    text = (
                        f"<|system|>\nYou are an expert industrial resource allocation assistant. "
                        f"Extract technical specifications into JSON.\n"
                        f"<|user|>\n{s['prompt']}\n<|assistant|>\n{json.dumps(s['completion'])}"
                    )
                    encoding = tok(
                        text,
                        max_length=max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    input_ids = encoding["input_ids"].squeeze(0)
                    attention_mask = encoding["attention_mask"].squeeze(0)
                    labels = input_ids.clone()
                    labels[labels == tok.pad_token_id] = -100
                    self.inputs.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    })

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return self.inputs[idx]

        train_dataset = InstructionDataset(raw_samples, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # 5. Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

        # 6. Training Loop
        print(f"\n4. Starting LoRA training loop ({num_epochs} epochs, {total_steps} total steps)...")
        model.train()
        step = 0
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                step += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                if step % 20 == 0 or step == total_steps:
                    print(f"   [Epoch {epoch+1}/{num_epochs}] Step {step:>3}/{total_steps} | Loss: {loss.item():.4f}")

            avg_epoch_loss = total_loss / len(train_loader)
            print(f"   --> Epoch {epoch+1} Complete. Average Loss: {avg_epoch_loss:.4f}")

        # 7. Save Artifacts
        print(f"\n5. Saving fine-tuned LoRA model to '{output_dir}'...")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print("\n" + "=" * 70)
        print("  [SUCCESS] LoRA Fine-Tuning Completed Successfully!")
        print(f"  Fine-tuned adapter saved in: {os.path.abspath(output_dir)}")
        print("=" * 70)
        print("\nYou can now run the resource allocation system with your model:")
        print(f"  python resource_allocation.py --use-hf --hf-model {output_dir}")

    except ImportError as e:
        print(f"\n[Notice] Required libraries missing: {e}")
        print("Install with: pip install torch transformers peft")
    except Exception as e:
        print(f"\n[Error during training]: {e}")
        import traceback
        traceback.print_exc()


def print_workflow_summary():
    print("""
================================================================================
             HUGGING FACE FINE-TUNING ARCHITECTURE & WORKFLOW
================================================================================
Selected Approach: Hugging Face Pre-trained LLM with LoRA Fine-Tuning

Pipeline Overview:
  1. Input Natural Language: "I need a boiler with at least 5 years experience, budget Rs 60,000"
  2. Hugging Face Base LLM : TinyLlama / Mistral / Flan-T5
  3. LoRA Adapter Checkpoint: ./fine_tuned_allocator
  4. Extracted Output JSON :
     {
       "resource_type": "Boiler",
       "min_experience_years": 5,
       "max_budget_inr": 60000.0,
       "availability": "immediate"
     }
  5. Python Constraint Matcher -> Gap Analysis -> Justification & Explanations

Why Hugging Face?
- Open-source, reproducible, and verifiable.
- Supports PEFT/LoRA (efficient parameter updates on consumer hardware / Colab).
- Seamless Python integration with PyTorch & Transformers ecosystem.
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Fine-Tuning Pipeline for Resource Allocation")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic training_data.jsonl dataset")
    parser.add_argument("--samples", type=int, default=150, help="Number of training samples to generate")
    parser.add_argument("--train", action="store_true", help="Execute Hugging Face LoRA training loop")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face base model ID")
    parser.add_argument("--output", type=str, default="./fine_tuned_allocator", help="Directory to save fine-tuned LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")

    args = parser.parse_args()
    data_path = os.path.join(os.path.dirname(__file__), "training_data.jsonl")

    if args.generate:
        save_dataset_to_jsonl(data_path, num_samples=args.samples)
    elif args.train:
        if not os.path.exists(data_path):
            save_dataset_to_jsonl(data_path, num_samples=args.samples)
        train_huggingface_lora(
            dataset_file=data_path,
            base_model_name=args.model,
            output_dir=args.output,
            num_epochs=args.epochs
        )
    else:
        save_dataset_to_jsonl(data_path, num_samples=args.samples)
        print_workflow_summary()


if __name__ == "__main__":
    main()
