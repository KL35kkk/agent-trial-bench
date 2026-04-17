"""
Prepare DAB-compatible datasets from DMind Benchmark data.

Outputs:
- data/dmind/dmind_objective.csv   (combined multiple-choice questions)
- data/dmind/dmind_subjective.jsonl (combined rubric questions with domain)
"""

import csv
import json
import os
from glob import glob

DMIND_OBJECTIVE_DIR = "data/dmind/objective"
DMIND_SUBJECTIVE_DIR = "data/dmind/subjective"
OUTPUT_OBJECTIVE = "data/dmind/dmind_objective.csv"
OUTPUT_SUBJECTIVE = "data/dmind/dmind_subjective.jsonl"


def load_objective():
    rows = []
    id_counter = 1
    for path in sorted(glob(os.path.join(DMIND_OBJECTIVE_DIR, "*.csv"))):
        domain = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_text = row.get("Question", "").strip()
                options = [
                    ("A", row.get("Option A", "").strip()),
                    ("B", row.get("Option B", "").strip()),
                    ("C", row.get("Option C", "").strip()),
                    ("D", row.get("Option D", "").strip()),
                ]
                option_str = " ".join([f"{key}) {val}" for key, val in options if val])
                question = f"{question_text} Options: {option_str}"
                correct_letter = (row.get("Correct option", "") or "").strip()
                answer_map = {key: val for key, val in options}
                answer = answer_map.get(correct_letter, correct_letter)
                rows.append({
                    "id": id_counter,
                    "question": question,
                    "answer": answer,
                    "category": f"dmind_{domain.lower()}",
                    "task_type": "multiple_choice",
                    "evaluation_method": "rule_based",
                })
                id_counter += 1
    return rows


def save_objective(rows):
    os.makedirs(os.path.dirname(OUTPUT_OBJECTIVE), exist_ok=True)
    with open(OUTPUT_OBJECTIVE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "category", "task_type", "evaluation_method"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_subjective():
    merged = []
    for path in sorted(glob(os.path.join(DMIND_SUBJECTIVE_DIR, "*.jsonl"))):
        domain = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data["domain"] = f"dmind_{domain.lower()}"
                merged.append(data)
    return merged


def save_subjective(items):
    os.makedirs(os.path.dirname(OUTPUT_SUBJECTIVE), exist_ok=True)
    with open(OUTPUT_SUBJECTIVE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    objective_rows = load_objective()
    save_objective(objective_rows)
    subjective_items = load_subjective()
    save_subjective(subjective_items)
    print(f"Objective rows: {len(objective_rows)} -> {OUTPUT_OBJECTIVE}")
    print(f"Subjective items: {len(subjective_items)} -> {OUTPUT_SUBJECTIVE}")


if __name__ == "__main__":
    main()
