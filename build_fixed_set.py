#!/usr/bin/env python3
"""
build_fixed_set.py — Build a deterministic, fixed evaluation scenario set.

For each category in SEEDS, deterministically pick 5 (persona_id, situation)
pairs using random.Random(1234).  Prefer English personas, no repeated
persona within a category, no repeated seed within a category.  Pick
num_turns once per scenario (between that category's turns_min and turns_max).

Writes:
    eval_sets/fixed_eval_v1.json
    eval_sets/fixed_eval_v1_seeds.txt

With --novel-seeds PATH (JSON: {category: [situations]}), builds a v2 set from
fresh openings that are NOT in SEEDS, so the eval set does not overlap with
training data.

Refuses to overwrite an existing file unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from prompts.seeds import SEEDS
from prompts.personas import PERSONAS
from prompts.meta import CATEGORIES, CATEGORY_PERSONAS

BASE_DIR = Path(__file__).parent.resolve()
EVAL_SETS_DIR = BASE_DIR / "eval_sets"

FIXED_SET_PATH = EVAL_SETS_DIR / "fixed_eval_v1.json"
SEEDS_TXT_PATH = EVAL_SETS_DIR / "fixed_eval_v1_seeds.txt"

RNG_SEED = 1234
SCENARIOS_PER_CATEGORY = 5


def _english_persona_ids_for(category: str) -> list[str]:
    """Persona IDs eligible for a category, English-language only."""
    raw = CATEGORY_PERSONAS.get(category) or [
        p["id"] for p in PERSONAS if p.get("language") == "english"
    ]
    ids = []
    for pid in raw:
        persona = next((p for p in PERSONAS if p["id"] == pid), None)
        if persona and persona.get("language") == "english":
            ids.append(pid)
    return ids


def _pick_scenarios(
    category: str,
    rng: random.Random,
    novel_seeds: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Pick 5 deterministic (persona_id, situation, num_turns) scenarios."""
    persona_ids = _english_persona_ids_for(category)

    if novel_seeds and category in novel_seeds:
        seed_pool = novel_seeds[category]
    else:
        seed_pool = SEEDS.get(category, [])

    if not seed_pool:
        raise RuntimeError(f"No seeds available for category {category!r}")
    if len(persona_ids) < SCENARIOS_PER_CATEGORY:
        raise RuntimeError(
            f"Category {category!r} has only {len(persona_ids)} English personas; "
            f"need {SCENARIOS_PER_CATEGORY}"
        )
    if len(seed_pool) < SCENARIOS_PER_CATEGORY:
        raise RuntimeError(
            f"Category {category!r} has only {len(seed_pool)} seeds; "
            f"need {SCENARIOS_PER_CATEGORY}"
        )

    # Deterministic shuffle of persona and seed pools.
    shuffled_personas = persona_ids[:]
    shuffled_seeds = seed_pool[:]
    rng.shuffle(shuffled_personas)
    rng.shuffle(shuffled_seeds)

    # Greedy selection: no repeated persona, no repeated seed.
    chosen: list[tuple[str, str]] = []
    used_personas: set[str] = set()
    used_seeds: set[str] = set()

    for persona_id, situation in zip(shuffled_personas, shuffled_seeds):
        if persona_id in used_personas:
            continue
        if situation in used_seeds:
            continue
        chosen.append((persona_id, situation))
        used_personas.add(persona_id)
        used_seeds.add(situation)
        if len(chosen) == SCENARIOS_PER_CATEGORY:
            break

    if len(chosen) < SCENARIOS_PER_CATEGORY:
        # Fallback: allow repeated seeds if personas are exhausted.
        for persona_id, situation in zip(shuffled_personas, shuffled_seeds):
            if persona_id in used_personas:
                continue
            chosen.append((persona_id, situation))
            used_personas.add(persona_id)
            if len(chosen) == SCENARIOS_PER_CATEGORY:
                break

    if len(chosen) < SCENARIOS_PER_CATEGORY:
        raise RuntimeError(
            f"Could not pick {SCENARIOS_PER_CATEGORY} scenarios for {category!r}"
        )

    turns_min = CATEGORIES.get(category, {}).get("turns_min", 5)
    turns_max = CATEGORIES.get(category, {}).get("turns_max", 9)

    scenarios: list[dict] = []
    for idx, (persona_id, situation) in enumerate(chosen):
        num_turns = rng.randint(turns_min, turns_max)
        scenarios.append(
            {
                "scenario_id": f"{category}__{persona_id}__{idx:02d}",
                "category": category,
                "persona_id": persona_id,
                "situation": situation,
                "num_turns": num_turns,
            }
        )
    return scenarios


def _check_novelty(
    novel_seeds: dict[str, list[str]],
    rng: random.Random,
) -> None:
    """Warn loudly if any novel situation already exists in SEEDS."""
    existing: set[str] = set()
    for situations in SEEDS.values():
        existing.update(situations)

    for category, situations in novel_seeds.items():
        for situation in situations:
            if situation in existing:
                print(
                    f"[WARN] --novel-seeds: situation for {category!r} already "
                    f"exists in SEEDS — this defeats the purpose of a novel set:\n"
                    f"       {situation[:120]}"
                )


def build_fixed_set(
    output_json: Path,
    output_txt: Path,
    novel_seeds: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Build and write the fixed scenario set. Returns the scenario list."""
    rng = random.Random(RNG_SEED)

    if novel_seeds:
        _check_novelty(novel_seeds, rng)

    all_scenarios: list[dict] = []
    for category in SEEDS.keys():
        all_scenarios.extend(_pick_scenarios(category, rng, novel_seeds))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=2)

    with open(output_txt, "w", encoding="utf-8") as f:
        for scenario in all_scenarios:
            f.write(scenario["situation"].strip() + "\n")

    print(f"[OK] Wrote {len(all_scenarios)} scenarios -> {output_json}")
    print(f"[OK] Wrote {len(all_scenarios)} situations -> {output_txt}")
    return all_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic fixed evaluation scenario set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--novel-seeds",
        metavar="PATH",
        default=None,
        help="JSON {category: [situations]} of fresh openings not in SEEDS.",
    )
    parser.add_argument(
        "--output-json",
        default=str(FIXED_SET_PATH),
        help="Output JSON path (default: eval_sets/fixed_eval_v1.json).",
    )
    parser.add_argument(
        "--output-txt",
        default=str(SEEDS_TXT_PATH),
        help="Output txt path (default: eval_sets/fixed_eval_v1_seeds.txt).",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)

    if output_json.exists() and not args.force:
        print(
            f"[ERR] {output_json} already exists. Pass --force to overwrite."
        )
        sys.exit(1)

    novel_seeds: dict[str, list[str]] | None = None
    if args.novel_seeds:
        with open(args.novel_seeds, "r", encoding="utf-8") as f:
            novel_seeds = json.load(f)

    build_fixed_set(output_json, output_txt, novel_seeds)


if __name__ == "__main__":
    main()