"""
Dataset utilities for ROME x LRH analysis.

Extends the existing dsets/ module (CounterFactDataset, KnownsDataset) to
provide relation-grouped and contrastive-pair datasets suitable for:
    - Linear probe training (binary classification per concept)
    - Contrastive direction extraction (positive/negative pairs)
    - LRE estimation (subject-object pairs within a relation)

CounterFact records use Wikidata property IDs as relation_id:
    P19  = place of birth          P27  = country of citizenship
    P36  = capital                 P37  = official language
    P101 = field of work           P103 = native language
    P106 = occupation              P127 = owned by
    P131 = located in              P136 = genre
    P159 = HQ location             P176 = manufacturer
    P264 = record label            P361 = part of
    P407 = language of work        P413 = position played
    P449 = original network        P463 = member of
    P495 = country of origin       P530 = diplomatic relation
    P740 = location of formation   P1376 = capital of
    P1412 = languages spoken
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from util.globals import DATA_DIR, REMOTE_ROOT_URL


class RelationGroupedDataset:
    """
    Groups CounterFact records by relation_id and provides methods for
    generating contrastive pairs and probe datasets.
    """

    def __init__(
        self,
        data_dir: str = None,
        counterfact_data: Optional[List[Dict]] = None,
        min_samples_per_relation: int = 10,
    ):
        if counterfact_data is not None:
            raw = counterfact_data
        else:
            data_dir = Path(data_dir or DATA_DIR)
            cf_loc = data_dir / "counterfact.json"
            if not cf_loc.exists():
                url = f"{REMOTE_ROOT_URL}/data/dsets/counterfact.json"
                print(f"Downloading CounterFact from {url}")
                data_dir.mkdir(exist_ok=True, parents=True)
                torch.hub.download_url_to_file(url, cf_loc)
            with open(cf_loc, "r") as f:
                raw = json.load(f)

        # Group by relation_id
        self._by_relation: Dict[str, List[Dict]] = defaultdict(list)
        for record in raw:
            rid = record.get("requested_rewrite", {}).get("relation_id")
            if rid is not None:
                self._by_relation[rid].append(record)

        # Filter by minimum samples
        self._by_relation = {
            k: v
            for k, v in self._by_relation.items()
            if len(v) >= min_samples_per_relation
        }

        print(
            f"RelationGroupedDataset: {len(self._by_relation)} relations, "
            f"{sum(len(v) for v in self._by_relation.values())} total records"
        )

    @property
    def relation_ids(self) -> List[str]:
        return sorted(self._by_relation.keys())

    def relation_size(self, relation_id: str) -> int:
        return len(self._by_relation.get(relation_id, []))

    def get_relation_records(self, relation_id: str) -> List[Dict]:
        return self._by_relation.get(relation_id, [])

    def get_subject_object_pairs(
        self,
        relation_id: str,
        n_pairs: Optional[int] = None,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Get (prompt, subject, target_true) triples for LRE estimation.

        Returns list of dicts with keys:
            'prompt': str (with {} placeholder for subject)
            'subject': str
            'target_true': str
        """
        records = self._by_relation.get(relation_id, [])
        rng = random.Random(seed)
        if n_pairs and n_pairs < len(records):
            records = rng.sample(records, n_pairs)

        pairs = []
        for r in records:
            rw = r["requested_rewrite"]
            pairs.append(
                {
                    "prompt": rw["prompt"],
                    "subject": rw["subject"],
                    "target_true": rw["target_true"]["str"],
                }
            )
        return pairs

    def get_contrastive_pairs(
        self,
        relation_id: str,
        target_value: str,
        n_pairs: int = 100,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate positive/negative examples for binary concept probing.

        Positive: records where target_true == target_value.
        Negative: records where target_true != target_value.

        Returns:
            (positive_examples, negative_examples), each a list of dicts
            with keys: 'prompt', 'subject', 'target_true'.
        """
        records = self._by_relation.get(relation_id, [])
        rng = random.Random(seed)

        positives, negatives = [], []
        for r in records:
            rw = r["requested_rewrite"]
            entry = {
                "prompt": rw["prompt"],
                "subject": rw["subject"],
                "target_true": rw["target_true"]["str"],
            }
            if rw["target_true"]["str"].strip() == target_value.strip():
                positives.append(entry)
            else:
                negatives.append(entry)

        # Balance and sample
        min_size = min(len(positives), len(negatives), n_pairs)
        if min_size == 0:
            return [], []
        positives = rng.sample(positives, min(min_size, len(positives)))
        negatives = rng.sample(negatives, min(min_size, len(negatives)))
        return positives, negatives

    def get_paired_contrastive(
        self,
        relation_id: str,
        n_pairs: int = 100,
        seed: int = 42,
    ) -> List[Tuple[Dict, Dict]]:
        """
        Generate paired examples for DAS direction extraction.

        Each pair shares the same relation but has different object values.
        Suitable for computing h_pos - h_neg difference vectors.
        """
        records = self._by_relation.get(relation_id, [])
        rng = random.Random(seed)

        by_target = defaultdict(list)
        for r in records:
            rw = r["requested_rewrite"]
            val = rw["target_true"]["str"].strip()
            by_target[val].append(
                {
                    "prompt": rw["prompt"],
                    "subject": rw["subject"],
                    "target_true": val,
                }
            )

        # Only use target values with enough samples
        viable = {k: v for k, v in by_target.items() if len(v) >= 2}
        if len(viable) < 2:
            return []

        target_values = list(viable.keys())
        pairs = []
        for _ in range(n_pairs):
            v1, v2 = rng.sample(target_values, 2)
            e1 = rng.choice(viable[v1])
            e2 = rng.choice(viable[v2])
            pairs.append((e1, e2))

        return pairs

    def get_unique_targets(self, relation_id: str) -> List[str]:
        """Get all unique target values for a relation."""
        records = self._by_relation.get(relation_id, [])
        targets = set()
        for r in records:
            targets.add(r["requested_rewrite"]["target_true"]["str"].strip())
        return sorted(targets)


class ProbeDataset:
    """
    Wraps relation-grouped data into train/val/test splits for probe training.

    For binary classification: concept_present vs concept_absent.
    For multi-class: each unique target value is a class.
    """

    def __init__(
        self,
        relation_grouped: RelationGroupedDataset,
        relation_id: str,
        target_value: Optional[str] = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.relation_id = relation_id
        self.target_value = target_value
        rng = random.Random(seed)

        records = relation_grouped.get_relation_records(relation_id)
        rng.shuffle(records)

        entries = []
        for r in records:
            rw = r["requested_rewrite"]
            true_val = rw["target_true"]["str"].strip()
            if target_value is not None:
                label = 1 if true_val == target_value.strip() else 0
            else:
                label = true_val
            entries.append(
                {
                    "prompt": rw["prompt"],
                    "subject": rw["subject"],
                    "target_true": true_val,
                    "label": label,
                }
            )

        n = len(entries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self._train = entries[:n_train]
        self._val = entries[n_train : n_train + n_val]
        self._test = entries[n_train + n_val :]

    @property
    def train(self) -> List[Dict]:
        return self._train

    @property
    def val(self) -> List[Dict]:
        return self._val

    @property
    def test(self) -> List[Dict]:
        return self._test

    @property
    def all_labels(self) -> List:
        return sorted(set(e["label"] for e in self._train))


def load_lrh_dataset(
    data_dir: str = None,
    size: Optional[int] = None,
    min_samples_per_relation: int = 10,
) -> RelationGroupedDataset:
    """
    Convenience function: load CounterFact and wrap as RelationGroupedDataset.
    """
    from dsets import CounterFactDataset

    cf = CounterFactDataset(data_dir or str(DATA_DIR), size=size)
    return RelationGroupedDataset(
        counterfact_data=cf.data,
        min_samples_per_relation=min_samples_per_relation,
    )