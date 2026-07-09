from __future__ import annotations

import sys
from pathlib import Path

import torch


def _podemquest_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_deepgate_importable() -> None:
    candidate_roots = [
        _podemquest_root() / "vendor" / "deepgate_recgnn_extractor",
        _podemquest_root().parent / "deepgate_recgnn_extractor",
    ]
    for deepgate_root in candidate_roots:
        deepgate_root_str = str(deepgate_root)
        if deepgate_root.exists() and deepgate_root_str not in sys.path:
            sys.path.insert(0, deepgate_root_str)
            return

    raise ModuleNotFoundError(
        "Unable to locate deepgate_recgnn_extractor. Expected it under "
        "'PodemQuest/vendor/deepgate_recgnn_extractor' or as a sibling project."
    )


def load_aligned_gate_embeddings(circuit, checkpoint_path: str) -> int:
    _ensure_deepgate_importable()

    from deepgate_recgnn_extractor import encode_bench

    bench_path = circuit.bench_path
    if bench_path is None:
        raise ValueError("Circuit is missing the source .bench path required for DeepGate encoding.")

    result = encode_bench(bench_path, checkpoint_path=checkpoint_path)
    node_embeddings = result["node_embeddings"]
    gate_meta = result["gate_meta"]
    embedding_by_name = {
        meta["name"]: node_embeddings[meta["index"]].detach().clone().float()
        for meta in gate_meta
    }

    if not embedding_by_name:
        raise ValueError("DeepGate extractor returned no node embeddings.")

    cache = {}
    visiting = set()

    def resolve_gate_embedding(gate):
        if gate.id in cache:
            return cache[gate.id]
        if gate.id in visiting:
            raise ValueError(f"Cycle detected while resolving embedding for gate '{gate.outputpin}'.")

        visiting.add(gate.id)
        gate_name = gate.outputpin
        if gate_name in embedding_by_name:
            embedding = embedding_by_name[gate_name]
        elif gate.type == "output_pin":
            if len(gate.input_gates) != 1:
                raise ValueError(
                    f"Expected output pin '{gate.outputpin}' to have exactly one driver, "
                    f"found {len(gate.input_gates)}."
                )
            embedding = resolve_gate_embedding(gate.input_gates[0])
        else:
            available_examples = ", ".join(sorted(embedding_by_name.keys())[:10])
            raise KeyError(
                f"DeepGate embedding not found for gate '{gate_name}' (type={gate.type}). "
                f"Sample available node names: {available_examples}"
            )

        visiting.remove(gate.id)
        cache[gate.id] = embedding
        return embedding

    for gate in circuit.gates.values():
        gate.deepgate_embedding = resolve_gate_embedding(gate)

    return int(node_embeddings.shape[1])
