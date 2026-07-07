# DeepGate RecGNN Extractor

Standalone `.bench` encoder extracted from `DeepGate-main`.

## What It Does

- parses `.bench` netlists
- constructs graph structure and topological order
- reproduces core DeepGate-style gate feature generation
- runs `recgnn` to produce node embeddings
- pools node embeddings into a graph embedding

## Supported Gates

- `INPUT`
- `AND`
- `NAND`
- `OR`
- `NOR`
- `NOT`
- `XOR`
- `BUF` / `BUFF`
- `XNOR`

## Quick Start

```python
from deepgate_recgnn_extractor import encode_bench

result = encode_bench("path/to/circuit.bench")
print(result["node_embeddings"].shape)
print(result["graph_embedding"].shape)
```

## CLI

```bash
python -m deepgate_recgnn_extractor.cli --bench path/to/circuit.bench
```

## Notes

- If no checkpoint is provided, the model runs with randomly initialized weights.
- If you train new checkpoints with the expanded gate vocabulary, the model input now uses distinct one-hot slots for `BUF` and `XNOR`.
- The package returns both raw gate attributes and the model input tensor so downstream code can choose what to consume.
- A practical environment for this package is Python `3.10` or `3.11` with `torch` and `torch-geometric` installed.
