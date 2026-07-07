import argparse

import torch

from .api import encode_bench
from .config import EncoderConfig


def main():
    parser = argparse.ArgumentParser(description="Encode a .bench circuit with DeepGate RecGNN")
    parser.add_argument("--bench", required=True, help="Path to the .bench file")
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint")
    parser.add_argument("--use-node-cop", action="store_true", help="Append C1 to model input features")
    parser.add_argument("--use-node-reconv", action="store_true", help="Append reconvergence flag to model input features")
    parser.add_argument("--include-pi-po-features", action="store_true", help="Append PI/PO flags to model input features")
    args = parser.parse_args()

    config = EncoderConfig(
        use_node_cop=args.use_node_cop,
        use_node_reconv=args.use_node_reconv,
        include_pi_po_features=args.include_pi_po_features,
    )
    result = encode_bench(
        bench_path=args.bench,
        checkpoint_path=args.checkpoint,
        config=config,
        device=torch.device("cpu"),
    )

    print("node_embeddings:", tuple(result["node_embeddings"].shape))
    print("graph_embedding:", tuple(result["graph_embedding"].shape))
    print("model_node_features:", tuple(result["model_node_features"].shape))
    print("raw_node_features:", tuple(result["raw_node_features"].shape))
    print("edge_index:", tuple(result["edge_index"].shape))
    print("num_gates:", len(result["gate_meta"]))


if __name__ == "__main__":
    main()
