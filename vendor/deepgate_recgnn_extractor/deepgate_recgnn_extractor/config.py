from dataclasses import dataclass, field


DEFAULT_GATE_TO_INDEX = {
    "INPUT": 0,
    "AND": 1,
    "NAND": 2,
    "OR": 3,
    "NOR": 4,
    "NOT": 5,
    "XOR": 6,
    "BUF": 7,
    "XNOR": 8,
}


@dataclass
class ModelConfig:
    aggr_function: str = "aggnconv"
    update_function: str = "gru"
    dim_hidden: int = 64
    dim_mlp: int = 32
    dim_pred: int = 1
    num_fc: int = 3
    wx_update: bool = True
    wx_mlp: bool = False
    intermediate_supervision: bool = False
    reverse: bool = True
    custom_backward: bool = False
    use_edge_attr: bool = False
    mask: bool = False
    num_rounds: int = 10
    num_aggr: int = 3
    dim_edge_feature: int = 16
    norm_layer: str = "batchnorm"
    activation_layer: str = "relu"


@dataclass
class EncoderConfig:
    gate_to_index: dict = field(default_factory=lambda: DEFAULT_GATE_TO_INDEX.copy())
    use_node_cop: bool = False
    use_node_reconv: bool = False
    include_pi_po_features: bool = False
    graph_pool: str = "mean"
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def num_gate_types(self) -> int:
        return len(self.gate_to_index)

    @property
    def dim_node_feature(self) -> int:
        dim = self.num_gate_types
        if self.use_node_cop:
            dim += 1
        if self.use_node_reconv:
            dim += 1
        if self.include_pi_po_features:
            dim += 2
        return dim
