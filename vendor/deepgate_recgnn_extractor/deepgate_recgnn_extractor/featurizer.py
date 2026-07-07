import torch


def one_hot(indices, length):
    if isinstance(indices, int):
        indices = torch.LongTensor([indices]).unsqueeze(0)
    else:
        indices = torch.LongTensor(indices).unsqueeze(0).t()
    return torch.zeros((len(indices), length)).scatter_(1, indices, 1)


def build_model_features(circuit, config):
    gate_list = [node[1] for node in circuit.x_data]
    features = one_hot(gate_list, config.num_gate_types)

    if config.use_node_cop:
        c1 = torch.tensor([node[3] for node in circuit.x_data], dtype=torch.float32).unsqueeze(1)
        features = torch.cat([features, c1], dim=1)

    if config.use_node_reconv:
        reconv = torch.tensor([node[7] for node in circuit.x_data], dtype=torch.float32).unsqueeze(1)
        features = torch.cat([features, reconv], dim=1)

    if config.include_pi_po_features:
        pi_flags = torch.tensor([1.0 if meta["is_pi"] else 0.0 for meta in circuit.gate_meta], dtype=torch.float32).unsqueeze(1)
        po_flags = torch.tensor([1.0 if meta["is_po"] else 0.0 for meta in circuit.gate_meta], dtype=torch.float32).unsqueeze(1)
        features = torch.cat([features, pi_flags, po_flags], dim=1)

    return features


def build_raw_feature_tensor(circuit):
    rows = []
    for node in circuit.x_data:
        rows.append([
            float(node[1]),
            float(node[2]),
            float(node[3]),
            float(node[4]),
            float(node[5]),
            float(node[6]),
            float(node[7]),
            float(node[8]),
        ])
    return torch.tensor(rows, dtype=torch.float32)
