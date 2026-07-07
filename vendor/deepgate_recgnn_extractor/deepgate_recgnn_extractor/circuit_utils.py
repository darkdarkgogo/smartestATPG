from collections import Counter


def logic(gate_type, signals):
    if gate_type == 1:
        for signal in signals:
            if signal == 0:
                return 0
        return 1
    if gate_type == 2:
        for signal in signals:
            if signal == 0:
                return 1
        return 0
    if gate_type == 3:
        for signal in signals:
            if signal == 1:
                return 1
        return 0
    if gate_type == 4:
        for signal in signals:
            if signal == 1:
                return 0
        return 1
    if gate_type == 5:
        for signal in signals:
            return 0 if signal == 1 else 1
    if gate_type == 6:
        zero_count = 0
        one_count = 0
        for signal in signals:
            if signal == 0:
                zero_count += 1
            elif signal == 1:
                one_count += 1
        if zero_count == len(signals) or one_count == len(signals):
            return 0
        return 1
    if gate_type == 7:
        for signal in signals:
            return signal
    if gate_type == 8:
        zero_count = 0
        one_count = 0
        for signal in signals:
            if signal == 0:
                zero_count += 1
            elif signal == 1:
                one_count += 1
        if zero_count == len(signals) or one_count == len(signals):
            return 1
        return 0
    raise ValueError(f"Unsupported gate type id: {gate_type}")


def prob_logic(gate_type, signals):
    one = 0.0
    zero = 0.0

    if gate_type == 1:
        mul = 1.0
        for signal in signals:
            mul *= signal[1]
        one = mul
        zero = 1.0 - mul
    elif gate_type == 2:
        mul = 1.0
        for signal in signals:
            mul *= signal[1]
        zero = mul
        one = 1.0 - mul
    elif gate_type == 3:
        mul = 1.0
        for signal in signals:
            mul *= signal[0]
        zero = mul
        one = 1.0 - mul
    elif gate_type == 4:
        mul = 1.0
        for signal in signals:
            mul *= signal[0]
        one = mul
        zero = 1.0 - mul
    elif gate_type == 5:
        for signal in signals:
            one = signal[0]
            zero = signal[1]
    elif gate_type == 6:
        mul0 = 1.0
        mul1 = 1.0
        for signal in signals:
            mul0 *= signal[0]
        for signal in signals:
            mul1 *= signal[1]
        zero = mul0 + mul1
        one = 1.0 - zero
    elif gate_type == 7:
        for signal in signals:
            zero = signal[0]
            one = signal[1]
    elif gate_type == 8:
        mul0 = 1.0
        mul1 = 1.0
        for signal in signals:
            mul0 *= signal[0]
        for signal in signals:
            mul1 *= signal[1]
        one = mul0 + mul1
        zero = 1.0 - one
    else:
        raise ValueError(f"Unsupported gate type id: {gate_type}")

    return zero, one


def obs_prob(x_data, root_idx, observability, input_signals):
    gate_type = x_data[root_idx][1]

    if gate_type == 1 or gate_type == 2:
        obs = observability[root_idx]
        for signal in input_signals:
            for other in input_signals:
                if signal != other:
                    obs *= x_data[other][3]
            if obs < observability[signal] or observability[signal] == -1:
                observability[signal] = obs
    elif gate_type == 3 or gate_type == 4:
        obs = observability[root_idx]
        for signal in input_signals:
            for other in input_signals:
                if signal != other:
                    obs *= x_data[other][4]
            if obs < observability[signal] or observability[signal] == -1:
                observability[signal] = obs
    elif gate_type == 5 or gate_type == 7:
        obs = observability[root_idx]
        for signal in input_signals:
            if obs < observability[signal] or observability[signal] == -1:
                observability[signal] = obs
    elif gate_type == 6 or gate_type == 8:
        if len(input_signals) != 2:
            raise ValueError("Only 2-input XOR/XNOR is supported for observability computation")
        obs = observability[root_idx]
        signal = input_signals[1]
        obs *= max(x_data[signal][3], x_data[signal][4])
        observability[input_signals[0]] = obs

        obs = observability[root_idx]
        signal = input_signals[0]
        obs *= max(x_data[signal][3], x_data[signal][4])
        observability[input_signals[1]] = obs

    return observability


def generate_prob_cont(x_data, pi_indexes, level_list, fanin_list):
    probabilities = [0] * len(x_data)
    for index in pi_indexes:
        probabilities[index] = [0.5, 0.5]

    for level in range(1, len(level_list)):
        for index in level_list[level]:
            source_signals = [probabilities[node] for node in fanin_list[index]]
            if source_signals:
                zero, one = prob_logic(x_data[index][1], source_signals)
                probabilities[index] = [zero, one]

    for index, probability in enumerate(probabilities):
        x_data[index].append(probability[1])
        x_data[index].append(probability[0])

    return x_data


def generate_prob_obs(x_data, level_list, fanin_list, fanout_list):
    observability = [-1] * len(x_data)
    for index, fanouts in enumerate(fanout_list):
        if len(fanouts) == 0:
            observability[index] = 1

    for level in range(len(level_list) - 1, -1, -1):
        for index in level_list[level]:
            source_signals = fanin_list[index]
            if source_signals:
                observability = obs_prob(x_data, index, observability, source_signals)

    for index, value in enumerate(observability):
        x_data[index].append(value)

    return x_data


def identify_reconvergence(x_data, level_list, fanin_list, fanout_list):
    for index, _node in enumerate(x_data):
        x_data[index].append(1 if len(fanout_list[index]) > 1 else 0)

    fol = [[] for _ in x_data]
    fanout_num = [len(fanouts) for fanouts in fanout_list]
    is_deleted = [False] * len(x_data)
    rc_list = []
    max_level = max(node[2] for node in x_data) if x_data else 0

    for level in range(max_level + 1):
        if level == 0:
            for index in level_list[0]:
                x_data[index].append(0)
                x_data[index].append(-1)
                if x_data[index][6]:
                    fol[index].append(index)
            continue

        for index in level_list[level]:
            fol_tmp = []
            fol_unique = []
            releasable = []
            for predecessor in fanin_list[index]:
                if is_deleted[predecessor]:
                    raise ValueError("Reconvergence cache was released too early")
                fol_tmp += fol[predecessor]
                fanout_num[predecessor] -= 1
                if fanout_num[predecessor] == 0:
                    releasable.append(predecessor)

            for releasable_index in releasable:
                fol[releasable_index].clear()
                is_deleted[releasable_index] = True

            counts = Counter(fol_tmp)
            source_node_idx = 0
            source_node_level = -1
            is_rc = False
            for candidate in counts:
                fol_unique.append(candidate)
                if counts[candidate] > 1:
                    is_rc = True
                    if x_data[candidate][2] > source_node_level:
                        source_node_level = x_data[candidate][2]
                        source_node_idx = candidate

            if is_rc:
                x_data[index].append(1)
                x_data[index].append(source_node_idx)
                rc_list.append(index)
            else:
                x_data[index].append(0)
                x_data[index].append(-1)

            fol[index] = fol_unique
            if x_data[index][6]:
                fol[index].append(index)

    return x_data, rc_list
