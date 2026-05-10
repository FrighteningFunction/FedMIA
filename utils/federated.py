import copy


def fed_avg_state_dicts(local_state_dicts, client_weights):
    if not local_state_dicts:
        raise ValueError("local_state_dicts must not be empty")
    if len(local_state_dicts) != len(client_weights):
        raise ValueError("Number of state dicts and weights must match")

    weight_sum = float(sum(client_weights))
    if weight_sum <= 0:
        raise ValueError("client_weights must sum to a positive value")
    normalized_weights = [float(weight) / weight_sum for weight in client_weights]

    averaged_state = copy.deepcopy(local_state_dicts[0])
    for key in averaged_state.keys():
        # Only floating tensors are learnable/averagable model state. Structural
        # buffers such as sparse graph indices and sparse matrix sizes are
        # identical across clients and must not be multiplied by fractional FedAvg
        # weights: integer buffers can otherwise be rounded/truncated on load.
        if hasattr(averaged_state[key], "is_floating_point") and not averaged_state[key].is_floating_point():
            averaged_state[key] = copy.deepcopy(local_state_dicts[0][key])
            continue
        averaged_state[key] = averaged_state[key] * normalized_weights[0]
        for idx in range(1, len(local_state_dicts)):
            averaged_state[key] += local_state_dicts[idx][key] * normalized_weights[idx]
    return averaged_state
