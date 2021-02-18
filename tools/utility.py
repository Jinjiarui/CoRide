def eps_decay(now_step, anchor: list, anchor_value: list):
    """Decay epsilon value"""

    i = 0
    while i < len(anchor) and now_step >= anchor[i]:
        i += 1
    
    if i == len(anchor):
        return anchor_value[-1]
    else:
        step_value = (anchor_value[i] - anchor_value[i - 1]) / (anchor[i] - anchor[i - 1])
        delta = now_step - anchor[i - 1]
        return anchor_value[i - 1] + delta * step_value
