def exponential_smoothing(data, weight = .1):
    result = [data[0]]
    for val in data[1:]:
      result.append(weight * val + (1 - weight) * result[-1])
    return result


