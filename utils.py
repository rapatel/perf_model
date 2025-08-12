"""
Set of utility functions.
"""

import torch

from roofline.roofline_model import RooflineModel


def generate_training_data(
    num_samples: int = 0,
    peak_gflops: float = 100.0,
    peak_memory_bw_gbps: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates training data using RooflineModel.

    :param num_samples: Number of samples to generate.
    :return: Tuple containing input data and target values and
    the max values used to normalize the tensors.
    """

    # The training data will be the following:
    # - Input: [total_ops, total_bytes]
    # - Target: [cost_ns]
    roofline_model = RooflineModel(
        peak_gflops=peak_gflops, peak_memory_bw_gbps=peak_memory_bw_gbps
    )

    # Generate flops and bytes using matrix multiplication
    largest_dimension = 8192
    step = 128
    input_data = []
    targets = []
    for m in range(step, largest_dimension + 1, step):
        for n in range(step, largest_dimension + 1, step):
            for k in range(step, largest_dimension + 1, step):
                total_ops = m * n * k * 2  # Assuming 2 flops per multiply-add operation
                total_bytes = (m * n + n * k + m * k) * 4  # Assuming 4 bytes per float
                cost_ns = roofline_model.get_cost_ns(total_ops, total_bytes)
                input_data.append(
                    [total_ops, total_bytes, peak_gflops, peak_memory_bw_gbps]
                )
                targets.append([cost_ns])
                if num_samples > 0 and len(input_data) >= num_samples:
                    break

    # Convert arrays to tensors
    input_data = torch.tensor(input_data, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Normalize input_data and targets
    input_max = input_data.max(dim=0, keepdim=True).values
    input_data_norm = input_data / input_max

    target_max = targets.max()
    targets_norm = targets / target_max

    return input_data_norm, targets_norm, input_max, target_max
