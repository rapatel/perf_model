"""
Roofline Model Module
This module implements the roofline model for performance analysis.
"""


class RooflineModel:
    def __init__(self, peak_gflops: float, peak_memory_bw_gbps: float):
        self.peak_gflops = peak_gflops
        self.peak_memory_bw_gbps = peak_memory_bw_gbps

    def get_compute_time_ns(self, total_ops: int) -> float:
        if total_ops <= 0:
            raise ValueError("Total operations must be greater than zero.")

        return total_ops / self.peak_gflops

    def get_memory_time_ns(self, total_bytes: int) -> float:
        if total_bytes <= 0:
            raise ValueError("Data size must be greater than zero.")

        return total_bytes / self.peak_memory_bw_gbps

    def get_cost_ns(self, total_ops: int, total_bytes: int) -> float:
        return max(
            self.get_compute_time_ns(total_ops=total_ops),
            self.get_memory_time_ns(total_bytes=total_bytes),
        )
