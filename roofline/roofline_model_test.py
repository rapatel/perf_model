"""
Unit tests for the RooflineModel class.
These tests cover the methods for computing time and cost based on operations and memory bandwidth.
"""

import pytest
from roofline_model import RooflineModel


def test_compute_time_ns_valid():
    model = RooflineModel(peak_gflops=100.0, peak_memory_bw_gbps=50.0)
    assert model.get_compute_time_ns(200) == 2.0


def test_compute_time_ns_invalid():
    model = RooflineModel(peak_gflops=100.0, peak_memory_bw_gbps=50.0)
    with pytest.raises(ValueError):
        model.get_compute_time_ns(0)
    with pytest.raises(ValueError):
        model.get_compute_time_ns(-10)


def test_memory_time_ns_valid():
    model = RooflineModel(peak_gflops=100.0, peak_memory_bw_gbps=50.0)
    assert model.get_memory_time_ns(100) == 2.0


def test_memory_time_ns_invalid():
    model = RooflineModel(peak_gflops=100.0, peak_memory_bw_gbps=50.0)
    with pytest.raises(ValueError):
        model.get_memory_time_ns(0)
    with pytest.raises(ValueError):
        model.get_memory_time_ns(-5)


def test_get_cost_ns_compute_bound():
    model = RooflineModel(peak_gflops=10.0, peak_memory_bw_gbps=100.0)
    # Compute time: 100/10 = 10, Memory time: 100/100 = 1
    assert model.get_cost_ns(100, 100) == 10


def test_get_cost_ns_memory_bound():
    model = RooflineModel(peak_gflops=1000.0, peak_memory_bw_gbps=10.0)
    # Compute time: 100/100 = 1, Memory time: 100/10 = 10
    assert model.get_cost_ns(100, 100) == 10


def test_get_cost_ns_equal():
    model = RooflineModel(peak_gflops=10.0, peak_memory_bw_gbps=10.0)
    # Both times: 100/10 = 10
    assert model.get_cost_ns(100, 100) == 10


def test_get_cost_ns_invalid_ops():
    model = RooflineModel(peak_gflops=10.0, peak_memory_bw_gbps=10.0)
    with pytest.raises(ValueError):
        model.get_cost_ns(0, 100)


def test_get_cost_ns_invalid_bytes():
    model = RooflineModel(peak_gflops=10.0, peak_memory_bw_gbps=10.0)
    with pytest.raises(ValueError):
        model.get_cost_ns(100, 0)
