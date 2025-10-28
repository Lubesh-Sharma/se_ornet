#!/usr/bin/env python
"""
Simple test script to verify geodesic error computation.
"""

import torch
import numpy as np
from models.metrics.metrics import compute_geodesic_error, compute_geodesic_distance_matrix


def test_geodesic_distance_matrix():
    """Test geodesic distance matrix computation on a simple grid."""
    print("Testing geodesic distance matrix computation...")
    
    # Create a simple 5x5 grid of points
    x = torch.linspace(0, 1, 5)
    y = torch.linspace(0, 1, 5)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten(), torch.zeros(25)], dim=1)
    
    print(f"Points shape: {points.shape}")
    
    # Compute geodesic distances
    geodesic_dist = compute_geodesic_distance_matrix(points, k_neighbors=8)
    
    print(f"Geodesic distance matrix shape: {geodesic_dist.shape}")
    print(f"Max geodesic distance: {geodesic_dist.max().item():.4f}")
    print(f"Min non-zero geodesic distance: {geodesic_dist[geodesic_dist > 0].min().item():.4f}")
    
    # Check that diagonal is zero
    assert torch.allclose(geodesic_dist.diag(), torch.zeros(25)), "Diagonal should be zero"
    
    # Check symmetry
    assert torch.allclose(geodesic_dist, geodesic_dist.T), "Matrix should be symmetric"
    
    print("✓ Geodesic distance matrix test passed!")


def test_geodesic_error():
    """Test geodesic error computation."""
    print("\nTesting geodesic error computation...")
    
    # Create random points
    torch.manual_seed(42)
    N = 100
    target_points = torch.randn(N, 3)
    
    # Create predictions and ground truth
    # Perfect prediction case
    pred_indices = torch.arange(N)
    gt_indices = torch.arange(N)
    
    geodesic_error_perfect = compute_geodesic_error(pred_indices, gt_indices, target_points, k_neighbors=10)
    print(f"Geodesic error (perfect prediction): {geodesic_error_perfect.item():.6f}")
    assert geodesic_error_perfect.item() < 1e-5, "Perfect prediction should have near-zero error"
    
    # Random prediction case
    pred_indices = torch.randperm(N)
    gt_indices = torch.arange(N)
    
    geodesic_error_random = compute_geodesic_error(pred_indices, gt_indices, target_points, k_neighbors=10)
    print(f"Geodesic error (random prediction): {geodesic_error_random.item():.6f}")
    assert geodesic_error_random.item() > 0, "Random prediction should have positive error"
    
    print("✓ Geodesic error test passed!")


def test_performance():
    """Test performance on typical point cloud size."""
    print("\nTesting performance on typical point cloud size...")
    
    import time
    torch.manual_seed(42)
    N = 1024
    target_points = torch.randn(N, 3)
    
    if torch.cuda.is_available():
        target_points = target_points.cuda()
        print("Using CUDA")
    
    pred_indices = torch.randperm(N, device=target_points.device)
    gt_indices = torch.arange(N, device=target_points.device)
    
    start_time = time.time()
    geodesic_error = compute_geodesic_error(pred_indices, gt_indices, target_points, k_neighbors=20)
    elapsed_time = time.time() - start_time
    
    print(f"Geodesic error: {geodesic_error.item():.6f}")
    print(f"Computation time: {elapsed_time:.2f} seconds")
    
    if elapsed_time > 10:
        print("⚠ Warning: Computation is slow for 1024 points")
    else:
        print("✓ Performance test passed!")


if __name__ == "__main__":
    try:
        test_geodesic_distance_matrix()
        test_geodesic_error()
        test_performance()
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
