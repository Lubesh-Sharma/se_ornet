# Geodesic Error Computation

This document explains the geodesic error computation feature added to SE-ORNet for evaluating shape correspondence quality.

## What is Geodesic Error?

Geodesic error is a standard metric in shape correspondence that measures the quality of predicted correspondences by computing the **geodesic distance** (distance along the surface) between the predicted point and the ground truth point on the target shape. This is more accurate than Euclidean distance for evaluating correspondences on curved surfaces.

## Formula

For a predicted correspondence `p` and ground truth `g` on target shape `T`:

```
geodesic_error = mean(geodesic_dist(p_i, g_i)) / geodesic_diameter(T)
```

where:
- `geodesic_dist(p_i, g_i)` is the geodesic distance between predicted and ground truth points
- `geodesic_diameter(T)` is the maximum geodesic distance on the target shape (used for normalization)

## Implementation

The geodesic error computation consists of two main functions:

### 1. `compute_geodesic_distance_matrix(points, k_neighbors=20)`

Computes approximate geodesic distances using a graph-based approach:
- Builds a k-nearest neighbor graph on the point cloud
- Computes Euclidean distances for graph edges
- Uses iterative relaxation (similar to Bellman-Ford) to compute shortest paths
- Returns an N×N matrix of geodesic distances

### 2. `compute_geodesic_error(pred_indices, gt_indices, target_points, k_neighbors=20, normalize=True)`

Computes the geodesic error for predicted correspondences:
- Computes the geodesic distance matrix for the target shape
- Looks up distances between predicted and ground truth points
- Normalizes by the geodesic diameter
- Returns the mean geodesic error

## Usage

### During Testing

The geodesic error is automatically computed during testing and logged as `"geodesic_error"` in the test metrics.

```bash
# Run testing with geodesic error computation (default)
python train.py --do_train false --resume_from_checkpoint <path>

# Disable geodesic error computation (faster but less accurate)
python train.py --do_train false --resume_from_checkpoint <path> --compute_geodesic_error false
```

### Command-line Arguments

- `--compute_geodesic_error`: Enable/disable geodesic error computation (default: `True`)
  - Set to `false` to skip geodesic error computation for faster testing

### Accessing Results

The geodesic error is logged along with other metrics:

```python
# In the test output, you'll see:
{
    "test/geodesic_error": <value>,
    "test/acc_mean_dist": <value>,
    "test/acc_0.00": <value>,
    ...
}
```

## Performance Considerations

Computing geodesic distances is more expensive than Euclidean distances:

- **Time complexity**: O(k × N²) where N is the number of points and k is the number of neighbors
- **Memory**: O(N²) for storing the distance matrix
- **Typical runtime**: 1-5 seconds per test sample for N=1024 points

For faster testing, you can:
1. Disable geodesic error: `--compute_geodesic_error false`
2. Reduce k_neighbors (uses `--num_neighs` parameter)
3. Use GPU acceleration (automatically enabled if CUDA is available)

## Interpretation

- **Lower is better**: Geodesic error ranges from 0 (perfect) to 1 (maximum error with normalization)
- **Comparison with Euclidean distance**: Geodesic error is always ≥ Euclidean distance
- **Typical values**:
  - < 0.1: Excellent correspondence
  - 0.1 - 0.3: Good correspondence
  - > 0.5: Poor correspondence

## Testing

A test script is provided to validate the implementation:

```bash
# Run the test script
python test_geodesic.py
```

This tests:
- Geodesic distance matrix computation on a simple grid
- Geodesic error for perfect vs. random predictions
- Performance on typical point cloud sizes (1024 points)

## References

The geodesic error metric is commonly used in shape correspondence papers:
- SHREC benchmarks use geodesic error as a standard metric
- Many shape matching papers report results normalized by geodesic diameter
- Graph-based approximation is a standard approach for point clouds without mesh connectivity

## Limitations

1. **Approximation**: Uses graph-based shortest path, not exact geodesic on surface
2. **Computational cost**: More expensive than Euclidean distance
3. **Requires dense sampling**: Accuracy depends on point cloud density
4. **No mesh connectivity**: Doesn't use mesh faces (only point positions)

## Future Improvements

Possible enhancements:
- Use mesh connectivity if available for exact geodesic computation
- Cache geodesic distance matrices to avoid recomputation
- Implement faster approximation algorithms (e.g., heat method)
- Support batch computation for multiple test samples
