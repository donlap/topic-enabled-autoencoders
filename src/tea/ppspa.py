"""The pseudo-point successive projection algorithm (ppSPA) of Jin et al. (2024).

ppSPA estimates the K vertices of a simplex from noisy points inside the simplex, in three steps.
First, the points are projected onto the (K-1)-dimensional affine plane that fits the points best.
Next, each point with enough neighbors is replaced by the mean of its nearest neighbors.
Finally, the successive projection algorithm (SPA) of Araujo et al. (2001) selects K of these pseudo-points as vertices.
Each original point is then written in barycentric coordinates over the vertices.
This implementation follows the implementation of the authors at https://github.com/gabrielmoryoussef/VertexHunting.
"""

import numpy as np
from scipy.optimize import lsq_linear, nnls
from scipy.spatial import cKDTree

RADIUS_DIVISOR = 5.0
NEIGHBORS = 4
MIN_NEIGHBORS = 3


def project(points, K):
    """Return the coordinates of the rows of points in the span of the top K-1 principal directions."""
    data = points.T
    centered = data - np.outer(data.mean(axis=1), np.ones(data.shape[1]))
    left, _, _ = np.linalg.svd(centered, full_matrices=True)
    return np.dot(left[:, :K - 1].T, data).T


def pseudo_points(points, radius, neighbors=NEIGHBORS, min_neighbors=MIN_NEIGHBORS):
    """Replace each point by the mean of its nearest neighbors, and drop the points with few neighbors.

    Let n_min = min_neighbors and n_max = neighbors.
    A point with fewer than n_min points within the radius is dropped.
    A point with at most n_max points within the radius is replaced by the mean of these points.
    Any other point is replaced by the mean of its n_max nearest points.
    """
    tree = cKDTree(points)
    within = tree.query_ball_tree(tree, r=radius)
    out = []
    for i, ball in enumerate(within):
        if len(ball) < min_neighbors:
            continue
        if len(ball) <= neighbors:
            out.append(points[ball].mean(axis=0))
        else:
            out.append(points[tree.query(points[i], k=neighbors)[1]].mean(axis=0))
    return np.array(out).reshape(-1, points.shape[1])


def successive_projection(points, K):
    """Return the K rows of points that SPA selects after a leading coordinate of one is added to every point."""
    residual = np.column_stack((np.ones(points.shape[0]), points))
    chosen = np.zeros((K, points.shape[1]))
    for k in range(K):
        j = int(np.argmax((residual ** 2).sum(axis=1)))
        chosen[k] = points[j]
        direction = residual[j] / np.sqrt((residual[j] ** 2).sum())
        residual -= np.outer(residual @ direction, direction)
    return chosen


def barycentric(points, vertices, C=1e3):
    """Return the non-negative weights over the vertices that best reproduce each point and sum to one.

    The weights solve a non-negative least-squares problem with the extra equation C sum_k w_k = C.
    The weights are then divided by their sum.
    """
    K = vertices.shape[0]
    A = np.vstack([vertices.T, C * np.ones((1, K))])
    out = np.zeros((points.shape[0], K))
    for i, x in enumerate(points):
        b = np.append(x, C)
        try:
            w, _ = nnls(A, b)
        except (RuntimeError, ValueError):
            w = lsq_linear(A, b, bounds=(0, np.inf), method="bvls").x
        s = w.sum()
        out[i] = w / s if s > 1e-12 else np.full(K, 1.0 / K)
    return out


def loadings(W):
    """Return the barycentric loadings H of the rows of W, with one column per column of W.

    If there are fewer pseudo-points than columns of W, SPA runs on the projected points instead.
    """
    K = int(W.shape[1])
    pts = project(W, K)
    radius = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))) / RADIUS_DIVISOR
    pseudo = pseudo_points(pts, radius)
    if pseudo.shape[0] < K:
        pseudo = pts
    return barycentric(pts, successive_projection(pseudo, K))
