from scipy.spatial import Delaunay
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from typing import Tuple
from missing_trees import _get_coords


def _graph_coords(tree_list: list) -> tuple:
    lat_coords = [[tree['lat'] for tree in tree_list]]
    lng_coords = [[tree['lng'] for tree in tree_list]]
    return lng_coords, lat_coords


def _graph_trees(tree_list: list):
    lngs, lats = _graph_coords(tree_list)

    plt.figure(figsize=(8, 8))
    plt.scatter(lngs, lats, c='blue')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def _apply_delauney(coords: np.ndarray) -> Tuple[Delaunay, np.ndarray]:
    """
    Gets the Delaunay triangles and areas using vertically stacked
    coordinate arrays
    """
    tri = Delaunay(coords)

    def get_triangle_areas(triangle_points):
        A, B, C = triangle_points
        return 0.5 * abs(
            (B[0] - A[0]) * (C[1] - A[1]) - 
            (C[0] - A[0]) * (B[1] - A[1])
        )
    area_array = np.array([get_triangle_areas(coords[s]) for s in tri.simplices])
    return tri, area_array


def sketch_plots(tree_list):
    long, lat = _get_coords(tree_list)
    triangle_coords = np.column_stack([long, lat])

    triangles, tri_areas = _apply_delauney(triangle_coords)

    plt.figure(figsize=(8, 8))
    # plt.plot(triangle_coords[:, 0], triangle_coords[:, 1], color='blue', marker='o')

    for triangle in triangle_coords[triangles.simplices]:
        t = np.vstack([triangle, triangle[0]])
        plt.plot(t[:, 0], t[:, 1], 'k-')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Points + Delaunay Triangulation')
    plt.show()


def sketch_triangle_areas(tree_list):
    lower = 1.5
    upper = 2.5
    long, lat = _get_coords(tree_list)
    triangle_coords = np.column_stack([long, lat])

    tri, tri_areas = _apply_delauney(triangle_coords)

    deviation = tri_areas/tri_areas.mean()
    mask = (deviation > lower) & (deviation < upper)

    triangles = np.array([triangle_coords[s] for s in tri.simplices])
    deviation = deviation[mask]
    triangles_coloured = triangles[mask]
    triangles = triangles[~mask]
    fig, ax = plt.subplots(figsize=(8, 8))

    # Build a PolyCollection
    coll = PolyCollection(triangles, facecolor='white', edgecolor='k')

    ax.add_collection(coll)

    colour_coll = PolyCollection(triangles_coloured, array=deviation, cmap='viridis', edgecolor='k')

    ax.add_collection(colour_coll)

    ax.autoscale()
    ax.set_aspect('equal')

    plt.colorbar(colour_coll, ax=ax, label='Area / Mean Area')
    plt.title('Delaunay Triangles Colored by Relative Area')
    plt.show()


def sketch_new_triangles(tree_list):
    lower = 1.7
    upper = 2.7
    long, lat = _get_coords(tree_list)
    triangle_coords = np.column_stack([long, lat])

    tri, tri_areas = _apply_delauney(triangle_coords)

    deviation = tri_areas/tri_areas.mean()
    bad_mask = (deviation > lower) & (deviation < upper)

    longest_edges = []
    longest_lengths = []
    other_edges = []

    for is_bad, simplex in zip(bad_mask, tri.simplices):
        if not is_bad:
            continue  # skip good triangles

        pts = triangle_coords[simplex]
        A, B, C = pts
        edges = [(A, B), (B, C), (C, A)]
        lengths = [np.linalg.norm(e[1] - e[0]) for e in edges]
        longest_idx = np.argmax(lengths)
        longest_edges.append(edges[longest_idx])
        longest_lengths.append(lengths[longest_idx])
        other_edges.extend([edges[i] for i in range(3) if i != longest_idx])
    
    fig, ax = plt.subplots(figsize=(8, 8))

    # Longest edges (color by length)
    lc_long = LineCollection(longest_edges, colors='red', linewidths=2)
    ax.add_collection(lc_long)

    # Other edges of bad triangles, neutral
    lc_other = LineCollection(other_edges, colors='grey', linewidths=1)
    ax.add_collection(lc_other)

    # Points for context
    ax.plot(triangle_coords[:, 0], triangle_coords[:, 1], 'ko', markersize=3)

    plt.colorbar(lc_long, ax=ax, label='Longest Edge Length')
    ax.autoscale()
    ax.set_aspect('equal')
    plt.title('Longest Edges of Bad Triangles (Red)')
    plt.show()


def plot_points(original_pts, candidate_coords, transformer):
    '''
    This function can be used to plot the coordinates of the proposed missing trees.
    '''
    combined_points_lonlat = np.vstack([original_pts, candidate_coords])

    # Project again for Delaunay (or just reuse xy)
    x2, y2 = transformer.transform(combined_points_lonlat[:, 0], combined_points_lonlat[:, 1])
    combined_points_xy = np.vstack([x2, y2]).T

    tri2 = Delaunay(combined_points_xy)

    # -----------------------
    # Plot
    # -----------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot new mesh triangles
    for simplex in tri2.simplices:
        pts = combined_points_lonlat[simplex]
        poly = Polygon(pts, edgecolor='gray', facecolor='none')
        ax.add_patch(poly)

    # Plot original points
    ax.plot(original_pts[:, 0], original_pts[:, 1], 'ko', label='Original points')

    # Plot new points in red
    if len(candidate_coords) > 0:
        ax.plot(candidate_coords[:, 0], candidate_coords[:, 1], 'ro', label='New midpoints')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("New Delaunay mesh with midpoints for bad triangles")
    ax.legend()
    plt.show()