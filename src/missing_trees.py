'''
The functionality will handle determining the coordinates of missing trees
within an orchard
'''
from sklearn.neighbors import BallTree
from scipy.spatial import Delaunay
import triangle as tr
import triangle.plot as tplot
from shapely.geometry import Polygon, Point
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
from utils import _get_orchard_info, _get_tree_information
import os
from sys import argv
from typing import Tuple


def _extract_coordinates(tree_list: list) -> np.ndarray:
    """
    This function will get the longitude and latitude positions per tree into radians.

    Parameters
    ----------
    tree_list:
        List of dictionaries for survey tree metrics.
    """

    coordinates = np.array([[np.radians(tree['lat']), np.radians(tree['lng'])]
                            for tree in tree_list])
    return coordinates


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


def _get_boundary(survey_summary):
    pass


def _get_coords(tree_list: list):
    coord_pairs = np.array([[tree['lat'], tree['lng']]
                            for tree in tree_list])
    coords = np.column_stack([coord_pairs[:, 1], coord_pairs[:, 0]])
    return coords


def _get_coords2(tree_list: list):
    coord_pairs = np.array([[tree['lat'], tree['lng']]
                            for tree in tree_list])
    return coord_pairs[:, 1], coord_pairs[:, 0]


def _apply_delauney(coords: np.ndarray) -> Tuple[Delaunay, np.ndarray]:
    tri = Delaunay(coords)

    def get_triangle_areas(triangle_points):
        A, B, C = triangle_points
        return 0.5 * abs(
            (B[0] - A[0]) * (C[1] - A[1]) - 
            (C[0] - A[0]) * (B[1] - A[1])
        )
    
    area_array = np.array([get_triangle_areas(coords[s]) for s in tri.simplices])
    
    return tri, area_array


def _get_tree_radius():
    """
    This function will get each trees radius and then add the offset radius based on the 
    average tree area + std deviation. This will allow the radius cut-off for neighbouring 
    tree searching to be

    """
    pass


def sketch_plots(tree_list):
    triangle_coords = _get_coords(tree_list)

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
    lower = 1.7
    upper = 2.7
    triangle_coords = _get_coords(tree_list)

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
    triangle_coords = _get_coords(tree_list)

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


def projection_test(tree_list):
    points_lonlat = _get_coords(tree_list)
    lon, lat = _get_coords2(tree_list)
    lon0, lat0 = lon.mean(), lat.mean()
    proj_str = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m"
    transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    transformer_inv = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)

    x, y = transformer.transform(lon, lat)
    points_xy = np.vstack([x, y]).T

    # -----------------------
    # Delaunay
    # -----------------------
    tri = Delaunay(points_xy)

    # -----------------------
    # Compute areas
    # -----------------------
    def triangle_area(pts):
        A, B, C = pts
        return 0.5 * abs(
            (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])
        )

    areas = np.array([
        triangle_area(points_xy[simplex])
        for simplex in tri.simplices
    ])

    A_mean = areas.mean()

    # -----------------------
    # Mark bad triangles
    # -----------------------
    lower = 1.7
    upper = 2.7
    deviation = areas/A_mean.mean()
    bad_mask = (deviation > lower) & (deviation < upper)
    # threshold = 1.5  # area factor
    # bad_mask = areas > (threshold * A_mean)

    # -----------------------
    # Find midpoints of longest edges in bad triangles
    # -----------------------
    new_points_xy = []

    for simplex, is_bad in zip(tri.simplices, bad_mask):
        if not is_bad:
            continue
        pts = points_xy[simplex]
        edges = [
            (pts[0], pts[1]),
            (pts[1], pts[2]),
            (pts[2], pts[0])
        ]
        lengths = [np.linalg.norm(e1 - e2) for e1, e2 in edges]
        idx_longest = np.argmax(lengths)
        A, B = edges[idx_longest]
        M = (A + B) / 2
        new_points_xy.append(M)

    new_points_xy = np.array(new_points_xy)

    # Transform new points back to lon/lat
    if len(new_points_xy) > 0:
        mx, my = new_points_xy.T
        mlon, mlat = transformer_inv.transform(mx, my)
        new_points_lonlat = np.vstack([mlon, mlat]).T
    else:
        new_points_lonlat = np.empty((0, 2))

    # -----------------------
    # Combine for new mesh
    # -----------------------
    combined_points_lonlat = np.vstack([points_lonlat, new_points_lonlat])

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
    ax.plot(points_lonlat[:, 0], points_lonlat[:, 1], 'ko', label='Original points')

    # Plot new points in red
    if len(new_points_lonlat) > 0:
        ax.plot(new_points_lonlat[:, 0], new_points_lonlat[:, 1], 'ro', label='New midpoints')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("New Delaunay mesh with midpoints for bad triangles")
    ax.legend()
    plt.show()


def new_points(tree_list):
    lower = 1.7
    upper = 2.7
    triangle_coords = _get_coords(tree_list)

    tri, tri_areas = _apply_delauney(triangle_coords)

    deviation = tri_areas/tri_areas.mean()
    bad_mask = (deviation > lower) & (deviation < upper)

    longest_edges = []
    longest_lengths = []
    other_edges = []


if __name__ == '__main__':
    # print(tr.__file__)
    # print(dir(tr))
    orchid_info = _get_orchard_info(str(argv[1]))
    if orchid_info['status'] == 200:
        print('Response successful')
        survey_id = orchid_info['id']
        polygon_coords = orchid_info['polygon']
    else:
        print("token state:", os.getenv("token"))
        raise ValueError(orchid_info['status'])
    
    try:
        tree_info = _get_tree_information(survey_id)
    except Exception as e:
        print("Failed")
        print(survey_id)
        raise e
    
    projection_test(tree_info['trees'])