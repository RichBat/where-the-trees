'''
The functionality will handle determining the coordinates of missing trees
within an orchard
'''
from sklearn.neighbors import BallTree
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from utils import _get_orchard_info, _get_tree_information
import os
from sys import argv

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


def _get_coords(tree_list: list):
    coord_pairs = np.array([[tree['lat'], tree['lng']]
                            for tree in tree_list])
    coords = np.column_stack([coord_pairs[:, 1], coord_pairs[:, 0]])
    return coords


def _apply_delauney(coords: np.ndarray):
    tri = Delaunay(coords)
    return tri


def _get_tree_radius():
    """
    This function will get each trees radius and then add the offset radius based on the 
    average tree area + std deviation. This will allow the radius cut-off for neighbouring 
    tree searching to be

    """
    pass


def sketch_plots(tree_list):
    triangle_coords = _get_coords(tree_list)

    triangles = _apply_delauney(triangle_coords)

    plt.figure(figsize=(8, 8))
    # plt.plot(triangle_coords[:, 0], triangle_coords[:, 1], color='blue', marker='o')

    for triangle in triangle_coords[triangles.simplices]:
        t = np.vstack([triangle, triangle[0]])
        plt.plot(t[:, 0], t[:, 1], 'k-')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Points + Delaunay Triangulation')
    plt.show()


if __name__ == '__main__':
    orchid_info = _get_orchard_info(str(argv[1]))
    if orchid_info.status_code == 200:
        print('Response successful')
        survey_id = orchid_info.json()["results"][0]
    else:
        print("token state:", os.getenv("token"))
        raise ValueError(orchid_info)
    
    try:
        tree_info = _get_tree_information(survey_id['id'])
    except Exception as e:
        print("Failed")
        print(survey_id)
        raise e
    
    sketch_plots(tree_info.json()['results'])