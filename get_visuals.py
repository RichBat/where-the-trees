
from src.utils import get_aero_tree_info
import src.tree_vis as tv
from sys import argv


if __name__ == '__main__':
    orchard_id = int(argv[1])
    trees, missing = get_aero_tree_info(orchard_id)

    print("Tree coord plot")
    tv.graph_trees(trees)

    print("Tree Triangularisation")
    tv.sketch_plots(trees)

    print("All Tree Triangle Area Deviations")
    tv.sketch_triangle_areas(trees, no_mask=True)

    print("Bad Tree Triangle Area Deviations")
    tv.sketch_triangle_areas(trees)

    print("Bad Triangle Longest Edges")
    tv.sketch_bad_triangle_long_edge(trees)

    print("Tree coords with missing tree candidates")
    tv.plot_points(orchard_id)

    print("Tree coords with missing tree candidates")
    tv.plot_points(orchard_id, False)