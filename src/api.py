from flask import Flask, jsonify
from .missing_trees import find_missing_trees
app = Flask(__name__)


def missing_trees_dict(orchard_id):
    """
    This function gets the coordinates of the missing trees as a list
    of missing tree coordinates structured as a list of dictionaries
    and then returns it with the relevant dictionary structre.
    """
    missing_tree_coords = find_missing_trees(orchard_id)
    tree_list = [{"lat": float(tree[1]), "lng": float(tree[0])} for tree in missing_tree_coords]
    return {"missing_trees": tree_list}


@app.route('/orchards/<int:orchard_id>/missing-trees', methods=['GET'])
def missing_trees(orchard_id):
    """
    Returns the coordinates of the missing trees in JSON format
    for the requested orchard by orchard id.
    Parameters
    ----------
    orchard_id:
        The id of the orchard which is retreived from the GET request.
    """
    try:
        missing_tree_response = missing_trees_dict(orchard_id)
        return jsonify(missing_tree_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
