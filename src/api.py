from flask import Flask, jsonify
from missing_trees import find_missing_trees
from utils import get_aero_tree_info
app = Flask(__name__)


def missing_trees_dict(survey_trees, num_missing):
    missing_tree_coords = find_missing_trees(survey_trees, num_missing)
    tree_list = [{"lat": float(tree[1]), "lng": float(tree[0])} for tree in missing_tree_coords]
    return {"missing_trees": tree_list}


@app.route('/orchards/<int:orchard_id>/missing-trees', methods=['GET'])
def missing_trees(orchard_id):
    try:
        trees, missing = get_aero_tree_info(orchard_id)
        missing_tree_response = missing_trees_dict(trees, missing)
        return jsonify(missing_tree_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
