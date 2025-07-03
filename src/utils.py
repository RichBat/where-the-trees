'''
This utilities file will handle API interactions to get and post data
'''

import os
import requests
from sys import argv

base_url = "https://api.aerobotics.com/farming/surveys/"


def _get_orchard_info(orchard_id: int) -> dict:
    """
    This function will query the API to get the orchard survey information.

    Parameters
    ----------
    orchard_id:
        The id for the specific orchard to be queried.
    """
    if type(orchard_id) is not int:
        raise TypeError("Supplied Orchard ID is not an integer")
    id = str(orchard_id)
    url = f"?orchard_id={id}"
    api_token = os.getenv("token")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    response = requests.get(base_url + url, headers=headers)
    if response.status_code == 200:
        response_data = response.json()['results'][0]
        return {'status': response.status_code, 'id': response_data['id']}
    else:
        return {'status': response.status_code, 'id': None, 'polygon': None}


def _get_tree_information(survey_id: int) -> dict:

    api_token = os.getenv("token")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }
    id = str(survey_id)
    missing_tree_response = requests.get(base_url + id + "/tree_survey_summaries/", headers=headers)
    response = requests.get(base_url + id + "/tree_surveys/", headers=headers)

    if response.status_code == 200 and missing_tree_response.status_code == 200:
        response_data = response.json()['results']
        missing_tree_count = missing_tree_response.json()["missing_tree_count"]
        return {'status': response.status_code,
                'trees': response_data,
                'missing': missing_tree_count}
    else:
        bad_status = response.status_code if response.status_code != 200 else missing_tree_response.status_code
        return {'status': bad_status,
                'trees': None}


def get_aero_tree_info(orchard_id):
    """
    A utility file to get the relevant orchard survey tree and missing tree information
    for a given orchard.

    NOTE
    Need to add status checking for API utilities that are sensible
    If failed here, there must be a try block that can pass
    on the valid error based on what is raised.
    """
    orchard_response = _get_orchard_info(orchard_id)
    survey_id = orchard_response['id']
    tree_results = _get_tree_information(survey_id)
    return tree_results['trees'], tree_results['missing']


if __name__ == '__main__':
    orchard_id = int(argv[1])
    orchid_info = _get_orchard_info(orchard_id)
    if orchid_info['status'] == 200:
        print('Response successful')
        survey_id = orchid_info['id']
    else:
        raise ValueError(orchid_info['status'])
    try:
        tree_info = _get_tree_information(survey_id)
        print(tree_info['status'])
        print(tree_info['trees'])
    except Exception as e:
        print("Failed")
        print(survey_id)
        raise e
    