'''
This utilities file will handle API interactions to get and post data
'''

# import json
import os
import requests
from sys import argv
# import numpy as np

base_url = "https://api.aerobotics.com/farming/surveys/"


def _get_orchard_info(orchard_id: str) -> dict:
    """
    This function will query the API to get the orchard survey information.

    Parameters
    ----------
    orchard_id:
        The id for the specific orchard to be queried.
    """
    url = f"?orchard_id={orchard_id}"
    print('URL', base_url + url)
    api_token = os.getenv("token")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    response = requests.get(base_url + url, headers=headers)
    if response.status_code == 200:
        response_data = response.json()['results'][0]
        polygon_pairs = [list(map(float, p.split(','))) for p in 
                         response_data['polygon'].split(' ')]
        return {'status': response.status_code, 'id': response_data['id'],
                'polygon': polygon_pairs}
    else:
        return {'status': response.status_code, 'id': None, 'polygon': None}


def _get_tree_information(survey_id: str) -> dict:

    url = f"{survey_id}/tree_surveys/"
    api_token = os.getenv("token")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    response = requests.get(base_url + url, headers=headers)

    if response.status_code == 200:
        response_data = response.json()['results']
        return {'status': response.status_code,
                'trees': response_data}
    else:
        return {'status': response.status_code,
                'trees': None}

    return response


if __name__ == '__main__':
    orchid_info = _get_orchard_info(str(argv[1]))
    if orchid_info['status'] == 200:
        print('Response successful')
        survey_id = orchid_info['id']
    else:
        print("token state:", os.getenv("token"))
        raise ValueError(orchid_info['status'])
    
    try:
        tree_info = _get_tree_information(survey_id)
        print(tree_info['status'])
        print(tree_info['trees'])
    except Exception as e:
        print("Failed")
        print(survey_id)
        raise e
    