'''
This utilities file will handle API interactions to get and post data
'''

import json
import os
import requests
from sys import argv

base_url = "https://api.aerobotics.com/farming/surveys/"

def _get_orchid_info(orchard_id):
    url = f"?orchard_id={orchard_id}"
    print('URL', base_url + url)
    api_token = os.getenv("token")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    response = requests.get(base_url + url, headers=headers)
    return response

def _get_tree_information(survey_id):

    url = f"{survey_id}/tree_surveys/"
    api_token = os.getenv("token")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    response = requests.get(base_url + url, headers=headers)

    return response



if __name__ == '__main__':
    orchid_info = _get_orchid_info(str(argv[1]))
    if orchid_info.status_code == 200:
        print('Response successful')
        survey_id = orchid_info.json()["results"][0]
    else:
        print("token state:", os.getenv("token"))
        raise ValueError(orchid_info)
    
    try:
        tree_info = _get_tree_information(survey_id['id'])
        print(tree_info.status_code)
        print(tree_info.json())
    except Exception as e:
        print("Failed")
        print(survey_id)
        raise e