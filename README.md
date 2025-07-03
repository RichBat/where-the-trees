# Where the missing trees go?

This codebase provides functionality to identify the location of missing trees within an orchard  
using a combination of the provided tree coordinates and the quantity of missing trees to position.  
The provided functionality provides: 
- Access to an external API to retrieve said tree data.
- Functions to process the tree coordinates to evaluate likely missing tree locations.
- Selection of the subset of missing trees coordinates that are the best fit. 

These are then hosted on an API which can be requested to provide the missing tree coordinates with  
a GET request. There is also additional functionality to visualise the tree coordinate grid at  
differing stages for investigation.

## Table of Contents
- [Codebase Structure](#Codebase-Structure)
- [Installation](#Installation)
- [Usage](#Usage)
- [Running API via Docker](#Running-API-via-Docker)
- [Caveats](#Caveats)
- [Additional Functionality](#Additional-Functionality)

## Codebase Structure
```
where-the-trees/
|   .dockerignore
|   .env
|   .gitignore
|   docker-compose.yaml
|   dockerfile
|   README.md
|   requirements.txt
|
\---src
        api.py
        missing_trees.py
        tree_vis.py
        utils.py

```

## Installation
If wanting to interact with all of the code and additional functionality for visualisation or testing use the steps below.
  
If only going to start up the container and host the API these installation steps are not necessary.
1. Clone the repository
```bash
git clone https://github.com/RichBat/where-the-trees.git
```
2. Set-up a Virtual Environment
- Windows (cmd)
```cmd
python3 -m venv\Scripts\activate
```
- Unix
```bash
python3 -m venv venv
```
3. Activate the Virtual Environment  
- Windows
```bash
source venv/Scripts/activate
```
- Unix
```bash
source venv/bin/activate
```
4. Install packages using PyPi
```bash
pip install -r requirements.txt 
```
5. Set-up the environmental variables in .env
```bash
export token={your API token here}
```

## Usage

The functionality has currently been written to only interact with the Aerobotics API  
to retrieve relevant tree survey data. The primary function of this code base is to  
provide a public API which is locally hosted by default at the endpoint below:
```
endpoint = http://localhost:5000
```
Currently the API can only service requests of the format below and will return a JSON  
response showing the coordinates of the missing trees in latitude and longitude.
```bash
curl https://{endpoint}/orchards/{orchard_id}/missing-trees
```

## Running API via Docker

Docker needs to be set-up on your machine and accessable via a terminal or command line  
interface. If running this without the full installion the steps are:

1. Clone the repository (if not already done)
```bash
git clone https://github.com/RichBat/where-the-trees.git
```
2. Set-up the environmental variables in .env if not present
```bash
export token={your API token here}
```
3. Run the docker composition to build the image and start the container
```bash
docker-compose up --build
```  
With the container up and running, the API will be hosted on the endpoint provided  
in the docker container and the default endpoint is shown under [Usage](#Usage).

## Caveats

The deviation lower and upper bounds for the best fit missing trees is fixed  
to specific values so as to capture the most likely missing trees. The next fix  
would be to address further criteria to avoid poor candidate missing trees from  
being considered so as to make this dynamic.

## Additional Functionality
#### Visualisation of intermediate stages
Visualisations for intermediate steps in the missing tree candidate coordinates
can be generated using the code below:
```
python3 get_visuals.py {orchard_id} 
```