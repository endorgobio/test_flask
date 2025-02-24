
import pandas as pd
import requests
import json 

# def read_data(json_path):
    
#     try:
#         data =  requests.get(json_path)
#         data = json.loads(data.text)
#     except:
#         f = open(json_path)
#         data = json.load(f)
        
def read_data(json_path, 
              url_coord, 
              url_dist, 
              url_demand):

    try:
        response = requests.get(json_path)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = json.loads(response.text)
        data['df_coord'] = pd.read_csv(url_coord) 
        data['df_dist'] = pd.read_csv(url_dist) 
        data['df_demand'] = pd.read_csv(url_demand) 
    except requests.exceptions.RequestException as e:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                data['df_coord'] = pd.read_csv(url_coord) 
                data['df_dist'] = pd.read_csv(url_dist) 
                data['df_demand'] = pd.read_csv(url_demand) 
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON content in file: {json_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the file: {e}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON content from URL: {json_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while fetching data from URL: {e}")
    
    return data

# # Read data
# df_coord = pd.read_csv('https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh') # coordenadas
# df_dist = pd.read_csv('https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV') # distancias
# df_demand = pd.read_csv('https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe') # demandas escenario base
# parameters['df_coord'] = df_coord
# parameters['df_dist'] = df_dist
# parameters['df_demand'] = df_demand