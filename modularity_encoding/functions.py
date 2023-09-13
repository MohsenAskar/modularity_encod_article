import pandas as pd
import networkx as nx
import itertools
import community
import tqdm
import matplotlib.pyplot as plt
import csv

def modularity_encode(data, 
                      code_system_col='code_system', 
                      patientid_col='patientid', 
                      resolution=1.0, 
                      random_state=42, 
                      output_col='module_number',
                      progress_bar=True):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")
    if code_system_col not in data.columns:
        raise ValueError("Column '{}' does not exist in the input data.".format(code_system_col))
    if patientid_col not in data.columns:
        raise ValueError("Column '{}' does not exist in the input data.".format(patientid_col))
    G = create_code_system_network(data, code_system_col, patientid_col, progress_bar)
    partition = community.best_partition(G, resolution=resolution, random_state=random_state)
    code_system_to_module = {}
    for code_system, module in partition.items():
        code_system_to_module[code_system] = module
        
    encoded_code_systems = data[code_system_col].map(code_system_to_module)
    data[output_col] = encoded_code_systems
    return G, data

def print_edge_list(G, filename):
    if not isinstance(G, nx.Graph):
        raise ValueError("Input data must be a NetworkX graph object.")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['source', 'target', 'weight'])
        for edge in G.edges.data():
            writer.writerow([edge[0], edge[1], edge[2]['weight']])
    print("Edge list saved to file: {}".format(filename))


def create_code_system_network(data, 
                               code_system_col='code_system', 
                               patientid_col='patientid', 
                               progress_bar=True):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")
    if code_system_col not in data.columns:
        raise ValueError("Column '{}' does not exist in the input data.".format(code_system_col))
    if patientid_col not in data.columns:
        raise ValueError("Column '{}' does not exist in the input data.".format(patientid_col))

    code_system_to_patients = data.groupby(code_system_col)[patientid_col].apply(set).to_dict()
    
    G = nx.Graph()
    code_systems = list(code_system_to_patients.keys())
    total_combinations = len(list(itertools.combinations(code_systems, 2)))
    combinations = itertools.combinations(code_systems, 2)
    if progress_bar:
        combinations = tqdm.tqdm(combinations, total=total_combinations)
    for code_system1, code_system2 in combinations:
        common_patients = len(code_system_to_patients[code_system1].intersection(code_system_to_patients[code_system2]))
        if common_patients > 0:
            G.add_edge(code_system1, code_system2, weight=common_patients)
            
    return G

def detect_code_system_communities(G, resolution=1.0, random_state=42):
    if not isinstance(G, nx.Graph):
        raise ValueError("Input data must be a NetworkX graph object.")
    partition = community.best_partition(G, resolution=resolution, random_state=random_state)
    code_system_to_module = {}
    for code_system, module in partition.items():
        code_system_to_module[code_system] = module
    return code_system_to_module

def encode_code_system_to_module(data, code_system_to_module, output_col='module_number'):
    if not isinstance(data, pd.DataFrame) or 'code_system' not in data.columns:
        raise ValueError("Input data must be a Pandas DataFrame with a 'code_system' column.")
    if not isinstance(code_system_to_module, dict):
        raise ValueError("Code system to module mapping must be a dictionary.")
    if not isinstance(output_col, str):
        raise ValueError("Output column name must be a string.")
    encoded_code_systems = data['code_system'].map(code_system_to_module)
    data[output_col] = encoded_code_systems
    return data
