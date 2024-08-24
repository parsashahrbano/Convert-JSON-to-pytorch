import json
import os
import torch

source_dir = '/directory-to/json_data'
data_tensors = []

for filename in os.listdir(source_dir):
    if filename.endswith('.json'):
        with open(os.path.join(source_dir, filename), 'r') as file:
            data = json.load(file)
            for item in data:
                numerical_data = item.get("data", [])
                data_tensor = torch.Tensor(numerical_data)
                data_tensors.append(data_tensor)

destination_dir = '/destination-folder/model'

for i, tensor in enumerate(data_tensors):
    torch.save(tensor, os.path.join(destination_dir, f'data_tensor_{i}.pt'))
