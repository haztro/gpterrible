import json
import numpy as np
import torch

n_head = 6
n_layer = 6

def tensor_to_dict(tensor):
    return {
        'data': tensor.cpu().tolist(),
        'shape': list(tensor.shape)
    }

state_dict = torch.load('model.pt', map_location='cpu')

json_model = {}
json_model['tok_emb'] = [tensor_to_dict(state_dict['token_embedding_table.weight'])]
json_model['pos_emb'] = [tensor_to_dict(state_dict['position_embedding_table.weight'])]
json_model['blocks'] = []
json_model['ln_f'] = []
json_model['lm_head'] = []

for i in range(n_layer):
    block = {}
    block['multihead'] = {'heads': [], 'proj': []}
    block['ffwd'] = []
    block['ln1'] = []
    block['ln2'] = []
    
    for j in range(n_head):
        head = {}
        head['key'] = [tensor_to_dict(state_dict[f'blocks.{i}.sa.heads.{j}.key.weight'])]
        head['query'] = [tensor_to_dict(state_dict[f'blocks.{i}.sa.heads.{j}.query.weight'])]
        head['value'] = [tensor_to_dict(state_dict[f'blocks.{i}.sa.heads.{j}.value.weight'])]
        block['multihead']['heads'].append(head)
        
    block['multihead']['proj'].append(tensor_to_dict(state_dict[f'blocks.{i}.sa.proj.weight']))
    block['multihead']['proj'].append(tensor_to_dict(state_dict[f'blocks.{i}.sa.proj.bias']))
    block['ffwd'].append(tensor_to_dict(state_dict[f'blocks.{i}.ffwd.net.0.weight']))
    block['ffwd'].append(tensor_to_dict(state_dict[f'blocks.{i}.ffwd.net.0.bias']))
    block['ffwd'].append(tensor_to_dict(state_dict[f'blocks.{i}.ffwd.net.2.weight']))
    block['ffwd'].append(tensor_to_dict(state_dict[f'blocks.{i}.ffwd.net.2.bias']))
    block['ln1'].append(tensor_to_dict(state_dict[f'blocks.{i}.ln1.weight']))
    block['ln1'].append(tensor_to_dict(state_dict[f'blocks.{i}.ln1.bias']))
    block['ln2'].append(tensor_to_dict(state_dict[f'blocks.{i}.ln2.weight']))
    block['ln2'].append(tensor_to_dict(state_dict[f'blocks.{i}.ln2.bias']))
    json_model['blocks'].append(block)

json_model['ln_f'].append(tensor_to_dict(state_dict['ln_f.weight']))
json_model['ln_f'].append(tensor_to_dict(state_dict['ln_f.bias']))
json_model['lm_head'].append(tensor_to_dict(state_dict['lm_head.weight']))
json_model['lm_head'].append(tensor_to_dict(state_dict['lm_head.bias']))

with open('../model.json', 'w') as f:
    json.dump(json_model, f)
