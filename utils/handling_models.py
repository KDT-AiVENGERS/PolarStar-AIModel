import torch

# Modules About Hugging face
from transformers import AutoTokenizer, AutoModel

import os
import shutil
import json

def get_models():
    all_list = os.listdir('models')
    pt_list = list(filter(lambda x: x[-3:] == '.pt', all_list))

    return list(map(lambda x: x[:-3], pt_list))

def generating_git_model(model_ref: str, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_ref)

    inputs = tokenizer([
        'sample tokenizing sentence 1',
        'sample tokenizing sentence 2'
    ], return_tensors='pt', padding=True, truncation=True)

    model = AutoModel.from_pretrained(model_ref, torchscript=True)

    model_jit = torch.jit.trace(model, list(inputs.values()))

    model_jit.save(f'models/{model_name}.pt')

    # change state
    with open('data/server_state.json', 'r') as fp:
        server_state = json.load(fp)

    server_state['current_model_ref'] = model_ref

    with open('data/server_state.json', 'w', encoding='utf-8') as fp:
        json.dump(server_state, fp)

def set_model(model_name: str):
    shutil.copy(f'models/{model_name}.pt', 'models/current_model.pt')

def delete_model(model_name: str):
    if os.isfile(f'models/{model_name}'):
        os.remove(f'models/{model_name}')

