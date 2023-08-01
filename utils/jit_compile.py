import torch

# Modules About Hugging face
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

inputs = tokenizer([
    'test for neuron compiler',
    'very happy'
], return_tensors='pt', padding=True, truncation=True)

model = AutoModel.from_pretrained('bert-base-multilingual-cased', torchscript=True)

model_jit = torch.jit.trace(model, list(inputs.values()))

model_jit.save('jit_test_model.pt')
