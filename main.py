from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Modules About Hugging Face Transformers
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, LongformerForMaskedLM, Trainer

from serving.inference import inference, inference_torchscript

app = FastAPI()

@app.get('/')
async def root():
    return {"message": "Hello World"}