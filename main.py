from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

# Modules About Hugging Face Transformers
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, LongformerForMaskedLM, Trainer

from infra.inference import inference, inference_torchscript

from utils.handling_models import get_models, generating_jit_model, set_model, delete_model
from utils.handling_data import update_jd_data, create_v_jd_data

app = FastAPI()

@app.get('/')
async def helloworld():
  return {"message": "Hello World"}

@app.get('/models')
def getmodels():
  model_list = get_models()

  return JSONResponse(
    content=model_list,
    status_code=200,
  )

@app.post('/models')
async def createmodel(data: dict):
  model_ref = await data.get('model_ref')
  model_name = await data.get('model_name')
  generating_jit_model(model_ref, model_name)

  return JSONResponse(
    content={"message": "success"},
    status_code=201,
  )

@app.put('/models')
async def setmodel(data: dict):
  model_name = await data.get('model_name')
  set_model(model_name)

  return JSONResponse(
    content={"message": "success"},
    status_code=20,
  )

@app.delete('/models')
async def deletemodel(data: dict):
  model_name = await data.get('model_name')
  delete_model(model_name)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.put('/jds')
async def updatedata(file: UploadFile):
  await update_jd_data(file)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.get('/v_jds')
async def getvjds():
  pass

@app.post('/v_jds')
async def embedding(data: dict):
  name = await data.get('name')
  date = await data.get('date')
  answers = await data.get('answers')
  new_id = create_v_jd_data(name, date, answers)

  return JSONResponse(
    content={
      "message": "success",
      "id": new_id,
    },
    status_code=201,
  )

@app.get('/find_jds/{id}')
async def getjds(data: dict):
  pass
