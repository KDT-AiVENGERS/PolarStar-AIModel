from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from utils.handling_models import get_models, generating_jit_model, set_model, delete_model
from utils.handling_data import update_jd_data, get_v_jd_data, get_v_jds_data, create_v_jd_data, update_udemy_data, update_keywords_jds, update_keywords_udemy
from utils.handling_requests import server_initialize, get_recommended_jds, get_recommended_lectures

tags_metadata = [
    {
        "name": "Embedding Model",
        "description": "모델 관련 API",
    },
    {
        "name": "JD Database",
        "description": "JD 데이터베이스 관련 API",
    },
    {
        "name": "Virture JD Database",
        "description": "가상 JD 데이터베이스 관련 API",
    },
    {
        "name": "Udemy Database",
        "description": "Udemy 데이터베이스 관련 API",
    },
    {
        "name": "Keywords Database",
        "description": "Keywords 데이터베이스 관련 API",
    },
    {
        "name": "Recommendation",
        "description": "추천 기능 관련 API",
    },
]

app = FastAPI(
  title="북극성",
  summary="길을 잃지 않도록 환히 빛나고 있습니다.",
  version="1.0.0",
  openapi_tags=tags_metadata
)

@app.get('/')
async def helloworld():
  return JSONResponse(
    content={"message": "Hello World"},
    status_code=200,
  )

@app.get('/init')
def initialize():
  server_initialize()

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.get('/models', tags=["Embedding Model"])
def getmodels():
  model_list = get_models()

  return JSONResponse(
    content={
      "message": "success",
      "data": model_list,
    },
    status_code=200,
  )

@app.post('/models', tags=["Embedding Model"])
def createmodel(data: dict):
  model_ref = data.get('model_ref')
  model_name = data.get('model_name')
  generating_jit_model(model_ref, model_name)

  return JSONResponse(
    content={"message": "success"},
    status_code=201,
  )

@app.put('/models', tags=["Embedding Model"])
def setmodel(data: dict):
  model_name = data.get('model_name')
  set_model(model_name)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.delete('/models', tags=["Embedding Model"])
def deletemodel(data: dict):
  model_name = data.get('model_name')
  delete_model(model_name)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.put('/jds', tags=["JD Database"])
async def updatedata(file: UploadFile):
  await update_jd_data(file)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.get('/v_jds', tags=["Virture JD Database"])
def getvjds():
  data = get_v_jds_data()

  return JSONResponse(
    content={
      "message": "success",
      "data": data,
    },
    status_code=200,
  )

@app.get('/v_jds/{id}', tags=["Virture JD Database"])
def getvjd(id: str):
  data = get_v_jd_data(id)

  return JSONResponse(
    content={
      "message": "success",
      "data": data,
    },
    status_code=200,
  )

@app.post('/v_jds', tags=["Virture JD Database"])
def embedding(data: dict):
  name = data.get('name')
  date = data.get('date')
  answers = data.get('answers')
  new_id = create_v_jd_data(name, date, answers)

  return JSONResponse(
    content={
      "message": "success",
      "id": new_id,
    },
    status_code=201,
  )

@app.put('/udemy', tags=["Udemy Database"])
async def updateudemydata(file: UploadFile):
  await update_udemy_data(file)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.put('/key_jds', tags=["Keywords Database"])
async def updatekeywordsjds(file: UploadFile):
  await update_keywords_jds(file)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.put('/key_udemy', tags=["Keywords Database"])
async def updatekeywordsudemy(file: UploadFile):
  await update_keywords_udemy(file)

  return JSONResponse(
    content={"message": "success"},
    status_code=200,
  )

@app.get('/find_jds/{v_jd_id}', tags=["Recommendation"])
def get_rec_jds(v_jd_id: str, columns: str, start: int, end: int):
  recommends_data, most_frequent_job, keyword_counts = get_recommended_jds(v_jd_id, columns, start, end)
  return JSONResponse(
    content={
      "message": "success",
      "data": {
        "jds": recommends_data,
        "most_frequent_job": most_frequent_job,
        "keyword_counts": keyword_counts,
      },
    },
    status_code=200,
  )

@app.get('/find_lectures/{jd_id}', tags=["Recommendation"])
def get_rec_lectures(jd_id: int, start: int, end: int):
  recommends_data = get_recommended_lectures(jd_id, start, end)
  return JSONResponse(
    content={
      "message": "success",
      "data": recommends_data,
    },
    status_code=200,
  )
