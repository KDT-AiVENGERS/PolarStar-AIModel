import torch

import lightning.pytorch as pl

from transformers import AutoTokenizer

import pandas as pd
import json
import os

from infra.data_module import HFBertDataModule
from infra.task_module import HFBertTask

async def update_jd_data(file):
  content = await file.read()
  with open('data/jd_data.csv', 'wb') as fp:
    fp.write(content)

  jd_data_pd = pd.read_csv('data/jd_data.csv')
  jd_data_pd.to_csv('data/jd_data.csv', index=True, index_label='jd_id')

  new_embedding = embedding_jds()
  torch.save(new_embedding, 'data/jd_embeddings.pth')

def update_jd_manual():
  new_embedding = embedding_jds()
  torch.save(new_embedding, 'data/jd_embeddings.pth')

def embedding_jds(target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']):
  with open('data/server_state.json', 'r') as fp:
    model_ref = json.load(fp)['current_model_ref']

  tokenizer = AutoTokenizer.from_pretrained(model_ref)
  data_module = HFBertDataModule(
    tokenizer=tokenizer,
    max_batch_size=60,
    data_path='data/jd_data.csv',
    predict_target_cols=target_columns,
  )

  model = torch.jit.load('models/current_model.pt')
  task = HFBertTask(tokenizer=tokenizer, predict_model=model, predict_target_cols=target_columns)

  trainer = pl.Trainer(
    logger=False
  )

  predicted_embedding_vectors = trainer.predict(task, datamodule=data_module)
  concatenated_embedding_vectors = torch.concat(predicted_embedding_vectors, dim=-2)

  return concatenated_embedding_vectors

def get_v_jds_data():
  with open('data/v_jd_info.json', 'r') as fp:
    v_jds = json.load(fp)

  return v_jds

def get_v_jd_data(v_jd_id):
  with open('data/v_jd_info.json', 'r') as fp:
    v_jds = json.load(fp)

  return v_jds.get(v_jd_id)

def create_v_jd_data(name, date, answers):
  new_id = date + '_' + name
  new_value = {
    'date': date,
    'name': name,
    'answers': answers
  }

  if os.path.isfile('data/v_jd_info.json'):
    with open('data/v_jd_info.json', 'r') as fp:
      v_jds = json.load(fp)
      v_jds[new_id] = new_value
  else:
    v_jds = { new_id: new_value }

  v_jd_generated = generate_v_jd(answers)

  v_jd_embedding_calculated = embedding_v_jds(v_jd_generated)

  if os.path.isfile('data/v_jd_embeddings.pth'):
    v_jd_embedding = torch.load('data/v_jd_embeddings.pth')
  else:
    v_jd_embedding = {}
  v_jd_embedding[new_id] = v_jd_embedding_calculated

  torch.save(v_jd_embedding, 'data/v_jd_embeddings.pth')
  with open('data/v_jd_info.json', 'w', encoding='utf-8') as fp:
    json.dump(v_jds, fp, indent="\t", ensure_ascii=False)

  return new_id

def embedding_v_jds(v_jd_dict, target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']):
  with open('data/server_state.json', 'r') as fp:
    model_ref = json.load(fp)['current_model_ref']

  tokenizer = AutoTokenizer.from_pretrained(model_ref)
  model = torch.jit.load('models/current_model.pt')

  input_sentences = list(map(lambda x: v_jd_dict[x], target_columns))

  input_tokens = tokenizer(input_sentences, return_tensors='pt', padding=True, truncation=True)

  outputs = model(**input_tokens)
  output_pooler = outputs[1]

  return output_pooler.view((1, -1))

def generate_v_jd(QnA_answer: dict):
  jd_dic = {'자격요건':'','우대조건':'','복지':'수습','회사소개':'','주요업무':''}

  for key, value in QnA_answer.items():
    question = key
    answer = ' '.join(value)

    match question:
      case 'personality':
        jd_dic['자격요건'] = jd_dic['자격요건'] + answer +' '
        jd_dic['회사소개'] = jd_dic['회사소개'] + answer +' '

      case 'stack':
        if len(answer) == 0:
          pass

        else:
          jd_dic['자격요건'] = jd_dic['자격요건'] + answer +' '
          jd_dic['우대조건'] = jd_dic['우대조건'] + answer +' '

      #default '수습' in welfare
      case 'welfare':
        if '수습' not in answer:
          jd_dic['복지'] = answer
        else:
          jd_dic['복지'] = jd_dic['복지'] + answer +' '

      case 'job':
        jd_dic['주요업무'] = jd_dic['주요업무'] + answer +' '


      case 'domain':
        jd_dic['회사소개'] = jd_dic['회사소개'] + answer +' '

  return jd_dic

async def update_udemy_data(file):
  content = await file.read()
  with open('data/udemy_data.csv', 'wb') as fp:
    fp.write(content)

async def update_keywords_jds(file):
  content = await file.read()
  with open('data/keywords_jds.csv', 'wb') as fp:
    fp.write(content)

async def update_keywords_udemy(file):
  content = await file.read()
  with open('data/keywords_udemy.csv', 'wb') as fp:
    fp.write(content)
