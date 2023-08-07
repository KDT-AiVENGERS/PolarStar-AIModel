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
  jd_dic = {'자격요건':'','우대조건':'','복지':'수습기간 급여지급\n수습기간 적용\n','회사소개':'','주요업무':''}

  for question, answer in QnA_answer.items():

        match question:
            case 'personality':
                for i in answer:
                        jd_dic['자격요건'] += (i + '능력을 가지고 계신분\n'
                                            )

                        jd_dic['우대조건'] += (i + '능력을 가지고 계신분\n'
                                            )

                        jd_dic['회사소개'] += (i + '문화를 가지고 있는 회사 입니다.\n'
                                           )

            case 'stack':
                if len(answer) == 0:
                    pass

                else:
                    for i in answer:
                        jd_dic['자격요건'] += (i + '개발 경험이 있으신 분\n'
                                            + i + '개발 경험\n'
                                            + i + '사용이 가능하신 분\n'
                                            + i + '에대한 이해\n'
                                            + i + '에 관심이 있으신 분'
                                            + '기술스택: ' + ','.join(i)
                                            )

                        jd_dic['우대조건'] += (i + '사용에 능숙하신 분\n'
                                           + i + '개발 경험이 있으신 분\n'
                                           + i + '개발 경험'
                                           )

            #default '수습' in welfare
            case 'welfare':
                if '수습' not in answer:
                    jd_dic['복지'] = ''

                if '장비' in answer:
                    jd_dic['복지'] += '필요한 장비지원\n장비 구매 지원\n'

                if '휴가' in answer:
                     jd_dic['복지'] += '휴가 지원\n연차\n'

                if '출퇴근' in answer:
                     jd_dic['복지'] += '유연근무제를 시행중입니다.\n재택근무를 시행중\n'

                if '식사' in answer:
                     jd_dic['복지'] += '구내식당, 사내식당\n중/석식비 제공, 식사비 제공\n'

                if '건강검진' in answer:
                     jd_dic['복지'] += '건강검진 지원\n '

                if '인센티브' in answer:
                     jd_dic['복지'] += '인센티브 제공\n스톡옵션 제공\n성과에 따른 보상 제공\n'

                if '계발' in answer:
                     jd_dic['복지'] += '온라인강의 제공\n 도서, 시험지 제공\n자기계발비 지원'


            case 'job':
                if len(answer) == 0:
                    pass

                for i in answer:
                        jd_dic['자격요건'] += (i + '관련 프로젝트 경험이 있는분\n'
                                            + i + '에 대한 지식을 보유하고 있는 분\n'
                                            + i + '관련 지식 보유\n'
                                            )

                        jd_dic['주요업무'] += (i + '관련 연구 및 개발\n'
                                           )

                        jd_dic['우대조건'] += (i + '관련 프로젝트 경험이 있는분\n'
                                            + i + '에 대한 지식을 보유하고 있는 분\n'
                                            + i + '관련 지식 보유\n'
                                           )


            case 'domain':

                for i in answer:
                    jd_dic['회사소개'] += (i + '관련 프로젝트 경험이 있는분\n'
                                        + i + '에 대한 지식을 보유하고 있는 분\n'
                                        + i + '관련 지식 보유\n'
                                        )

                    jd_dic['우대조건'] += (i + '에 대한 지식을 보유하고 있는 분\n'
                                        + i + '관련 지식 보유\n'
                                        )

                    jd_dic['주요업무'] += (i + '관련 연구 및 개발\n'
                                           )

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
