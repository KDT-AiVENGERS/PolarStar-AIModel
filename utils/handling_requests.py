import torch

import pandas as pd
import json

def get_recommended_jds(id, columns, start, end):
    target_columns = json.loads(columns)

    jds_pd = pd.read_csv('data/jd_data.csv')

    vec_origin = torch.load('data/jd_embeddings.pth')
    vec_mock = torch.load('data/v_jd_embeddings.pth').get(id)

    if vec_mock is not None:
        recommends = find_matched_jds(vec_mock.squeeze(0), vec_origin, start, end, target_columns)
        recommends_data = list(map(lambda x: jds_pd.loc[x].to_dict(), recommends.tolist()))

    return recommends_data

def find_matched_jds(vec_mock, vec_origin, start, end, target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']) -> list:
    result = []
    col_dic = {'자격요건':0,'우대조건':1,'복지':2,'회사소개':3,'주요업무':4}
    col_list = [col_dic.get(col) for col in target_columns if col in col_dic]

    mocks = torch.chunk(vec_mock, chunks=5, dim=0)
    origins = torch.chunk(vec_origin, chunks=5, dim=1)

    for col_idx in col_list:
        result.append(torch.nn.functional.cosine_similarity(mocks[col_idx].unsqueeze(0),origins[col_idx], dim=1))

    return (sum(result)/len(col_list)).argsort()[start:end]

def get_recommended_lectures(id, start, end):
    jd_pd = pd.read_csv('data/jd_data.csv').loc[id]
    udemy_pd = pd.read_csv('data/udemy_data.csv')
    keyword_jds_pd = pd.read_csv('data/keywords_jds.csv')
    keyword_udemy_pd = pd.read_csv('data/keywords_udemy.csv')

    index_score_dict = find_matched_lectures(jd_pd, udemy_pd, keyword_jds_pd, keyword_udemy_pd)
    index_sorted = list(sorted(index_score_dict.keys(), lambda x: index_score_dict[x]))

    recommends_data = list(map(lambda x: udemy_pd.loc[x].to_dict(), index_sorted[start:end]))
    return recommends_data

def find_matched_lectures(jd_pd, udemy_pd, keyword_jds_pd, keyword_udemy_pd):
    pass