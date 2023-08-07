import torch
from collections import Counter
import pandas as pd

import os
import json


def server_initialize():
    if not os.path.isdir('models'):
        os.mkdir('models')

    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isfile('data/server_state.json'):
        with open('data/server_state.json', 'w', encoding='utf-8') as fp:
            json.dump({
                'current_model_ref': '',
                'model_refs': {},
            }, fp, indent='\t', ensure_ascii=False)

    if not os.path.isfile('data/v_jd_info.json'):
        with open('data/v_jd_info.json', 'w', encoding='utf-8') as fp:
            json.dump({}, fp, indent='\t', ensure_ascii=False)


def get_recommended_jds(id, columns, start, end):
    target_columns = json.loads(columns)

    if not os.path.isfile('data/jd_data.csv'):
        return [], '', {}

    jds_pd = pd.read_csv('data/jd_data.csv')

    vec_origin = torch.load('data/jd_embeddings.pth')
    vec_mock = torch.load('data/v_jd_embeddings.pth').get(id)

    if vec_mock is None:
        return [], '', {}

    recommends = find_matched_jds(vec_mock.squeeze(0), vec_origin, start, end, target_columns)
    recommends_data = list(map(lambda x: jds_pd.loc[x].to_dict(), recommends.tolist()))

    with open('data/tech_stack.json', 'r') as fp:
        tech_stack = json.load(fp)

    most_frequent_job, keyword_counts = statistics_extracting(recommends_data, tech_stack)

    return recommends_data, most_frequent_job, keyword_counts


def find_matched_jds(vec_mock, vec_origin, start, end, target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']) -> list:
    result_df = pd.DataFrame()
    col_dic = {'자격요건':0,'우대조건':1,'복지':2,'회사소개':3,'주요업무':4}
    col_list = [col_dic.get(col) for col in target_columns if col in col_dic]

    mocks = torch.chunk(vec_mock, chunks=5, dim=0)
    origins = torch.chunk(vec_origin, chunks=5, dim=1)

    for col_idx in col_list:
        if col_idx == 2  or col_idx == 3:
            result_df[col_idx] = pd.Series(torch.nn.functional.cosine_similarity(mocks[col_idx].unsqueeze(0), origins[col_idx], dim=1).detach() / 2)
        elif col_idx ==0 or col_idx ==1:
            result_df[col_idx] = pd.Series(torch.nn.functional.cosine_similarity(mocks[col_idx].unsqueeze(0), origins[col_idx], dim=1).detach() * 2)
        else:
            result_df[col_idx] = pd.Series(torch.nn.functional.cosine_similarity(mocks[col_idx].unsqueeze(0), origins[col_idx], dim=1).detach() * 1.5)

    sorted_idx = result_df.mean(axis=1).sort_values(ascending=False)

    return sorted_idx.index[start:end]


def statistics_extracting(recommends_data: list, tech_stack: dict) -> (str, dict):

    recommends_data = pd.DataFrame(recommends_data)
    tech_li = []

    recommends_data['직무내용'] = recommends_data['직무내용'].str.split(', ')
    word_frequency = recommends_data['직무내용'].explode().value_counts()
    max_frequency = word_frequency.max()
    most_common_words = ', '.join(word_frequency[word_frequency == max_frequency].index)

    for words in recommends_data['자격요건'] + recommends_data['주요업무']:
        for key, value in tech_stack.items():

            if isinstance(value, list):
                for tech in value:
                    if tech in words.lower():
                        tech_li.append(key)
                        continue

            elif value in words.lower():
                tech_li.append(key)

    most_tech_words = dict(Counter(tech_li).most_common())

    return  most_common_words, most_tech_words


def get_recommended_lectures(id, start, end):
    if not os.path.isfile('data/jd_data.csv')\
    or not os.path.isfile('data/udemy_data.csv')\
    or not os.path.isfile('data/keywords_jds.csv')\
    or not os.path.isfile('data/keywords_udemy.csv'):
        return []

    jd_pd = pd.read_csv('data/jd_data.csv').loc[id]
    udemy_pd = pd.read_csv('data/udemy_data.csv')
    keyword_jds_pd = pd.read_csv('data/keywords_jds.csv')
    keyword_udemy_pd = pd.read_csv('data/keywords_udemy.csv')

    index_score_dict = find_matched_lectures(jd_pd, udemy_pd, keyword_jds_pd, keyword_udemy_pd)
    index_sorted = list(index_score_dict.keys())

    recommends_data = list(map(lambda x: udemy_pd.loc[x].to_dict(), index_sorted[start:end]))
    return recommends_data


def find_matched_lectures(jd_pd, udemy_pd, keyword_jds_pd, keyword_udemy_pd):
    #공고의 문자열 병합 및 소문자화
    string = jd_pd['자격요건'][0] + jd_pd['우대조건'][0] + jd_pd['주요업무'][0]
    string = string.lower().replace(' ', '')

    # udemy, 공고 사이에 겹치는 키워드를 추출하는 함수
    def make_keywords(keyword_jds_pd, keyword_udemy_pd):
        keywords = pd.merge(keyword_jds_pd, keyword_udemy_pd, on = '단어명', how = 'inner')['단어명'].values

        return keywords

    keywords = make_keywords(keyword_jds_pd, keyword_udemy_pd)

    # 중복된 키워드 안에서 JD 안에 있는 keyword를 추출
    k_list = []
    for keyword in keywords:
        if keyword in string:
            k_list.append(keyword)

    # 강의별로 keyword를 순회시키면서 해당 keyword가 몇개가 들어있는지 counting
    n = len(udemy_pd)
    Rec_dict = dict()
    for idx in range(n):
        Udemy_string = udemy_pd.loc[idx, '강의소개'] + udemy_pd.loc[idx, '강의명']
        Udemy_string = Udemy_string.lower().replace(' ', '')
        count = 0

        for key in k_list:
            count += Udemy_string.count(key)

        Rec_dict[idx] = count

    Rec_dict = dict(sorted(Rec_dict.items(), key = lambda x: x[1], reverse = True))

    return Rec_dict