import torch

def find_matched_jds(vec_mock, vec_origin, start, end, target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']) -> list:
    
    result = []
    col_dic = {'자격요건':0,'우대조건':1,'복지':2,'회사소개':3,'주요업무':4}
    col_list = [col_dic.get(col) for col in target_columns]

    mocks = torch.chunk(vec_mock, chunks=5, dim=0)
    origins = torch.chunk(vec_origin, chunks=5, dim=1)
    
    for col_idx in col_list:
        result.append(torch.nn.functional.cosine_similarity(mocks[col_idx].unsqueeze(0),origins[col_idx], dim=1))

    return (sum(result)/3).argsort()[start:end]