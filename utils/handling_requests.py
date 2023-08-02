import torch

def find_matched_jds(v_jd, jds, k = 20, target_columns = ['자격요건', '우대조건', '복지', '회사소개', '주요업무']) -> list:
    '''
    inputs
      v_jd: 가상 jd 의 torch 임베딩 벡터 (1 x 3840)
      jds: 전체 jd 의 torch 임베딩 벡터 (n x 3840)
      target_columns: 비교 대상 항목
    output
      가상 jd 와 연관성이 높은 순서대로 k 개의 index(jds 의 index) 의 리스트
    '''

    pass