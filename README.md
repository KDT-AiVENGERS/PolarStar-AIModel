# Model Server

[북극성 통합 repository 바로가기](https://github.com/KDT-AiVENGERS/.github/tree/develop/profile/polarstar)

Model Server 저장소는 북극성 프로젝트의 인공지능 모델을 서빙하기 위한 서버입니다. 본 서버에는 공고 추천 서비스를 위한 추론용 BERT 모델과, 커리큘럼 추천 서비스를 위한 알고리즘이 구현되어 있습니다. Browser (client) 에서 사용자 질의 사항에 대한 정보가 전송되면, 이를 토대로 encoded vector 를 추론하여 저장한 후, 미리 저장되어 있던 공고들의 vector 와 비교하여 추천 공고 정보를 내보냅니다.

Model 은 torchscript 의 JIT compiler 로 변환하여 저장되며, 필요할 경우 저장해둔 torchscript 파일 중에서 골라 교체할 수 있습니다.

본 서버는 FastAPI 프레임워크로 구현되었으며, 본 프로젝트가 아직 시험 단계임을 고려하여 데이터는 별도의 DB 없이 서버의 저장소만을 활용하여 보관됩니다.

Client 에서 요청하기 위한 endpoint 이외에도, 모델과 데이터 (공고, 강의, keywords 등) 을 갱신하기 위한 API 가 구성되어 있습니다.

### Installation

터미널 앱을 통해 Git clone 한 폴더에 들어가 아래의 지시를 따릅니다.

1. Conda 등 가상환경 생성 (Python 버전 3.10.12)

   ```bash
   conda create -n model_serving python=3.10.12 -y
   ```

2. 가상환경 진입

   ```bash
   conda activate model_serving
   ```

3. 의존성 모듈 설치

   ```bash
   pip install -r requirements.txt
   ```

4. 서버 실행

   ```bash
   uvicorn main:app --reload
   ```

5. 서버 초기화
   1. 초기화 요청
      1. GET `/init` route
   2. 모델 세팅
      1. POST `/models` route
      2. PUT `/models` route
   3. 파일 업로드
      1. JD 공고 파일 업로드
         1. PUT `/jds` route
      2. UDEMY 강의 파일 업로드
         1. PUT `/udemy` route
      3. JD 키워드 파일 업로드
         1. PUT `/key_jds` route
      4. UDEMY 키워드 파일 업로드
         1. PUT `/key_udemy` route
      5. Tech_stack 키워드 파일 업로드
         1. PUT `/key_tech_stack` route

### API endpoints

1. API 테스트를 위해서는, 서버를 실행한 후 아래 주소로 접속하시기 바랍니다.
   1. Endpoint 목록 자동생성 1: `/docs` 로 접속 (예: [http://localhost:8000/docs](http://localhost:8000/docs))
   2. Endpoint 목록 자동생성 2: `/redoc` 로 접속 (예: [http://localhost:8000/redoc](http://localhost:8000/redoc))
2. Hello world!!
   1. 서버 실행 및 연결 확인 목적
      1. 요청 라우트 `/`
      2. 요청 타입 `GET`
      3. 요청 데이터: 없음
      4. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'Hello World',
         }
         ```
3. 서버 초기화
   1. 서버 초기화
      1. 초기 설정 파일 생성 및 필수 파일 유무 체크
      2. 요청 라우트 `/init`
      3. 요청 타입 `GET`
      4. 요청 데이터: 없음
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success',
         	'data': {
         		'jd_data.csv': 'ok',
         		'udemy_data.csv': 'empty',
         		...
         	},
         }
         ```
4. 임베딩 모델 관련
   1. 임베딩 모델 리스트 요청
      1. 현재 존재하는 모델의 리스트를 요청한다. 그 중 ‘current_model’ 은 inference 에 사용하는 것으로 적용된 파일이다.
      2. 요청 라우트 `/models`
      3. 요청 타입 `GET`
      4. 요청 데이터: 없음
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success',
         	'data': [
         		'model1',
         		'model2',
         		...
         	],
         }
         ```
   2. 임베딩 모델 생성 요청
      1. 설명: 허깅페이스 저장소로부터 새로운 모델을 다운로드 받고, 이를 torchscript 로 변환하여 저장한다. model_ref 는 다운로드 받을 저장소 이름이고, model_name 은 저장할 jit 파일의 이름이다.
      2. 요청 라우트 `/models`
      3. 요청 타입 `POST`
      4. 요청 데이터: dict

         ```python
         {
         	'model_ref': 'bert-base-cased',
         	'model_name': '230801_new_model',
         }
         ```

      5. 응답 데이터

         1. Response.status_code: 201

         ```python
         {
         	'message': 'success'
         }
         ```
   3. 임베딩 모델 업데이트 요청
      1. inference 로 적용하는 ‘current_model’ 을 지정된 모델로 변경한다. 지정한 모델이 서버에 없을 경우 아무런 동작도 하지 않는다.
      2. 요청 라우트 `/models`
      3. 요청 타입 `PUT`
      4. 요청 데이터: dict

         ```python
         {
         	'model_name': '230801_new_model',
         }
         ```

      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success'
         }
         ```
   4. 임베딩 모델 삭제 요청
      1. jit 모델 파일을 삭제할 수 있음. 해당 파일이 없을 경우 아무런 동작을 하지 않음.
      2. 요청 라우트 `/models`
      3. 요청 타입 `DELETE`
      4. 요청 데이터: dict

         ```python
         {
         	'model_name': '230801_new_model'
         }
         ```

      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success'
         }
         ```
5. JD 데이터베이스 관련
   1. JD 데이터베이스 업데이트 요청
      1. 공고 데이터를 csv 파일로 받아서 이를 저장하고 embedding vector 화를 진행한다. embedding vector 도 저장한다.
      2. 요청 라우트 `/jds`
      3. 요청 타입 `PUT`
      4. 요청 데이터: file
         1. csv 파일
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success'
         }
         ```
6. 가상 JD 데이터베이스 관련
   1. 가상 JD 데이터 전체 요청
      1. 저장된 전체 가상 JD 데이터를 요청한다. 응답 데이터는 id 를 key 로 가지고 데이터를 value 로 가지는 dictionary 형태로 반환된다.
      2. 요청 라우트 `/v_jds`
      3. 요청 타입 `GET`
      4. 요청 데이터: 없음
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success',
         	'data': {
         		'1995-12-17T03:24:00Z_아무개': {
         			'date': '1995-12-17T03:24:00Z',
         			'name': '아무개',
         			'answers': {
         				'personality':['꼼꼼함','책임감','적극적','주체적'],
         				'stack':['Node.js', 'Python','Pandas'],
         				'welfare':['장비','휴가','출퇴근','식사','건강검진'],
         				'job':['프론트엔드', '백엔드', 'ai'],
         				'domain':['화장품', '식품', '컨설팅', '금융']
         			}
         		},
         		...
         	}
         }
         ```
   2. 가상 JD 데이터 하나 요청
      1. 특정 가상 JD id 에 대한 데이터를 요청한다.
      2. 요청 라우트 `/v_jds/{id}`
         1. 주소창에 id 특정하여 요청
      3. 요청 타입 `GET`
      4. 요청 데이터: 없음
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success',
         	'data': {
         		'date': '1995-12-17T03:24:00Z',
         		'name': '아무개',
         		'answers': {
         			'personality':['꼼꼼함','책임감','적극적','주체적'],
         			'stack':['Node.js', 'Python','Pandas'],
         			'welfare':['장비','휴가','출퇴근','식사','건강검진'],
         			'job':['프론트엔드', '백엔드', 'ai'],
         			'domain':['화장품', '식품', '컨설팅', '금융']
         		},
         		...
         	}
         }
         ```
   3. 새 가상 JD 데이터 생성
      1. 새로운 사용자 응답 데이터를 받아 이를 저장하고, 가상 JD 를 생성한 후 embedding_vector 변환하여 저장한다. 저장되는 형식은 dictionary 로서, key 는 id 이고 value 는 embedding_vector tensor 이다.
      2. 요청 라우트 `/v_jds`
      3. 요청 타입 `POST`
      4. 요청 데이터: dict

         ```python
         {
         	'date': '1995-12-17T03:24:00Z',
         	'name': '아무개',
         	'answers': {
         		'personality':['꼼꼼함','책임감','적극적','주체적'],
         		'stack':['Node.js', 'Python','Pandas'],
         		'welfare':['장비','휴가','출퇴근','식사','건강검진'],
         		'job':['프론트엔드', '백엔드', 'ai'],
         		'domain':['화장품', '식품', '컨설팅', '금융']
         	}
         }
         ```

      5. 응답 데이터

         1. Response.status_code: 201

         ```python
         {
         	'message': 'success',
         	'id': '1995-12-17T03:24:00Z_아무개',
         }
         ```
7. Udemy 데이터베이스 관련
   1. Udemy 데이터베이스 업데이트 요청
      1. Udemy 강의 데이터를 csv 파일로 받아서 이를 저장한다.
      2. 요청 라우트 `/udemy`
      3. 요청 타입 `PUT`
      4. 요청 데이터: file
         1. csv 파일
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success'
         }
         ```
8. Keywords 데이터베이스 관련
   1. JD keywords 데이터베이스 업데이트 요청
      1. JD keywords 데이터를 csv 파일로 받아서 이를 저장한다.
      2. 요청 라우트 `/key_jds`
      3. 요청 타입 `PUT`
      4. 요청 데이터: file
         1. csv 파일
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success'
         }
         ```
   2. Udemy keywords 데이터베이스 업데이트 요청
      1. Udemy keywords 데이터를 csv 파일로 받아서 이를 저장한다.
      2. 요청 라우트 `/key_udemy`
      3. 요청 타입 `PUT`
      4. 요청 데이터: file
         1. csv 파일
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success'
         }
         ```
   3. Tech_stack keywords 데이터베이스 업데이트 요청
      1. Tech_stack keywords 데이터를 csv 파일로 받아서 이를 저장한다.
      2. 요청 라우트 `/key_tech_stack`
      3. 요청 타입 `PUT`
      4. 요청 데이터: file
         1. csv 파일
      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success'
         }
         ```
9. 추천 관련
   1. 추천 JD 리스트 요청
      1. 유사도 계산을 위해 조합할 columns 정보와 유사도 순위 정보 (start, end) 를 받아서, 해당 순위에 해당하는 공고 내용을 응답으로 내보낸다. 그리고 가장 빈도가 높은 직무 하나와, 주요 키워드 별 카운트 수를 함께 응답으로 내보낸다.
      2. 요청 라우트 `/find_jds/{v_jd_id}`
         1. 주소창에 가상 jd 의 id 를 특정하여 요청
      3. 요청 타입 `GET`
      4. 요청 데이터: params (columns 는 json string 화 하여 보낸다)

         ```python
         {
         	'columns': [
         		'자격요건', '우대사항'
         	],
         	'start': 0,
         	'end': 100,
         }

         ```

      5. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success',
         	'data': {
         		'jds': [
         			{
         				'회사명': '~~~~',
         				...
         			},
         			...
         		],
         		'most_frequent_job': '데이터 엔지니어',
         		'keyword_counts': {
         			'react': 3,
         			...
         		},
         }
         ```
   2. 추천 커리큘럼 리스트 요청
      1. 목표 JD 의 키워드 특성과 가장 잘 부합하는 강의들을 찾아 그 강의의 정보를 제시함
      2. 요청 라우트 `/find_lectures/{jd_id}`
         1. 주소창에 JD 의 id 를 특정하여 요청
      3. 요청 데이터: params

         ```python
         {
         	'start': 0,
         	'end': 100,
         }
         ```

      4. 응답 데이터

         1. Response.status_code: 200

         ```python
         {
         	'message': 'success',
         	'data': [
         		{
         			'대분류': '~~~~',
         			...
         		},
         		...
         	]
         }
         ```
