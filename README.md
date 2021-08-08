# Recommender-System

MATHFLAT's Algorithm for Recommendation of Similar Problems.
Extract the image feature vectors using a pre-trained MobileNetV2 and measure cosine similarity.
Parse the texts from .hwp files.
Use Elasticsearch to get top-k questions with high similarity.

 

## server_test


### 1. indexing
1) put_server_mathText_img.py : Input datetime or ID -> indexing data into ES


### Environment setup

  1. `conda create -n RecoSys_server python=3.7` : create a conda environment 
  2. `source activate RecoSys_server` : activate the conda environment
  3. `pip install -r requirements.txt` : install python packages



### Run

  1. put_server_mathText_img.py
  ```
  $ conda activate ${CONDA_VIRTUAL_ENV}
  $ python server_test/indexing/put_server_mathText_img.py 
  ```


  
## utils


  1. general_utils.py : Collection of image-related functions
  
  2. hwpmath2latex.py : hwp 파서 using hwp.jar
  
  3. hwp_handler.py : methods used with hwp files
  
  4. konlpy_utils.py : konlpy 관련 methods 저장.
  
  5. komoran_dict.tsv : konlpy중 komoran을 커스터마이즈하기 위한 수학 용어 사전.
  
  6. all_15_hwp_tokenized.csv : 모든 15년도개정 문제 text, komoran을 사용하여 tokenized text

  7. hwp.jar : hwp파싱에 쓰이는 jar파일 (https://github.com/neolord0/hwplib)

  8. hwp_parser.py : Extracts plain text or images from problem data(not used)

  9. hml_equation_parser dir : methods for converting hwp math expression to LaTEX
  
  10. pypdf2.py : pdf utils

  11, latex2img.py : methods to make LaTEX to Image file(수식 인식 training을 위한 데이터 생성을 위해 만들었으나 사용하지 않음.)
  

설명되어 있지 않은 파일은 사용하지 않는 파일.