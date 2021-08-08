#!/home/dev3/anaconda3/envs/Rec_SH/bin/python3.7
# -*-coding:utf-8-*-
import math
import datetime
import sys
import time
import argparse
import requests
from tqdm import tqdm
import re
import os
from collections import defaultdict

import pandas as pd
import pymysql

os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "2"  # 'Value 'sm_86' is not defined for option 'gpu-name' 경고 무시
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 0,1 조회되게 함

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 사용할때만 메모리 할당

# config.gpu_options.per_process_gpu_memory_fraction = 0.5 #미리 50퍼센트의메모리 할당


import tensorflow.keras.layers as layers
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model


sys.path.append("../../utils")
from hwpmath2latex import hwp_parser
from konlpy_utils import komoran_using_dic, customize_komoran_model
from find_cut import find_filename


'''
DB로부터 필요한 데이터를 가져와 dataframe형태로 돌려줌.
Input : ID or DateTime (d1은 시작날짜, d2는 끝 날짜. ex. d1 = 20210405, d2 = 20210406 이면 update_datetime이 2021년 4/5 - 4/6까지의 문제 데이터를 가져옴
'''
def get_all_info(prob_db, ID= None, d1=None, d2=None):

    # 기준이 날짜가 될 수도 있고 아이디가 될 수도 있음

    curs = prob_db.cursor(pymysql.cursors.DictCursor)  # to make a dataframe

    if d2:

        sql = f"""
               select p.id, p.problem_concept_id, p.level, p.url, p.type from problem p join problem_curriculum_concept_cache pccc
               on p.problem_concept_id = pccc.relation_id
               and pccc.relation_type = 'CONCEPT'
               where pccc.revision_id = 2 and p.update_datetime >= str_to_date({d1},'%Y%m%d') and p.update_datetime < str_to_date({d2},'%Y%m%d')
               """

        print(f"{d1}부터 {d2}사이의 문제들을 인덱싱합니다.")
    elif d1:

        sql = f"""
              select p.id, p.problem_concept_id, p.level, p.url, p.type from problem p join problem_curriculum_concept_cache pccc
              on p.problem_concept_id = pccc.relation_id
              and pccc.relation_type = 'CONCEPT'
              where pccc.revision_id = 2 and p.update_datetime >= str_to_date({d1},'%Y%m%d')
              """

        print(f"{d1}부터의 문제들을 인덱싱합니다.")

    elif ID:
        sql = f"""
                  select p.id, p.problem_concept_id, p.level, p.url, p.type from problem p join problem_curriculum_concept_cache pccc
                  on p.problem_concept_id = pccc.relation_id
                  and pccc.relation_type = 'CONCEPT'
                  where pccc.revision_id = 2 and p.id > {ID}
                  """

        print(f"ID {ID}부터의 문제들을 인덱싱합니다.")

    else:

        sql = """
            select p.id, p.problem_concept_id, p.level, p.url, p.type from problem p join problem_curriculum_concept_cache pccc
            on p.problem_concept_id = pccc.relation_id
            and pccc.relation_type = 'CONCEPT'
            where pccc.revision_id = 2
            """

        # ans = input("모든 15개정 문제를 인덱싱 하겠습니까? [y/n]")
        # if ans!="y": return
        print("모든 15개정 문제를 인덱싱합니다.")
    curs.execute(sql)
    df = pd.DataFrame(curs.fetchall())

    print(f"{df.shape[0]}개의 데이터가 처리될 예정입니다.")
    return df


'''
Pretrained-MobileNetv2에 넣기위해 이미지를 전처리 하는 함수. 
Input : requests를 이용하여 url로부터 읽어들인 content, input_shape
Output : Image

'''

def preprocess_from_url(content, input_shape):
    """이미지 resize"""

    img = tf.io.decode_png(content, channels=3, name="jpeg_reader")
    img = tf.image.resize(img, input_shape[:2])
    img = preprocess_input(img)

    return img

'''
ES에 batch size만큼 데이터를 넣는 함수.(하나씩 넣는 것 보다 bulk로 여러개씩 넣는게 더 빨라서 배치별로 넣음.)
Input : 엘라스틱서치 커넥션, 문제 데이터를 가지고 있는 데이터프레임, 엘라스틱서치 인덱스 이름, model(mobilenetv2), 이미지 input_shape
Output : None

'''
def bulk_batchwise(es, part_df, INDEX_NAME, model, input_shape):
    """batch별로 데이터 elasticsearch에 넣음"""

    batch_size = 100

    part_df.set_index("id", inplace=True)

    dic = defaultdict(
        list
    )

    img_list = []

    for prob_id in list(part_df.index):

        p_img_url = f"***"

        p_hwp_url = f"***"

        s_img_url = f"***"

        s_hwp_url = f"***"

        cut = find_filename(p_hwp_url)



        try:
            img_res = requests.get(p_img_url)  # png

            try:
                # t = time.time()
                p_txt = hwp_parser(p_hwp_url)  # problem text
                # print("txt parsing : ", time.time()-t, "s")
                s_txt = hwp_parser(s_hwp_url)  # solution text

            except:
                pass

            if p_txt.startswith("Server") or s_txt.startswith("Server"):
                pass

            p_txt = p_txt.strip()  # 공백 제거
            s_txt = s_txt.strip()  # 공백 제거


            img_list.append(preprocess_from_url(img_res.content, input_shape))


            dic[prob_id].extend(
                [
                    part_df.loc[prob_id, "problem_concept_id"],
                    part_df.loc[prob_id, "level"],
                    part_df.loc[prob_id, "type"],
                    p_txt,
                    s_txt,
                    cut,
                    p_img_url,
                    s_img_url,
                ]
            )

            p_txt = None  # excessive memory usage 방지
            s_txt = None

        # {prob_id : [problem_concept_id, unitCode, level, type, p_text, s_text, cut, p_url, s_url]} 형태 dictionary

        except:  # png가 존재하지 않으면
            print(
                f"ID : {prob_id} 의 url {p_img_url}/{p_hwp_url}/{s_hwp_url}이 유효하지 않습니다."
            )
            pass

    list_ds = tf.data.Dataset.from_tensor_slices(img_list)

    dataset = list_ds.batch(batch_size).prefetch(-1)

    for batch in dataset:
        fvecs = model.predict(batch)

    dataset = None  # excessive memory usage 방지

    id_list = list(dic.keys())

    if len(list(id_list)) < 1:
        return

    bulk(
        es,
        [
            {
                "_index": INDEX_NAME,
                "_id": id_list[i],
                "image_vec": list(normalize(fvecs[i : i + 1])[0].tolist()),
                "problem_concept_id": dic[id_list[i]][0],
                "level": dic[id_list[i]][1],
                "type": dic[id_list[i]][2],
                "p_text": dic[id_list[i]][3],
                "s_text": dic[id_list[i]][4],
                "cut": dic[id_list[i]][5],
                "p_url": dic[id_list[i]][6],
                "s_url": dic[id_list[i]][7],
            }
            for i in range(len(id_list))
        ],
    )



    return


'''
bs(batch_size)만큼 엘라스틱서치에 데이터를 넣는 함수
Input : 문제 데이터 전체 데이터 프레임, 인덱스 매핑 파일(인덱스 새로 만들 때 필요), 엘라스틱서치 인덱스 이름
Output : None
'''
def bulk_all(df, INDEX_FILE, INDEX_NAME):

    bs = 10

    es = Elasticsearch(
        hosts="http://****9200",
        http_auth=("elastic", "***"),
        timeout=100,
        max_retries=10,
        retry_on_timeout=True,
    )

    # Index 생성
    #  es.indices.delete(index=INDEX_NAME, ignore=[404])  # Delete if already exists

    # mappings 정의

    # with open(INDEX_FILE) as index_file:
    #     source = index_file.read().strip()
    # es.indices.create(index=INDEX_NAME, body=source)  # Create ES index

    # print("Elasticsearch Index :", INDEX_NAME, "created!")

    nloop = math.ceil(df.shape[0] / bs)

    input_shape = (224, 224, 3)
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    model = Model(
        inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output)
    )

    #komoran = customize_komoran_model("../../utils/komoran_dict5.tsv")

    for k in tqdm(range(nloop)):
        bulk_batchwise(
            es,
            df.loc[k * bs : min((k + 1) * bs, df.shape[0])],
            INDEX_NAME,
            model,
            input_shape,
        )

    es.indices.refresh(index=INDEX_NAME)
    print(es.cat.indices(v=True))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", dest="d1", default=None)
    parser.add_argument("--d2", dest="d2", default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    print(datetime.datetime.now())
    args = parse_arguments()

    prob_db = pymysql.connect(
            user="***",
            passwd="***",
            host="***",
            db="mathflat",
            charset="utf8",
        )


    df = get_all_info(prob_db, ID = 553000, d1=None, d2=None)


    INDEX_FILE = "es_indexing.json"

    INDEX_NAME = "mathflat2"

    bulk_start = time.time()

    bulk_all(df, INDEX_FILE, INDEX_NAME)

    print(f"총 데이터 {df.shape[0]}개 bulk 소요시간은 {time.time() - bulk_start}")
    print("Success!")
