import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from flask import Flask, request,jsonify

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import api


def countVector(routines):
    # CountVectorizer
    count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
    theme_mat = count_vect.fit_transform(routines['Theme'])
    from sklearn.metrics.pairwise import cosine_similarity

    theme_sim = cosine_similarity(theme_mat, theme_mat)
    # print(theme_sim.shape)
    theme_sim[:1]

    theme_sim_sorted_ind = theme_sim.argsort()[:, ::-1]
    # print(theme_sim_sorted_ind[:1])
    similar_routines = find_sim_theme(routines, theme_sim_sorted_ind, '건강한 생활', 7)

    return similar_routines


def find_sim_theme(df, sorted_ind, title_name, top_n=10):
    # 인자로 입력된 movies_df DataFrame에서 'title' 칼럼이 입력된 title_name 값인 DataFrame 추출
    routine_title = df[df['title'] == title_name]

    # title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n개의 index 추출
    title_index = routine_title.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

    # 추출된 top_n index 출력. top_n index는 2차원 데이터임.
    # DataFrame 에서 index로 사용하기 위해서 1차원 array로 변경
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]


#cb = Flask(__name__)


@api.route("/cb", methods=["GET"])
def ver1():
    routines = pd.read_csv('routines1.csv', delimiter=",")
    similar_routines = countVector(routines)
    # similar_routines = find_sim_theme(routines, theme_sim_sorted_ind, '건강한 생활', 7)
    # similar_routines[['title', 'avgPreference']]

    message = {
        "name": "get요청:id 담아주기",
        "id": similar_routines[['title', 'avgPreference']]
    }
    return jsonify(message)


