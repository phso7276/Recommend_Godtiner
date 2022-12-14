from flask import Flask, request, jsonify
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
api = Api(app)


def countVector(routines):
    # CountVectorizer
    count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
    theme_mat = count_vect.fit_transform(routines['tags'])
    from sklearn.metrics.pairwise import cosine_similarity

    theme_sim = cosine_similarity(theme_mat, theme_mat)
    # print(theme_sim.shape)
    theme_sim[:1]

    theme_sim_sorted_ind = theme_sim.argsort()[:, ::-1]
    # print(theme_sim_sorted_ind[:1])


    return theme_sim_sorted_ind


def find_sim_theme(df, sorted_ind, title_name, top_n=10):
    # 인자로 입력된 movies_df DataFrame에서 'title' 칼럼이 입력된 title_name 값인 DataFrame 추출
    routine_title = df[df['Id'] == title_name]

    # title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n개의 index 추출
    title_index = routine_title.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

    # 추출된 top_n index 출력. top_n index는 2차원 데이터임.
    # DataFrame 에서 index로 사용하기 위해서 1차원 array로 변경

    similar_indexes = similar_indexes.reshape(-1)
    print(similar_indexes)
    return similar_indexes




@api.route("/hello")
class HelloWorld(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"hello": "world!"}


#cb = Flask(__name__)


@api.route("/cb/<int:id>",methods=['GET'])
class cb(Resource):
    def get(self,id):
        routines = pd.read_csv('routines1.csv', delimiter=",")
        theme_sim_sorted_ind = countVector(routines)
        similar_routines = find_sim_theme(routines, theme_sim_sorted_ind, id, 2)
        # similar_routines[['title', 'avgPreference']]

        message = {
            "id": similar_routines.tolist()
        }
        #json_data = json.dumps(message)
        return jsonify(message)


if __name__ == 'main':
    app.run(host="0.0.0.0", port=5000)
