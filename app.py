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

routines = pd.read_csv('routines1.csv', delimiter=",")


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


percentile = 0.2
m = np.quantile(routines['count'], percentile)
c = np.mean(routines['avgPreference'])


def weighted_avgPreference(record):
    v = record['count']
    r = record['avgPreference']
    weighted = ((v / (v + m)) * r) + ((m / (m + v)) * c)

    return weighted


def find_sim_routine(df, sorted_ind, title_name, top_n=10):
    routine_title = df[df['Id'] == title_name]
    title_index = routine_title.index.values

    # top_n에 해당하는 태그 유사성이 높은 인덱스 추출
    similar_indexes = sorted_ind[title_index, :(top_n)]
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 루틴 인덱스는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]

    #print(df.iloc[similar_indexes].sort_values('weighted_avg', ascending=False)[:top_n])

    temp = df.iloc[similar_indexes].sort_values('weighted_avg', ascending=False)[:top_n]
    temp_ids = temp['Id'].values.tolist()
    #print(temp_ids)
    #print(similar_indexes)

    return temp_ids


@api.route("/cb/<int:id>", methods=['GET'])
class cb(Resource):
    def get(self, id):
        theme_sim_sorted_ind = countVector(routines)

        routines['weighted_avg'] = routines.apply(weighted_avgPreference, axis=1)

        similar_routines = find_sim_routine(routines.sort_values('weighted_avg', ascending=False), theme_sim_sorted_ind,
                                            id, 3)

        message = {
            "id": similar_routines
        }
        # json_data = json.dumps(message)
        return jsonify(message)


if __name__ == 'main':
    app.run(host="0.0.0.0", port=5000)
