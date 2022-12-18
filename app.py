from flask import Flask, request, jsonify
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
api = Api(app)

routines = pd.read_csv('routines1.csv', delimiter=",")
ratings = pd.read_csv('rating1.csv', delimiter=",")


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

    # print(df.iloc[similar_indexes].sort_values('weighted_avg', ascending=False)[:top_n])

    temp = df.iloc[similar_indexes].sort_values('weighted_avg', ascending=False)[:top_n]
    temp_ids = temp['Id'].values.tolist()
    print(temp_ids)
    # print(similar_indexes)

    return temp_ids


# 아이템 협업
def data_cleansing(routines, ratings):
    # ratings 데이터와 routine 데이터 결합
    rating_routines = pd.merge(ratings, routines, on="sharedRoutineId")

    # 사용자-아이템 평점 행렬 생성
    ratings_matrix = rating_routines.pivot_table("preference", "memberId", "sharedRoutineId")

    # NaN값은 0으로 변환
    ratings_matrix.fillna(0, inplace=True)


    return ratings_matrix


# 인수로 사용자-아이템 평점 행렬(NaN은 현재 0으로 대체), 아이템 유사도 행렬 사용
def predict_rating(ratings_arr, item_sim_arr):
    # ratings_arr: u x i, item_sim_arr: i x i
    sum_sr = ratings_arr @ item_sim_arr
    sum_s_abs = np.array([np.abs(item_sim_arr).sum(axis=1)])

    ratings_pred = sum_sr / sum_s_abs

    return ratings_pred


def predict_rating_topsim(ratings_arr, item_sim_arr, N=4):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 열 크기(아이템 수)만큼 반복 (row: 사용자, col: 아이템)
    for col in range(ratings_arr.shape[1]):

        # 특정 아이템의 유사도 행렬 오름차순 정렬시 index .. (1)
        temp = np.argsort(item_sim_arr[:, col])

        # (1)의 index를 역순으로 나열시 상위 N개의 index = 특정 아이템의 유사도 상위 N개 아이템 index .. (2)
        top_n_items = [temp[:-1 - N:-1]]

        # 개인화된 예측 평점을 계산: 반복당 특정 아이템의 예측 평점(사용자 전체)
        for row in range(ratings_arr.shape[0]):
            # (2)의 유사도 행렬
            item_sim_arr_topN = item_sim_arr[col, :][top_n_items].T  # N x 1

            # (2)의 실제 평점 행렬
            ratings_arr_topN = ratings_arr[row, :][top_n_items]  # 1 x N

            # 예측 평점
            pred[row, col] = ratings_arr_topN @ item_sim_arr_topN
            pred[row, col] /= np.sum(np.abs(item_sim_arr_topN))

    return pred


# 아직 보지 않은 루틴 리스트 함수
def get_unseen_routines(ratings_matrix, userId):
    # user_rating: userId의 아이템 평점 정보 (시리즈 형태: title을 index로 가진다.)
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating=0인 아직 안본 루틴
    unseen_routine_list = user_rating[user_rating == 0].index.tolist()

    # 모든 영화명을 list 객체로 만듬.
    routines_list = ratings_matrix.columns.tolist()

    # 한줄 for + if문으로 안본 영화 리스트 생성
    unseen_list = [routine for routine in routines_list if routine in unseen_routine_list]

    return unseen_list


# 보지 않은 루틴 중 예측 높은 순서로 시리즈 반환
def recomm_routine_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_routines = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]

    return recomm_routines


def make_matrix(ratings_matrix_T, ratings_matrix):
    # 아이템 유사도 행렬
    item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

    # 데이터 프레임 형태로 저장
    item_sim_df = pd.DataFrame(item_sim, index=ratings_matrix_T.index, columns=ratings_matrix_T.index)

    ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, N=4)

    return ratings_pred


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




@api.route("/recommend/<int:id>", methods=['GET'])
class recommend(Resource):
    def get(self, id):
        # 아이템-사용자 평점 행렬로 전치
        ratings_matrix = data_cleansing(routines,ratings)
        ratings_matrix_T = ratings_matrix.T

        ratings_pred =make_matrix(ratings_matrix_T,ratings_matrix)

        # 예측 평점 데이터 프레임
        ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=ratings_matrix.index,
                                           columns=ratings_matrix.columns)

        # 아직 보지 않은 영화 리스트
        unseen_list = get_unseen_routines(ratings_matrix, id)

        # 아이템 기반의 최근접 이웃 협업 필터링으로 영화 추천
        recomm_routines = recomm_routine_by_userid(ratings_pred_matrix, id, unseen_list, top_n=4)

        # 데이터 프레임 생성
        #recomm_routines = pd.DataFrame(data=recomm_routines.values, index=recomm_routines.index, columns=['pred_score'])
        print(recomm_routines.index.tolist())
        ids=recomm_routines.index.tolist()

        #temp_ids = recomm_routines['sharedRoutineId'].values.tolist()

        message = {
            "id": ids
        }
        # json_data = json.dumps(message)
        return jsonify(message)


if __name__ == 'main':
    app.run(host="0.0.0.0", port=5000)
