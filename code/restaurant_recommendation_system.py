from __future__ import division
import pandas as pd
import numpy as np
import itertools, numba

class RestaurantRecommender(object):

def __init__(self, data_path):
    self.processed_data = self._load_data(data_path)
    self.data = None
    self.satisfaction_matrix = self._get_satisfaction_matrix()

def _load_data(self, data_path):
    """
    input : string
    output: dataframe
    Returns a preprocessed dataframe.
    """
    data = pd.read_pickle(data_path)
    data['topics'] = data['topics'].apply(self._topic_bins)
    return data

def _topic_bins(self, topic):
    """
    input: string
    output: string
    Converts 10 topics into 4 bins.
    """
    if topic in ['2', '6', '7']:
        return 'food'
    elif topic in ['3', '4']:
        return 'service'
    elif topic in ['8', '9']:
        return 'price'
    else:
        return 'ambiance'

def create_user_restaurant_feature(self):
    """
    Creates a dataframe with user, restaurant, feature combinations.
    """
    self.data = self.processed_data.groupby(['user_id','business_id','topics']).mean().reset_index()[['user_id','business_id','topics','scores']]
    users = self.data['user_id'].unique()
    features = self.data['topics'].unique()
    restaurants = self.data['business_id'].unique()
    result = list(itertools.product(users,restaurants,features))
    user_restaurant_feature = pd.DataFrame(result)
    user_restaurant_feature.columns = ['user_id','business_id','topics']
    self.data = pd.merge(user_restaurant_feature, self.data, how = 'left', on = ['user_id','business_id','topics'])
    # data.to_pickle('../data/merged_data.pkl')

def _calculate_restaurant_feature_scores(self):
    """
    Calculates and returns restaurnt feature scores.
    """
    restaurant_feature = self.data.groupby(['business_id','topics'])
    restaurant_feature_scores = self.data.groupby(['business_id','topics']).sum().reset_index()[['business_id','topics','scores']]
    restaurant_feature_scores.columns = ['business_id', 'topics', 'sum_scores']
    restaurant_feature_rated = restaurant_feature.count().reset_index()[['business_id', 'topics', 'scores']]
    restaurant_feature_rated.columns = ['business_id', 'topics', 'users_rated_rf']
    restaurant_feature = pd.merge(restaurant_feature_scores, restaurant_feature_rated, how = 'left', on = ['business_id','topics'])
    self.data = pd.merge(self.data, restaurant_feature, how = 'left', on = ['business_id','topics'])
    self.data['restaurant_scores'] = self.data['sum_scores']/data['users_rated_rf']
    self.data.drop(['sum_scores', 'users_rated_rf'], axis=1, inplace=True)

def _calculate_user_concern(self):
    """
    Calculates user conern on a feature.
    """
    user_rated = self.data.dropna().groupby(['user_id','business_id']).count().reset_index()[['user_id','business_id','scores']]
    user_rated.columns = ['user_id','business_id','user_rated_count']
    user_rated_feature = self.data.dropna().groupby(['user_id','topics']).count().reset_index()[['user_id','topics','scores']]
    user_rated_feature.columns = ['user_id', 'topics', 'user_rated_feature_count']
    feature_rated = self.data.dropna().groupby(['business_id','topics']).count().reset_index()[['business_id','topics','scores']]
    feature_rated.columns = ['business_id', 'topics', 'feature_rated_count']
    self.data = pd.merge(self.data, user_rated, how='left', on=['user_id','business_id'])
    self.data = pd.merge(self.data, user_rated_feature, how='left', on=['user_id','topics'])
    self.data = pd.merge(self.data, feature_rated, how='left', on=['business_id','topics'] )
    total_restaurants_reviewed = len(self.data['business_id'].unique())
    self.data['concern'] = ((self.data['user_rated_feature_count'] + 1)/self.data['user_rated_count']) * (total_restaurants_reviewed/(self.data['feature_rated_count'] + 1))

def _average_feature_score(self):
    """
    Calculates average feature score of a restaurant.
    """
    feature_rating = self.data.groupby(['business_id','topics']).mean().reset_index()[['business_id','topics','scores']]
    feature_rating.columns = ['business_id','topics','average_feature_score']
    self.data = pd.merge(self.data, feature_rating, how = 'left', on = ['business_id','topics'])
    self.data['scores'].fillna('0', inplace = True)
    self.data['user_rated_feature_count'].fillna(0, inplace = True)

@numba.jit
def _delta(self, average_score, score):
    """
    input : float, float
    output: float
    Computes delta which helps in calculating user requirement on a feature.
    """
    n = len(average_score)
    result = np.empty(n, dtype = 'float64')
    # average_score = row['avg_feature_rating']
    # score = row['scores']
    assert len(score) == n
    for i in xrange(n):
        if average_score[i] > score[i] > 0:
            result[i] = (average_score[i] - score[i] + 1)/average_score[i]
        else:
            result[i] = 1/average_score[i]
    return result

def _compute_numba_delta(self):
    """
    output : series
    Returns delta value.
    """
    result = self._delta(self.data['average_feature_score'].values, self.data['scores'].values)
    return pd.Series(result, index=self.data.index, name = 'delta')

@numba.jit
def _requirement(self, delta_sum, user_feature_rated_count, feature_rated_count, avg_rating_sum):
    """
    input : float, int, int, float
    output: float
    Computes users requirement.
    """
    n = len(delta_sum)
    result = np.empty(n, dtype = 'float64')
    assert len(user_feature_rated_count) == len(avg_rating_sum) == len(feature_rated_count) == n
    for i in xrange(n):
        if user_feature_rated_count[i]:
            result[i] = delta_sum[i]/user_feature_rated_count[i]
        else:
            result[i] = avg_rating_sum[i]/feature_rated_count[i]
    return result

def _compute_numba_requirement(self):
    """
    output: series
    Returns users requirement.
    """
    self.data['delta'] = self._compute_numba_delta()
    self.data['rec_avg_feature_rating'] = 0.1/self.data['average_feature_score']
    summations = self.data.groupby(['business_id','topics']).sum().reset_index()[['business_id','topics','delta','rec_avg_feature_rating']]
    summations.columns = ['business_id','topics','delta_sum','rec_avg_rating_sum']
    self.data = pd.merge(self.data, summations, how = 'left', on = ['business_id','topics'])
    result = requirement(self.data['delta_sum'].values, self.data['user_rated_feature_count'].values, self.data['feature_rated_count'].values, self.data['rec_avg_rating_sum'].values)
    return pd.Series(result, index=self.data.index, name = 'requirement')

def _weights(self):
    """
    Computes weights from users concern and requirement on a feature.
    """
    self.data['requirement'] = self._compute_numba_requirement()
    self.data['user_rated_count'].fillna(0, inplace = True)
    self.data['concern'].fillna(0, inplace = True)
    self.data['weights'] = self.data['concern'] * self.data['requirement']
    summed_weights = self.data.groupby('user_id').sum().reset_index()[['user_id','weights']]
    summed_weights.columns = ['user_id','weights_sum']
    self.data = pd.merge(self.data, summed_weights, how= 'left', on =['user_id'])

def _satisfaction(self):
    """
    output : data frame
    Computes scores based on user feature weights and restaurant feature scores.
    """
    self._weights()
    self.data['weighted_scores'] = self.data['weights'] * self.data['restaurant_scores']
    satisfaction = self.data.groupby(['user_id','business_id']).sum().reset_index()[['user_id','business_id','weighted_scores']]
    satisfaction.columns = ['user_id','business_id','weighted_scores_sum']
    self.data = pd.merge(self.data, satisfaction, how = 'left', on = ['user_id','business_id'])
    self.data['satisfaction'] = self.data['weighted_scores_sum']/self.data['weights_sum']
    self.data['predicted_scores'] = pd.cut(self.data['satisfaction'].values, bins=9, right=True, labels = [1,1.5,2,2.5,3,3.5,4,4.5,5],
            retbins=False, precision=2, include_lowest=True)
    self.data['predicted_scores'] = self.data['predicted_scores'].astype(float)
    return self.data[['user_id','business_id','predicted_scores']]

def _get_satisfaction_matrix(self):
    """
    output : data frame
    Returns satisfaction matrix with users scores for each restaurant.
    """
    satisfaction_matrix = self._satisfaction()
    satisfaction_matrix.pivot_table(index = ['user_id'], columns = ['business_id'], values= ['predicted_scores'])
    return satisfaction_matrix

def recommend_restaurants(self, user):
    """
    input : string
    output: list
    Recommends Restaurants to the user.
    """
    return satisfaction_matrix.ix(user).sort()[::-1][:5]

if __name__ == "__main__":
    path = '../data/processed_review_data.pkl'
    rr = RestaurantRecommender(path)
    # example
    user = 'SVLUlc3OuvuYQN6dlyqrnQ'
    restaurants = rr.recommend_restaurants(user)
