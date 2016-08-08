from __future__ import division
from geopy.geocoders import Nominatim
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
pd.set_option('display.width', 100)

def extract_locations(df):
    """
    input : Data Frame
    output: string
    Extracts the location based on latitude and longitude.
    """
    db = DBSCAN(eps = 1, min_samples = 100) # eps 1 = 111.2KM, since the cities provided is atleast 100 miles away from each other
    labels = db.fit_predict(df[['latitude','longitude']])
    labels = np.expand_dims(labels, axis = 1)
    df['location'] = labels
    geocoder = Nominatim()
    states = {}
    centroids = {}
    exceptions = []
    for label in np.unique(labels):
        X = df[df.location == label][['latitude','longitude']]
        lat, lon = X.mean()
        location = geocoder.reverse((lat,lon))
        time.sleep(1)
        try:    # Works for most countries
            states[label] = location.raw['address']['state']
            centroids[label] = (lat,lon)
        except:
            exceptions.append((lat,lon))
            states[label] = (lat,lon)
            centroids[label] = (lat,lon)
    df['city_center'] = df['location'].map(centroids)
    df['location'] = df['location'].map(states)
    return exceptions

def get_business_ids(file_path, cuisine, location):
    """
    input : string, string, string
    output: Data Frame
    Returns a data frame with filters on the input arguments.
    """
    df_restaurants = pd.read_pickle(file_path)
    df_cuisine = df_restaurants[df_restaurants['categories'].str.contains(cuisine)]
    return df_cuisine[df_cuisine['location'] == cuisine]['business_id']

def get_business_reviews(file_path, business_ids):
    """
    input : string, string
    output: numpy array
    Returns the reviews with the business ids as given in input.
    """
    reviews = pd.read_pickle(file_path)
    return reviews[reviews['business_id'].isin(business_ids)]

if __name__=="__main__":
    df_business = pd.read_csv("../data/yelp_dataset_business.csv")
    df_restaurants = df_business[(df_business['categories']).str.contains('Restaurants')]
    extract_locations(df_restaurants)
    df_restaurants.to_pickle("../data/restaurant_data.pkl")
    business_ids = get_business_ids('../data/restaurant_data.pkl', 'Italian', 'Nevada')
    df_reviews = get_business_reviews('../data/review_data.pkl', business_ids)
