import numpy as np
from geopy.distance import geodesic

def compute_distance_matrix(centroids, country_list):  # This is where the magic happens :). for more info refer to readMe
    dist_matrix = np.zeros((len(country_list), len(country_list)))
    for i, ci in enumerate(country_list):
        coord_i = (centroids.loc[centroids['country_code'] == ci, 'lat'].values[0],
                   centroids.loc[centroids['country_code'] == ci, 'lon'].values[0])
        for j, cj in enumerate(country_list):
            coord_j = (centroids.loc[centroids['country_code'] == cj, 'lat'].values[0],
                       centroids.loc[centroids['country_code'] == cj, 'lon'].values[0])
            dist_matrix[i, j] = geodesic(coord_i, coord_j).km
    return dist_matrix

def compute_geodesic_error(preds, targets, centroids, country_list):
    errors = []
    for p, t in zip(preds, targets):
        coord_pred = (centroids.loc[centroids['country_code'] == country_list[p], 'lat'].values[0],
                      centroids.loc[centroids['country_code'] == country_list[p], 'lon'].values[0])
        coord_true = (centroids.loc[centroids['country_code'] == country_list[t], 'lat'].values[0],
                      centroids.loc[centroids['country_code'] == country_list[t], 'lon'].values[0])
        errors.append(geodesic(coord_pred, coord_true).km)
    return np.mean(errors)
