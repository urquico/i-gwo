"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library and optimizes using IGWO.
"""

from pathlib import Path
from typing import Tuple, List
import io
import logging

import implicit
import scipy
import numpy as np
import matplotlib.pyplot as plt
import csv
import unicodedata

from data import load_user_artists, ArtistRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.rcParams['font.family'] = 'DejaVu Sans'

def unicode_to_ascii(text):
    """Convert Unicode characters to ASCII representation."""
    return ''.join(
        c if ord(c) < 128 else f'\\N{{{unicodedata.name(c)}}}'
        for c in str(text)
    )

class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[user_id], N=n, filter_already_liked_items=True
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

# IGWO Functions
def initial_variables(size, min_values, max_values, target_function, start_init = None):
    dim = len(min_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows       = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        population     = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    return population

def alpha_position(min_values, max_values, target_function):
    alpha       = np.zeros((1, len(min_values) + 1))
    alpha[0,-1] = target_function(np.clip(alpha[0,0:alpha.shape[1]-1], min_values, max_values))
    return alpha[0,:]

def beta_position(min_values, max_values, target_function):
    beta       = np.zeros((1, len(min_values) + 1))
    beta[0,-1] = target_function(np.clip(beta[0,0:beta.shape[1]-1], min_values, max_values))
    return beta[0,:]

def delta_position(min_values, max_values, target_function):
    delta       =  np.zeros((1, len(min_values) + 1))
    delta[0,-1] = target_function(np.clip(delta[0,0:delta.shape[1]-1], min_values, max_values))
    return delta[0,:]

def update_pack(position, alpha, beta, delta):
    idx   = np.argsort(position[:, -1])
    alpha = position[idx[0], :]
    beta  = position[idx[1], :] if position.shape[0] > 1 else alpha
    delta = position[idx[2], :] if position.shape[0] > 2 else beta
    return alpha, beta, delta

def update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function):
    dim                     = len(min_values)
    alpha_position          = np.copy(position)
    beta_position           = np.copy(position)
    delta_position          = np.copy(position)
    updated_position        = np.copy(position)
    r1                      = np.random.rand(position.shape[0], dim)
    r2                      = np.random.rand(position.shape[0], dim)
    a                       = 2 * a_linear_component * r1 - a_linear_component
    c                       = 2 * r2
    distance_alpha          = np.abs(c * alpha[:dim] - position[:, :dim])
    distance_beta           = np.abs(c * beta [:dim] - position[:, :dim])
    distance_delta          = np.abs(c * delta[:dim] - position[:, :dim])
    x1                      = alpha[:dim] - a * distance_alpha
    x2                      = beta [:dim] - a * distance_beta
    x3                      = delta[:dim] - a * distance_delta
    alpha_position[:,:-1]   = np.clip(x1, min_values, max_values)
    beta_position [:,:-1]   = np.clip(x2, min_values, max_values)
    delta_position[:,:-1]   = np.clip(x3, min_values, max_values)
    alpha_position[:, -1]   = np.apply_along_axis(target_function, 1, alpha_position[:, :-1])
    beta_position [:, -1]   = np.apply_along_axis(target_function, 1, beta_position [:, :-1])
    delta_position[:, -1]   = np.apply_along_axis(target_function, 1, delta_position[:, :-1])
    updated_position[:,:-1] = np.clip((alpha_position[:, :-1] + beta_position[:, :-1] + delta_position[:, :-1]) / 3, min_values, max_values)
    updated_position[:, -1] = np.apply_along_axis(target_function, 1, updated_position[:, :-1])
    updated_position        = np.vstack([position, updated_position, alpha_position, beta_position, delta_position])
    updated_position        = updated_position[updated_position[:, -1].argsort()]
    updated_position        = updated_position[:position.shape[0], :]
    return updated_position

def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

def improve_position(position, updt_position, min_values, max_values, target_function):
    i_position  = np.copy(position)
    dist_matrix = build_distance_matrix(position[:, :-1])
    min_values  = np.array(min_values)
    max_values  = np.array(max_values)
    for i in range(position.shape[0]):
        dist = euclidean_distance(position[i, :-1], updt_position[i, :-1])
        idx  = np.where(dist_matrix[i, :] <= dist)[0]
        for j in range(len(min_values)):
            rand             = np.random.rand()
            ix_1             = np.random.choice(idx)
            ix_2             = np.random.choice(position.shape[0])
            i_position[i, j] = np.clip(i_position[i, j] + rand * (position[ix_1, j] - position[ix_2, j]), min_values[j], max_values[j])
        i_position[i, -1] = target_function(i_position[i, :-1])
        min_fitness       = min(position[i, -1], updt_position[i, -1], i_position[i, -1])
        if (updt_position[i, -1] == min_fitness):
            i_position[i, :] = updt_position[i, :]
        elif (position[i, -1] == min_fitness):
            i_position[i, :] = position[i, :]
    return i_position

def improved_grey_wolf_optimizer(pack_size, min_values, max_values, iterations, target_function, verbose = True, start_init = None, target_value = None):   
    alpha    = alpha_position(min_values, max_values, target_function)
    beta     = beta_position (min_values, max_values, target_function)
    delta    = delta_position(min_values, max_values, target_function)
    position = initial_variables(pack_size, min_values, max_values, target_function, start_init)
    count    = 0
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', alpha[-1])      
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        updt_position      = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function)      
        position           = improve_position(position, updt_position, min_values, max_values, target_function)
        if (target_value is not None):
            if (alpha[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1       
    return alpha

# Modified functions to incorporate IGWO
def optimize_model_parameters(user_artists_matrix):
    def target_function(params):
        factors, regularization = params
        model = implicit.als.AlternatingLeastSquares(
            factors=int(factors),
            regularization=regularization,
            iterations=10  # Changed to 10
        )
        model.fit(user_artists_matrix)
        
        # Use mean squared error as a proxy for model performance
        user_items = user_artists_matrix.T.tocsr()
        user_factors = model.user_factors
        item_factors = model.item_factors
        predictions = user_factors.dot(item_factors.T)
        mse = np.mean((user_artists_matrix.data - predictions[user_artists_matrix.nonzero()]) ** 2)
        return mse  # We want to minimize this

    pack_size = 10
    min_values = [10, 0.001]  # Minimum values for factors and regularization
    max_values = [100, 1.0]   # Maximum values for factors and regularization
    iterations = 10  # Changed to 10

    best_params = improved_grey_wolf_optimizer(
        pack_size=pack_size,
        min_values=min_values,
        max_values=max_values,
        iterations=iterations,
        target_function=target_function,
        verbose=True
    )

    return int(best_params[0]), best_params[1]  # factors, regularization

def generate_results(user_index: int, recommend_limit: int = 10):
    logging.info(f"Generating results for user {user_index}")
    
    # load user artists matrix
    user_artists = load_user_artists(Path("../../dataset/user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../../dataset/artists.dat"))

    # Optimize model parameters using IGWO
    factors, regularization = optimize_model_parameters(user_artists)
    logging.info(f"Using parameters: factors={factors}, regularization={regularization}")

    # instantiate ALS using implicit with optimized parameters
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=10, regularization=regularization
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(user_index, user_artists, n=recommend_limit)
    
    # store the top 10 artists that the user has listened to
    top_10_artists = []
    
    # store the top 10 recommendations
    top_10_recommendations = []
    top_10_scores = []
    # store the top 10 artists that the user has listened to
    user_artists_indices = user_artists[user_index].nonzero()[1]
    for artist_id in user_artists_indices[:recommend_limit]:  # limit to top 10
        artist_name = artist_retriever.get_artist_name_from_id(artist_id)
        top_10_artists.append(artist_name)

    # store the top 10 recommendations
    for artist, score in zip(artists, scores):
        top_10_recommendations.append(artist)
        top_10_scores.append(score)
   
    # Combine the listened artists and recommended artists into a list of tuples
    table_data = list(zip(top_10_artists, top_10_recommendations, top_10_scores))

    # Plot the table using matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size
    ax.axis('tight')
    ax.axis('off')
    
    # Create a table with ASCII representation of Unicode strings
    cell_text = [[unicode_to_ascii(str(cell)) for cell in row] for row in table_data]
    table = ax.table(cellText=cell_text, colLabels=["Listened Artist", "Recommended Artist", "Score"], 
                     cellLoc='center', loc='center', colWidths=[0.4, 0.4, 0.2])
    
    # Adjust font size and wrapping
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(3)))

    # Wrap text in cells
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(wrap=True)

    # Save the table as an image based on the user ID
    plt.savefig(f"results/result_user_{user_index}.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Save the table as a CSV file inside the results folder
    with io.open(f"results/result_user_{user_index}.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Listened Artist", "Recommended Artist", "Score"])
        for row in table_data:
            writer.writerow([unicode_to_ascii(str(cell)) for cell in row])

    logging.info(f"Results generated for user {user_index}")

def evaluate_recommendations(user_index: int, recommend_limit: int = 50):
    logging.info(f"Evaluating recommendations for user {user_index}")
    
    # Load user artists matrix
    user_artists = load_user_artists(Path("../../dataset/user_artists.dat"))

    # Instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../../dataset/artists.dat"))

    # Optimize model parameters using IGWO
    factors, regularization = optimize_model_parameters(user_artists)
    logging.info(f"Using parameters: factors={factors}, regularization={regularization}")

    # Instantiate ALS using implicit with optimized parameters
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=10, regularization=regularization
    )

    # Instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    recommended_artists, _ = recommender.recommend(user_index, user_artists, n=recommend_limit)

    # Get actual listened artists
    actual_artists_indices = user_artists[user_index].nonzero()[1]
    actual_artists = [
        artist_retriever.get_artist_name_from_id(artist_id)
        for artist_id in actual_artists_indices
    ]

    logging.info(f"User {user_index} - Actual artists: {len(actual_artists)}, Recommended artists: {len(recommended_artists)}")
    logging.info(f"Sample actual artists: {actual_artists[:5]}")
    logging.info(f"Sample recommended artists: {recommended_artists[:5]}")

    precision, recall, f1_score = calculate_precision_recall_f1(actual_artists, recommended_artists)
    
    with io.open(f"results/evaluation_user_{user_index}.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Precision", "Recall", "F1-Score"])
        writer.writerow([precision, recall, f1_score])
    
    logging.info(f"User {user_index} - Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")

def calculate_precision_recall_f1(actual_items: List[str], recommended_items: List[str]) -> Tuple[float, float, float]:
    actual_set = set(actual_items)
    recommended_set = set(recommended_items)

    true_positives = len(actual_set.intersection(recommended_set))
    false_positives = len(recommended_set - actual_set)
    false_negatives = len(actual_set - recommended_set)

    precision = true_positives / len(recommended_set) if len(recommended_set) > 0 else 0
    recall = true_positives / len(actual_set) if len(actual_set) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logging.info(f"True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}")
    logging.info(f"Actual Set: {len(actual_set)}, Recommended Set: {len(recommended_set)}")

    return precision, recall, f1_score

def analyze_user_data(user_index: int):
    user_artists = load_user_artists(Path("../../dataset/user_artists.dat"))
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../../dataset/artists.dat"))

    user_data = user_artists[user_index].tocsr()
    non_zero_indices = user_data.nonzero()[1]
    
    logging.info(f"User {user_index} has listened to {len(non_zero_indices)} unique artists")
    
    if len(non_zero_indices) > 0:
        top_5_indices = non_zero_indices[np.argsort(user_data[0, non_zero_indices].toarray()[0])[-5:]]
        top_5_artists = [artist_retriever.get_artist_name_from_id(idx) for idx in top_5_indices]
        logging.info(f"Top 5 artists for User {user_index}: {top_5_artists}")
    else:
        logging.warning(f"User {user_index} has no listening history")

if __name__ == "__main__":
    for user_index in range(2, 11):
        try:
            analyze_user_data(user_index)
            generate_results(user_index=user_index, recommend_limit=10)
            evaluate_recommendations(user_index=user_index, recommend_limit=50)
        except Exception as e:
            logging.error(f"Error processing user {user_index}: {str(e)}")

