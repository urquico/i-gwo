"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""

from pathlib import Path
from typing import Tuple, List
import io

import implicit
import scipy
import matplotlib.pyplot as plt
import csv
import unicodedata
from datetime import datetime

from data import load_user_artists, ArtistRetriever

# Set a font that supports a wide range of Unicode characters
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
            user_id, user_artists_matrix[user_id], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

def generate_results(user_index: int, recommend_limit: int = 10, factors: int = 50, iterations: int = 10, regularization: float = 0.01):
    # 2 - 2100
    
    # load user artists matrix
    user_artists = load_user_artists(Path("../../dataset/user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../../dataset/artists.dat"))

    # instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iterations, regularization=regularization
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
        
    # create folder for specific user
    Path(f"results/user_{user_index}").mkdir(parents=True, exist_ok=True)
    
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the table as an image based on the user ID
    plt.savefig(f"results/user_{user_index}/time_{time_stamp}_factors_{factors}_iterations_{iterations}_regularization_{regularization}.png", bbox_inches='tight', dpi=300)
    plt.close()

    # create csv folder for specific user
    Path(f"results/user_{user_index}/csv").mkdir(parents=True, exist_ok=True)

    # Save the table as a CSV file inside the results folder
    with io.open(f"results/user_{user_index}/csv/time_{time_stamp}_factors_{factors}_iterations_{iterations}_regularization_{regularization}.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Listened Artist", "Recommended Artist", "Score"])
        for row in table_data:
            writer.writerow([unicode_to_ascii(str(cell)) for cell in row])
            
    print(f"Results saved for user {user_index} with factors {factors}, iterations {iterations}, and regularization {regularization}")

def evaluate_recommendations(user_index: int, recommend_limit: int = 50):
    # Load user artists matrix
    user_artists = load_user_artists(Path("../../dataset/user_artists.dat"))

    # Instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../../dataset/artists.dat"))

    # Instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # Instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    recommended_artists, _ = recommender.recommend(user_index, user_artists, n=recommend_limit)

    # Get actual listened artists
    actual_artists_indices = user_artists[user_index].nonzero()[1]
    actual_artists = [
        artist_retriever.get_artist_name_from_id(artist_id)
        for artist_id in actual_artists_indices[:recommend_limit]
    ]

if __name__ == "__main__":
    # Define the ranges for each parameter
    factors_range = range(10, 101, 10)  # Factors from 10 to 100
    iterations_range = range(10, 101, 10)  # Iterations from 10 to 100
    regularization_range = [0.001, 0.01, 0.1, 1.0]  # Regularization values

    best_score = 0
    best_params = None

    for factors in factors_range:
        for iterations in iterations_range:
            for regularization in regularization_range:
                for user_index in range(5, 6):
                    model = implicit.als.AlternatingLeastSquares(
                        factors=factors, iterations=iterations, regularization=regularization
                    )
                    
                    # Generate results for each combination of parameters
                    generate_results(
                        user_index=user_index,
                        recommend_limit=10,
                        factors=factors,
                        iterations=iterations,
                        regularization=regularization
                    )