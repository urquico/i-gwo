"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""


from pathlib import Path
from typing import Tuple, List

import implicit
import scipy
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv

from data import load_user_artists, ArtistRetriever


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
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

def generate_results(user_index: int, recommend_limit: int = 10):
    # 2 - 2100
    
    # load user artists matrix
    user_artists = load_user_artists(Path("../dataset/user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../dataset/artists.dat"))

    # instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
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
        # print(artist_name)

    # store the top 10 recommendations
    for artist, score in zip(artists, scores):
        top_10_recommendations.append(artist)
        top_10_scores.append(score)
        # print(f"{artist}: {score}")
   
    # Combine the listened artists and recommended artists into a list of tuples
    table_data = list(zip(top_10_artists, top_10_recommendations, top_10_scores))

    # Create the table with headers
    table = tabulate(table_data, headers=["Listened Artist", "Recommended Artist", "Score"])

    # Plot the table using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=table_data, colLabels=["Listened Artist", "Recommended Artist", "Score"], cellLoc='center', loc='center')

    # Save the table as an image based on the user ID
    plt.savefig(f"original/result_user_{user_index}.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Save the table as a CSV file inside the original folder
    with open(f"original/result_user_{user_index}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Listened Artist", "Recommended Artist", "Score"])
        writer.writerows(table_data)

if __name__ == "__main__":
    generate_results(user_index=5, recommend_limit=10)
