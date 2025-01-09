from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import jaccard, hamming
import numpy as np

class SimilarityCalculator:
    def __init__(self, user_item_ratings):
        """
        Initializes the SimilarityCalculator with a user-item ratings matrix.
        :param user_item_ratings: Pandas DataFrame where rows represent users, columns represent items,
                                  and values represent ratings.
        """
        self.user_item_ratings = user_item_ratings
    def calculate_similarity(self, method='cosine'):
        """
        Calculates the similarity matrix using the specified method.
        :param method: The similarity method to use ('cosine', 'euclidean', 'manhattan', 'hamming', 'jaccard').
        :return: A 2D numpy array representing the similarity matrix, or None if an error occurs.
        """
        try:
            if method == 'cosine':
                # Calculate cosine similarity
                return cosine_similarity(self.user_item_ratings)
            elif method == 'euclidean':
                # Calculate Euclidean similarity (inverse of Euclidean distance)
                return 1 / (1 + euclidean_distances(self.user_item_ratings))
            elif method == 'manhattan':
                # Calculate Manhattan similarity (inverse of Manhattan distance)
                return 1 / (1 + manhattan_distances(self.user_item_ratings))
            elif method == 'hamming':
                # Calculate Hamming similarity
                # Convert user-item ratings to binary: 1 if rated, 0 otherwise
                binary_ratings = self.user_item_ratings.applymap(lambda x: 1 if x > 0 else 0).values
                return 1 - np.array([[hamming(binary_ratings[i], binary_ratings[j])
                                      for j in range(len(binary_ratings))]
                                     for i in range(len(binary_ratings))])
            elif method == 'jaccard':
                # Calculate Jaccard similarity
                # Convert user-item ratings to binary: 1 if rated, 0 otherwise
                binary_ratings = self.user_item_ratings.applymap(lambda x: 1 if x > 0 else 0).values
                return 1 - np.array([[jaccard(binary_ratings[i], binary_ratings[j])
                                      for j in range(len(binary_ratings))]
                                     for i in range(len(binary_ratings))])
            else:
                # Raise an error for unknown methods
                raise ValueError(f"Unknown similarity method: {method}")
        
        except ValueError as ve:
            print(f"ValueError: {ve}")
            return None
        except KeyError as ke:
            print(f"KeyError: {ke}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
