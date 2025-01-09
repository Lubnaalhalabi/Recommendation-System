import numpy as np

class CollaborativeFilteringRecommender:
    def __init__(self, user_similarity, user_item_ratings):
        """
        Initializes the recommender system with user similarity scores and user-item ratings.
        :param user_similarity: 2D numpy array or DataFrame containing user similarity scores.
        :param user_item_ratings: Pandas DataFrame where rows represent users, columns represent books,
                                  and values represent ratings.
        """
        self.user_similarity = user_similarity
        self.user_item_ratings = user_item_ratings

    def recommend(self, user_id, top_n=3):
        """
        Recommends books to a given user based on collaborative filtering.
        :param user_id: ID of the target user for whom recommendations are to be generated.
        :param top_n: Number of book recommendations to return.
        :return: List of recommended book IDs.
        """
        try:
            # Convert user_id (1-based) to user_idx (0-based)
            user_idx = user_id - 1

            # Validate user_id
            if user_idx < 0 or user_idx >= len(self.user_similarity):
                raise ValueError(f"Invalid user_id: {user_id}. Please provide a valid user ID.")

            # Get similarity scores for the target user
            sim_scores = list(enumerate(self.user_similarity[user_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Identify top similar users (excluding the target user themselves)
            similar_users = [user_idx for user_idx, score in sim_scores[1:top_n+1]]

            # Get books already rated by the target user
            user_rated_books = set(self.user_item_ratings.columns[self.user_item_ratings.loc[user_id] > 0])

            # Generate recommendations based on ratings by similar users
            recommended_books = []
            for user in similar_users:
                similar_user_ratings = self.user_item_ratings.iloc[user]
                for book, rating in similar_user_ratings.items():
                    if rating > 0 and book not in user_rated_books:
                        recommended_books.append(book)

            # Remove duplicates and return the top_n recommendations
            return list(np.unique(recommended_books))[:top_n]

        except KeyError as e:
            print(f"Error: User ID not found in the data. Details: {e}")
            return []
        except IndexError as e:
            print(f"Error: User index out of range. Details: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []
