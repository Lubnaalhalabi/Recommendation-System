from data_loader import DataLoader
from similarity_calculator import SimilarityCalculator
from collaborative_filtering_recommender import CollaborativeFilteringRecommender
import streamlit as st

def main():
    """
    Main function to run the Book Recommendation System.
    Loads data, calculates similarities, and provides recommendations via a Streamlit UI.
    """
    try:
        # Load data from the specified CSV file
        file_path = "lubna.csv"
        data_loader = DataLoader(file_path)
        user_item_ratings, df = data_loader.load_data()

        # Calculate user similarity using the specified method
        similarity_calculator = SimilarityCalculator(user_item_ratings)
        user_similarity = similarity_calculator.calculate_similarity(method='hamming')

        # Initialize the collaborative filtering recommender
        collaborative_recommender = CollaborativeFilteringRecommender(user_similarity, user_item_ratings)

        # Streamlit UI
        st.title("Book Recommendation System")

        # User input for User ID
        user_id = st.number_input(
            "Enter User ID:",
            min_value=int(user_item_ratings.index.min()), 
            max_value=int(user_item_ratings.index.max()), 
            step=1
        )

        # Handle the recommendation process when the button is clicked
        if st.button("Get Collaborative Filtering Recommendations"):
            try:
                # Get recommendations for the given User ID
                collaborative_recommendations = collaborative_recommender.recommend(user_id, top_n=3)
                
                # Display recommendations
                st.write(f"Collaborative Filtering Recommended Books for User {user_id}:")
                for book_id in collaborative_recommendations:
                    # Filter the book details from the dataset
                    filtered_books = df[df['book_id'] == book_id]
                    
                    if not filtered_books.empty:
                        # Display book details if available
                        book_details = filtered_books.iloc[0]
                        st.subheader(book_details['title'])
                        st.image(book_details['image'], caption=book_details['description'])
                    else:
                        # Warn if book details are not found
                        st.warning(f"Book details for ID {book_id} not found!")
            
            except KeyError as ke:
                st.error(f"KeyError: {ke} - Please check the dataset for missing or incorrect data.")
            except Exception as e:
                st.error(f"An unexpected error occurred while generating recommendations: {e}")

    except FileNotFoundError as fnfe:
        st.error(f"FileNotFoundError: {fnfe} - The specified file '{file_path}' was not found.")
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}")

if __name__ == "__main__":
    main()
