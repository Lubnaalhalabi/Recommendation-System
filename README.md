# Book Recommendation System
This project is a Book Recommendation System that uses Collaborative Filtering to suggest books to users based on their preferences and the preferences of similar users. The system includes a web-based interface built with Streamlit and allows users to explore book recommendations interactively.

# Features
1. Load and preprocess book rating data.
2. Calculate user similarity using various methods:
  - Cosine Similarity
  - Euclidean Distance
  - Manhattan Distance
  - Hamming Distance
  - Jaccard Distance
3. Generate top-N book recommendations for a specific user.
4. Display recommended books with their titles, images, and descriptions.
5. Interactive user input and output through a Streamlit interface.

# Project Structure
- data_loader.py: DataLoader class for loading and preprocessing data
- similarity_calculator.py: SimilarityCalculator class for calculating user similarity
- collaborative_filtering_recommender.py: CollaborativeFilteringRecommender class for generating recommendations
- lubna.csv: Dataset containing user-book ratings and details
- app.py: Main Streamlit application

# Prerequisites
Ensure you have Python 3.8 or later installed on your machine. The following libraries are required:
- pandas
- numpy
- scikit-learn
- scipy
- streamlit

# How to Run
1. Start the Streamlit application.
2. Open your web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).
3. Enter a User ID and click the Get Collaborative Filtering Recommendations button to view recommended books.

# Dataset Format
The dataset (`lubna.csv`) should contain the following columns:
- user_id: The unique identifier for the user.
- book_id: The unique identifier for the book.
- rating: The rating given by the user to the book.
- title: The title of the book.
- image: URL or path to the book cover image.
- description: A short description of the book.

# How It Works
1. Data Loading and Preprocessing:  
   The DataLoader class loads the dataset, removes duplicate ratings, and creates a user-item matrix.
2. Similarity Calculation:  
   The SimilarityCalculator class calculates user similarity using the specified method (default is cosine similarity).
3. Recommendations:  
   The CollaborativeFilteringRecommender class identifies similar users and recommends books that the target user hasn't rated yet.
4. Streamlit UI:  
   The Streamlit interface allows users to input their User ID and view recommendations interactively.

# Customization
- Similarity Method: You can change the similarity method by modifying the method argument in the calculate_similarity function (e.g., method='jaccard').
- Top-N Recommendations: Adjust the top_n parameter in the recommend method to change the number of recommendations.
