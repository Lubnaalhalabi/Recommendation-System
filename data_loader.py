import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        """
        Initializes the DataLoader with the file path.
        :param file_path: Path to the CSV file containing the data.
        """
        self.file_path = file_path
    def load_data(self):
        """
        Loads data from the specified CSV file, removes duplicate rows, and creates a user-item matrix.
        :return: A tuple containing:
                 - user_item_ratings: Pandas DataFrame (user-item matrix with users as rows and books as columns)
                 - df: Original DataFrame after removing duplicates
        """
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"Error: File not found at '{self.file_path}'. Please check the file path.")
            return None, None
        except pd.errors.EmptyDataError:
            print(f"Error: The file at '{self.file_path}' is empty. Please provide a valid dataset.")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred while reading the file: {e}")
            return None, None
        try:
            # Remove duplicate rows based on user_id and book_id columns
            df = df.drop_duplicates(subset=['user_id', 'book_id'], keep='last')
            # Create a user-item matrix where rows represent users and columns represent books
            user_item_ratings = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        except KeyError as e:
            print(f"Error: Missing required column(s) in the dataset: {e}")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred during data processing: {e}")
            return None, None
        return user_item_ratings, df
