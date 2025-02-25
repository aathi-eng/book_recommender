import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Custom JSON encoder to handle NaT, NaN, and other non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj):
            return None
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)

# Fetch books from Firestore
def fetch_books(db):
    books_ref = db.collection('books').stream()
    resale_ref = db.collection('book_submissions').stream()
    
    books_data = []
    
    for book in books_ref:
        book_data = book.to_dict()
        book_data['id'] = book.id
        book_data['type'] = 'new'
        # Ensure 'genres' exists, convert to string for processing
        if 'genres' in book_data and book_data['genres']:
            if isinstance(book_data['genres'], list):
                book_data['genres_str'] = ' '.join(book_data['genres'])
            else:
                book_data['genres_str'] = str(book_data['genres'])
        else:
            book_data['genres_str'] = ''
        books_data.append(book_data)

    for book in resale_ref:
        book_data = book.to_dict()
        book_data['id'] = book.id
        book_data['type'] = 'resale'
        # Ensure 'category' exists, use as genres_str for consistent processing
        if 'category' in book_data and book_data['category']:
            book_data['genres_str'] = str(book_data['category'])
        else:
            book_data['genres_str'] = ''
        books_data.append(book_data)
    
    return books_data

# Helper function to clean dictionary for JSON serialization
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj) or obj is pd.NaT:
        return None
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Build the content-based recommendation model
def build_recommendation_model(db):
    """Build and return the recommendation model components"""
    books = fetch_books(db)
    
    if not books:
        return None, None, None
    
    df = pd.DataFrame(books)
    
    # Handle missing descriptions and create a combined text field for better recommendations
    df['description'] = df['description'].fillna('')
    df['title'] = df['title'].fillna('')
    df['author'] = df['author'].fillna('')
    df['genres_str'] = df['genres_str'].fillna('')
    
    # Create a combined text field for better recommendations
    df['combined_features'] = df['title'] + ' ' + df['author'] + ' ' + df['description'] + ' ' + df['genres_str']
    
    # Create TF-IDF vectorizer and matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, similarity_matrix, tfidf

# Get recommendations based on book_id
def get_recommendations(book_id, books_df, similarity_matrix, num_recommendations=5):
    """Get book recommendations with full details based on book_id"""
    # Check if the given book exists
    if book_id not in books_df['id'].values:
        print(f"Book ID {book_id} not found in dataset.")
        return []

    # Get the index of the book
    idx = books_df.index[books_df['id'] == book_id].tolist()[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    # Get book indices and similarity scores
    book_indices = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]
    
    # Get recommended books with all fields
    recommended_books = []
    for i, sim_value in zip(book_indices, similarity_values):
        book_data = books_df.iloc[i].to_dict()
        book_data['similarity_score'] = float(sim_value)  # Add similarity score
        # Clean data for JSON serialization
        book_data = clean_for_json(book_data)
        recommended_books.append(book_data)
    
    return recommended_books

# Get recommendations based on search query
def search_based_recommendations(search_query, books_df, tfidf_vectorizer, num_recommendations=10):
    """Get book recommendations based on a search query"""
    if books_df is None or tfidf_vectorizer is None:
        return []
    
    # Ensure combined_features exists
    if 'combined_features' not in books_df.columns:
        books_df['title'] = books_df['title'].fillna('')
        books_df['author'] = books_df['author'].fillna('')
        books_df['description'] = books_df['description'].fillna('')
        books_df['genres_str'] = books_df['genres_str'].fillna('')
        books_df['combined_features'] = books_df['title'] + ' ' + books_df['author'] + ' ' + books_df['description'] + ' ' + books_df['genres_str']
    
    # Transform the search query into the TF-IDF space
    try:
        query_tfidf = tfidf_vectorizer.transform([search_query])
        
        # Calculate similarity between query and all books
        similarity_scores = cosine_similarity(query_tfidf, tfidf_vectorizer.transform(books_df['combined_features']))
        
        # Convert to list of (index, similarity) tuples and sort
        sim_scores = list(enumerate(similarity_scores[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
        
        # Get book indices and similarity scores
        book_indices = [i[0] for i in sim_scores]
        similarity_values = [i[1] for i in sim_scores]
        
        # Return books matching the search query
        recommended_books = []
        for i, sim_value in zip(book_indices, similarity_values):
            if sim_value > 0:  # Only include books with some relevance
                book_data = books_df.iloc[i].to_dict()
                book_data['similarity_score'] = float(sim_value)
                # Clean data for JSON serialization
                book_data = clean_for_json(book_data)
                recommended_books.append(book_data)
        
        return recommended_books
    except Exception as e:
        print(f"Error in search_based_recommendations: {str(e)}")
        return []

# Get recommendations based on category/genre
def category_based_recommendations(category, books_df, similarity_matrix, num_recommendations=10):
    """Get book recommendations based on category/genre"""
    if books_df is None or similarity_matrix is None:
        return []
    
    # Filter books by category/genre
    filtered_books = []
    
    for idx, book in books_df.iterrows():
        # Check if it's a new book with genres
        if book['type'] == 'new' and 'genres' in book:
            genres = book.get('genres', [])
            if isinstance(genres, list) and category in genres:
                filtered_books.append((idx, book))
            elif isinstance(genres, str) and category.lower() in genres.lower():
                filtered_books.append((idx, book))
        
        # Check if it's a resale book with category
        elif book['type'] == 'resale' and 'category' in book:
            if book['category'] and category.lower() in str(book['category']).lower():
                filtered_books.append((idx, book))
    
    if not filtered_books:
        return []
    
    # Calculate average similarity scores for all books in the category
    all_recommendations = []
    
    # Limit to prevent excessive processing
    filtered_books = filtered_books[:min(len(filtered_books), 5)]
    
    for idx, _ in filtered_books:
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        # Add to overall recommendations list
        all_recommendations.extend(sim_scores)
    
    # Aggregate recommendations by averaging scores for duplicate book indices
    book_scores = {}
    for idx, score in all_recommendations:
        if idx not in book_scores:
            book_scores[idx] = []
        book_scores[idx].append(score)
    
    # Calculate average score for each book
    avg_scores = [(idx, sum(scores) / len(scores)) for idx, scores in book_scores.items()]
    
    # Sort by average similarity score
    avg_scores = sorted(avg_scores, key=lambda x: x[1], reverse=True)
    
    # Filter out books that were used as seeds
    seed_indices = [idx for idx, _ in filtered_books]
    avg_scores = [(idx, score) for idx, score in avg_scores if idx not in seed_indices]
    
    # Take top recommendations
    avg_scores = avg_scores[:num_recommendations]
    
    # Get recommended books with all fields
    recommended_books = []
    for idx, sim_value in avg_scores:
        book_data = books_df.iloc[idx].to_dict()
        book_data['similarity_score'] = float(sim_value)
        # Clean data for JSON serialization
        book_data = clean_for_json(book_data)
        recommended_books.append(book_data)
    
    return recommended_books

# Combined search and category recommendations
def get_hybrid_recommendations(search_query, category, books_df, similarity_matrix, tfidf_vectorizer, num_recommendations=10):
    """Get recommendations based on both search query and category if available"""
    search_results = []
    category_results = []
    
    # Get search-based recommendations if search query provided
    if search_query and len(search_query.strip()) > 0:
        search_results = search_based_recommendations(search_query, books_df, tfidf_vectorizer, num_recommendations)
    
    # Get category-based recommendations if category provided
    if category and len(category.strip()) > 0:
        category_results = category_based_recommendations(category, books_df, similarity_matrix, num_recommendations)
    
    # If we have both search and category results, combine them with weights
    if search_results and category_results:
        # Create dictionaries for easy lookup of scores
        search_dict = {book['id']: book for book in search_results}
        category_dict = {book['id']: book for book in category_results}
        
        # Combine unique books from both lists
        all_ids = set(search_dict.keys()) | set(category_dict.keys())
        combined_results = []
        
        for book_id in all_ids:
            if book_id in search_dict and book_id in category_dict:
                # If book appears in both lists, take average of scores with weight
                book = search_dict[book_id].copy()
                search_score = search_dict[book_id]['similarity_score']
                category_score = category_dict[book_id]['similarity_score']
                # Weight search results higher (0.7) vs category results (0.3)
                combined_score = (search_score * 0.7) + (category_score * 0.3)
                book['similarity_score'] = combined_score
                combined_results.append(book)
            elif book_id in search_dict:
                combined_results.append(search_dict[book_id])
            else:
                combined_results.append(category_dict[book_id])
        
        # Sort by combined similarity score
        combined_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return combined_results[:num_recommendations]
    
    # If we only have one type of results, return those
    elif search_results:
        return search_results
    elif category_results:
        return category_results
    
    # If no results from either method, return empty list
    return []