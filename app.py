import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Import the recommender module functions
from recommender import (
    build_recommendation_model,
    get_recommendations,
    search_based_recommendations,
    category_based_recommendations,
    get_hybrid_recommendations,
    CustomJSONEncoder
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter web integration

# Configure Flask app to use custom JSON encoder
app.json_encoder = CustomJSONEncoder

# Initialize Firebase (only if not already initialized)
if not firebase_admin._apps:
   firebase_credentials = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
   cred = credentials.Certificate(firebase_credentials)
   firebase_admin.initialize_app(cred)
db = firestore.client()

# Build model at startup
print("Building recommendation model...")
books_df, similarity_matrix, tfidf_vectorizer = build_recommendation_model(db)
print(f"Model built successfully. {len(books_df) if books_df is not None else 0} books loaded.")

@app.route('/health', methods=['GET'])
def health_check():
    """Basic endpoint to verify server is running"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Server is running',
        'books_loaded': len(books_df) if books_df is not None else 0
    })

@app.route('/fetch-books', methods=['GET'])
def fetch_books_api():
    """Fetch all books from both collections"""
    try:
        books_ref = db.collection('books').stream()
        resale_ref = db.collection('book_submissions').stream()
        
        all_books = []
        
        for book in books_ref:
            book_data = book.to_dict()
            book_data['id'] = book.id
            book_data['type'] = 'new'
            all_books.append(book_data)
        
        for book in resale_ref:
            book_data = book.to_dict()
            book_data['id'] = book.id
            book_data['type'] = 'resale'
            all_books.append(book_data)
        
        return jsonify({
            'status': 'success',
            'count': len(all_books),
            'books': all_books
        })
    
    except Exception as e:
        traceback.print_exc()  # Print full error for debugging
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/recommend-books', methods=['POST'])
def recommend_books():
    """Recommend books based on a given book ID with full book details"""
    try:
        data = request.get_json()
        book_id = data.get('book_id')
        num_recommendations = data.get('num_recommendations', 5)  # Optional parameter

        if not book_id:
            return jsonify({'status': 'error', 'message': 'Book ID is required'}), 400
        
        # Use the globally defined model components
        recommendations = get_recommendations(book_id, books_df, similarity_matrix, num_recommendations)
        
        return jsonify({
            'status': 'success',
            'count': len(recommendations),
            'recommendations': recommendations
        })
    
    except Exception as e:
        traceback.print_exc()  # Print full error for debugging
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/search-recommendations', methods=['POST'])
def search_recommendations():
    """Recommend books based on a search query"""
    try:
        data = request.get_json()
        search_query = data.get('search_query', '')
        category = data.get('category', '')  # Can be genre for new books or category for resale
        num_recommendations = data.get('num_recommendations', 10)
        
        # Use hybrid recommendations that combine search and category
        recommendations = get_hybrid_recommendations(
            search_query, 
            category, 
            books_df, 
            similarity_matrix, 
            tfidf_vectorizer, 
            num_recommendations
        )
        
        return jsonify({
            'status': 'success',
            'count': len(recommendations),
            'search_query': search_query,
            'category': category,
            'recommendations': recommendations
        })
    
    except Exception as e:
        traceback.print_exc()  # Print full error for debugging
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/category-recommendations', methods=['POST'])
def category_recommendations_api():
    """Recommend books based on category/genre"""
    try:
        data = request.get_json()
        category = data.get('category', '')
        num_recommendations = data.get('num_recommendations', 10)
        
        if not category:
            return jsonify({'status': 'error', 'message': 'Category is required'}), 400
        
        recommendations = category_based_recommendations(
            category, 
            books_df, 
            similarity_matrix, 
            num_recommendations
        )
        
        return jsonify({
            'status': 'success',
            'count': len(recommendations),
            'category': category,
            'recommendations': recommendations
        })
    
    except Exception as e:
        traceback.print_exc()  # Print full error for debugging
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
