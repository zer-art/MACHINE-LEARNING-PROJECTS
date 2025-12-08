import time
import numpy as np
import pandas as pd
import os
from src.data.load_data import load_data
from src.models.recommender import HybridRecommender

def create_dummy_data():
    print("Creating dummy data for benchmark...")
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create movies.dat
    movies = pd.DataFrame({
        'movieId': range(1, 101),
        'title': [f'Movie {i} ({1990+i%30})' for i in range(1, 101)],
        'genres': ['Action|Adventure', 'Comedy', 'Drama|Romance', 'Thriller', 'Sci-Fi']*20
    })
    
    with open('data/movies.dat', 'w', encoding='latin-1') as f:
        for _, row in movies.iterrows():
            f.write(f"{row['movieId']}::{row['title']}::{row['genres']}\n")
            
    # Create ratings.dat
    ratings = []
    for user_id in range(1, 21): # 20 users
        for _ in range(20): # 20 ratings each
            movie_id = np.random.randint(1, 101)
            rating = np.random.randint(1, 6)
            timestamp = int(time.time())
            ratings.append([user_id, movie_id, rating, timestamp])
            
    with open('data/ratings.dat', 'w', encoding='latin-1') as f:
        for r in ratings:
            f.write(f"{r[0]}::{r[1]}::{r[2]}::{r[3]}\n")
            
    print("Dummy data created in data/")

def evaluate_recommender():
    print("--- Starting Recommendation System Benchmark ---")
    
    # Check for data
    if not os.path.exists('data/movies.dat'):
        print("Data not found.")
        create_dummy_data()
        USING_DUMMY = True
    else:
        USING_DUMMY = False
    
    # 1. Load Data
    print("Loading data...")
    movies, ratings = load_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")

    # 2. Train/Test Split (Time-based or Random)
    # We'll use a simple random split for this benchmark
    print("Splitting data for evaluation...")
    test_ratio = 0.2
    # Sort by timestamp to emulate real-world scenario (past -> future)
    ratings = ratings.sort_values('timestamp')
    split_index = int(len(ratings) * (1 - test_ratio))
    
    train_ratings = ratings.iloc[:split_index]
    test_ratings = ratings.iloc[split_index:]
    
    # We only care about users in test who are also in train (to have history)
    test_users = test_ratings['userId'].unique()
    train_users = train_ratings['userId'].unique()
    valid_test_users = np.intersect1d(test_users, train_users)
    
    print(f"Training on {len(train_ratings)} ratings.")
    print(f"Evaluation on {len(valid_test_users)} users (subset of test set).")

    # 3. Initialize and Train Model
    print("Initializing and training model...")
    start_train = time.time()
    recommender = HybridRecommender(movies, train_ratings)
    recommender.train()
    train_time = time.time() - start_train
    print(f"Training Time: {train_time:.4f} seconds")

    # 4. Evaluate Precision@K and Latency
    k = 10
    precisions = []
    latencies = []
    
    # Sample a subset of users to keep benchmark fast
    sample_users = np.random.choice(valid_test_users, size=min(50, len(valid_test_users)), replace=False)
    
    print(f"Benchmarking on {len(sample_users)} sampled users...")
    
    for user_id in sample_users:
        # Get Ground Truth: Movies user rated highly (>=4) in test set
        user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
        ground_truth = user_test_ratings[user_test_ratings['rating'] >= 4]['movieId'].values
        
        if len(ground_truth) == 0:
            continue
            
        # Get Recommendations
        start_query = time.time()
        
        # Strategy: Use last liked movie from training set as pivot
        last_liked_movie_id = train_ratings[(train_ratings['userId'] == user_id) & (train_ratings['rating'] >= 4)]['movieId']
        if len(last_liked_movie_id) == 0:
             # Fallback to just last movie
             last_liked_movie_id = train_ratings[train_ratings['userId'] == user_id]['movieId']
        
        if len(last_liked_movie_id) > 0:
            pivot_movie_id = last_liked_movie_id.iloc[-1] # Last one
            
            # Ensure pivot movie exists in movies df (it should)
            if pivot_movie_id in movies['movieId'].values:
                recs = recommender.recommend(user_id=user_id, movie_id=pivot_movie_id, top_n=k)
                query_time = time.time() - start_query
                latencies.append(query_time)
                
                recommended_ids = recs['hybrid']['movieId'].values
                
                # precision = (hits) / k
                hits = len(set(recommended_ids).intersection(set(ground_truth)))
                precisions.append(hits / k)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print("\n--- Results ---")
    if USING_DUMMY:
        print("(Use with caution: Generated Synthetic Data)")
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Query Latency: {avg_latency:.4f} seconds")
    print(f"Total Training Time: {train_time:.4f} seconds")

if __name__ == "__main__":
    evaluate_recommender()
