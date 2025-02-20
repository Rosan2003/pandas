import numpy as np
import pandas as pd

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def recommend_players(target_player, top_n=5):

    df = pd.read_csv('./data.csv')

    target_vector = np.array([target_player['age'], target_player['rating'], target_player['gender']])
    
    similarity_scores = []

    for _, row in df.iterrows():
        player_vector = np.array([row['age'], row['rating'], row['gender']])
        similarity = cosine_similarity(target_vector, player_vector)
        similarity_scores.append((row['name'], similarity))
    
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    return similarity_scores[:top_n]


target_player = {"age": 20, "rating": 2.8, "gender":2}


top_similar_players = recommend_players(target_player, top_n=5)

print("Top Recommended Players:")
for player, similarity in top_similar_players:
    print(f"{player} (Similarity: {similarity:.4f})")
