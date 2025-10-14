import numpy as np
from gensim.models import Word2Vec

import os
os.makedirs('models', exist_ok = True)

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1):
    # Training Word2Vec Model
    model = Word2Vec(
        sentences = sentences,
        vector_size = vector_size,
        window = window,
        min_count = min_count,
        workers = 4
    )
    model.train(sentences, total_example = len(sentences), epochs=10)
    return model

def get_avg_vector(tokens, model):
    # Calculating the average vector for a list of tokens.
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# --- New: Add execution logic when the script is run directly ---
if __name__ == "__main__":
    from data_processing import load_data, preprocess_dataframe

    # Loading and pre-processing the data
    print("Loading and preprocessing data...")
    df = load_data('data/cuisine_updated.csv')
    df = preprocess_dataframe(df)

    # Training model
    print("Training Word2Vec model...")
    sentences = df['name_combined_tokens']
    model = train_word2vec_model(sentences)

    # Save the trained model to a file
    model_path = 'models/word2vec.model'
    model.save(model_path)
    print(f"Model saved to {model_path}")