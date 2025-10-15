import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import clean_text_for_inference, clean_text_for_word2vec
from gensim.models import Word2Vec

def find_similar_recipies(user_ingredients, df, model):
    """Finds the most similar recipes based on user input."""
    # Clean and vectorize the user input
    clean_input = clean_text_for_word2vec(user_ingredients)
    input_vec = [model.wv[word] for word in clean_input if word in model.wv]

    if not input_vec:
        return "no recipies found with the ingredients entered, please enter other ingredients"
    
    avg_vec = np.mean(input_vec, axis=0)
    input_vector = avg_vec.reshape(1,-1)

    # Calculate similarities and sort
    similarities = []
    for recipe_name, recipe_vec in zip(df['name'], df['avg_vec']):
        recipe_vec = np.array(recipe_vec).reshape(1, -1)
        similarity_score = cosine_similarity(input_vector, recipe_vec)
        similarities.append((recipe_name, similarity_score[0][0]))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:5]

# Example of how you would use this function
if __name__ == "__main__":
    from model_training import get_avg_vector

    #  loading trained model
    trained_model = Word2Vec.load('models/word2vec.model')

    # Load your DataFrame here (replace with your actual loading code)
    
    df = pd.read_csv('A:\ML practice\ingredients to recipie\data\processed_recipes.csv')  # Update the path and filename as needed

    # calculating average vector of ingredients and name combined coloum
    df['avg_vec'] = df['v_cleaned'].apply(lambda x: get_avg_vector(x, trained_model))

    user_input = input('Enter the ingredients:')
    similar_recipes = find_similar_recipies(user_input, df, trained_model)

    print('\n Top 5 similar recipies are: \n')
    for recipe, score in similar_recipes:
        print(f"- {recipe} (Similarity: {score:.4f})")  