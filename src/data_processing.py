import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_text_for_word2vec(text):
    doc = nlp(text)
    tokens = [token.lower_ for token in doc if not token.is_space]
    return tokens.lemma_

def clean_text_for_inference(text):
    doc = nlp(text)

    custom_stopwords = ['cup','wash','clean','nicely','dice','raw','gram','indian','teaspoon','tablespoon','chop','inch','dry','fresh','grate','powder','salt','chilli','masala','green','turmeric','oil','water','finely','jeera','cumin','taste','elaichi','clove','garam','hing','sugar','haldi','mustard','cut','paste','red','black','ingredient','pinch','slice']
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num and token.lemma_.lower() not in custom_stopwords
    ]
    return " ".join(tokens)

def preprocess_dataframe(df):
    # Drop exact duplicate rows
    df = df.drop_duplicates()

    df['ingredients'] = df['ingredients'].apply(lambda x: re.sub(r'[\n\t]+', ' ', re.sub(r'\s+', ' ', x)).strip())
    df['prep_time'] = df['prep_time'].fillna('Within in 40 M')
    df['course'] = df['course'].fillna('Lunch')
    df['diet'] = df['diet'].fillna('Unknown')
    df['cuisine'] = df['cuisine'].fillna('Unknown')

    # Combine relevant columns for Word2Vec training
    df['name_combined'] = (df['name'] + df['description'] + df['cuisine'] +
                           df['course'] + df['diet'] + df['prep_time'] +
                           df['ingredients'] + df['instructions']).apply(str.lower)

    # Apply the tokenization for Word2Vec
    df['name_combined_tokens'] = df['name_combined'].apply(clean_text_for_word2vec)

    # Apply the cleaning for similarity search ('v' column)
    df['v_cleaned'] = (df['name'] + df['ingredients']).apply(clean_text_for_inference)
    
    return df