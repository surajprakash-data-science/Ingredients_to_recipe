## Ingredients To Recipe Recommendation System.
********************************************
This project is a recipe recommendation system that is designed to solve a very common real world problem. It uses a combination of data preprocessing, natural language processing (NLP), and vector similarity to suggest recipes based on a user's entered ingredients. The core of the system is a Word2Vec model trained on a dataset of diverse recipes, which is used to find semantic similarities between ingredients.

This project follows MLOps principles, with a focus on reproducibility, modularity, and automation.

# Project Structure.
******************
The project is organized into a modular structure to separate different parts of the machine learning pipeline:

This file contains the raw dataset: cuisine_updated.csv.

notebooks/: Holds exploratory data analysis (EDA) notebooks, such as recipe_eda.ipynb, for experimentation and visualization.

src/: Contains the core Python scripts for the application:

data_processing.py: Handles data cleaning, preprocessing, and feature engineering.

model_training.py: Manages the training and saving of the Word2Vec model.

inference.py: Contains the logic for making recipe recommendations based on user input.

models/: Stores the trained model artifacts, versioned using DVC.

Dockerfile: Defines the environment and dependencies for building a containerized version of the application.

# Key Technologies.
*****************
Python: The main programming language used.

pandas & scikit-learn: For data manipulation and scientific computing.

spaCy & gensim: For natural language processing and training the Word2Vec model.

Git: For code version control.

DVC (Data Version Control): For versioning the large data and model files.

Docker: For packaging the application and its environment for consistent deployment.

# Getting Started.
****************
Step 1: Clone the Repository:

"git clone https://github.com/your-username/my-recipe-recommender.git
cd my-recipe-recommender"

Step 2: Set up the Environment

Install the required dependencies:
"pip install -r requirements.txt"

Step 3: Download the Data and Model:

This project uses DVC to manage large files. After installing DVC, pull the data and the trained model from the remote storage.

"dvc pull"

# How to Use.
***********
You can run the inference script from your terminal to get recipe recommendations.

"python src/inference.py"

This will prompt you to enter a list of ingredients, and the system will output the top 5 most similar recipes from the dataset.

# Contact.
********
If you have any questions or feedback, please feel free to open an issue or contact me at 
Email: surajprakash612@gmail.com
