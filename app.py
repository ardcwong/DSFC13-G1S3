import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import openai
import joblib

from skllm.config import SKLLMConfig
#from skllm.models.gpt.text2text.summarization import GPTSummarizer

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
import spacy
import contractions
import string

from wordcloud import WordCloud

#st.set_page_config(layout="wide") # Page expands to full width


st.image('s4g4-waits-moodguard-banner.png')
st.write("MoodGuard is not a replacement for professional mental health guidance; rather, it is an exploration of how LLMs can contribute to mental health services. Seeking assistance from professionals is highly recommended")
st.write("Instructions: Upload the input file through the sidebar and then select the \"Start Analyze Data\" button.")

#--------------------------------------------------------------------------------------------------

import spacy

# Install spaCy model
spacy.cli.download("en_core_web_sm")


# Install spaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Load spaCy model
nlp = spacy.load("en_core_web_sm")



# import streamlit as st

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# #import seaborn as sns

# import time
# import openai
# from openai import OpenAI
# import joblib

# from skllm.config import SKLLMConfig
# from skllm.models.gpt.text2text.summarization import GPTSummarizer
# from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.pipeline import Pipeline

# import nltk
# from nltk.corpus import stopwords
# # import spacy
# # import contractions
# # import string

# from wordcloud import WordCloud

# api_key = open('openaiapikey.txt').read()
# SKLLMConfig.set_openai_key(api_key)
# client = OpenAI(api_key=api_key)



# # Import libraries for nltk preprocessing
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import string

# # Download necessary nltk data
# nltk.download('punkt')  # Downloads the Punkt tokenizer models
# nltk.download('stopwords')  # Downloads the list of stopwords
# nltk.download('wordnet')  # Downloads the WordNet lemmatizer data

# import spacy

# # Install spaCy model
# spacy.cli.download("en_core_web_sm")


# # Install spaCy model
# subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Define your focus areas
# focus_areas = [
#     "data science",
#     "web development",
#     "machine learning",
#     "computer vision",
#     "natural language processing",
#     "software engineering",
#     "cybersecurity"
# ]

# # Function to calculate similarity scores
# def calculate_similarity(keyword):
#     similarity_scores = {}
#     keyword_doc = nlp(keyword)
#     for area in focus_areas:
#         area_doc = nlp(area)
#         similarity_scores[area] = keyword_doc.similarity(area_doc)
#     return similarity_scores

# # Streamlit UI
# def main():
#     st.title('Semantic Similarity Calculator')
#     st.write("Enter a keyword and see its similarity to predefined focus areas.")

#     # User input for keyword
#     keyword = st.text_input('Enter your keyword:')
#     if keyword:
#         similarity_scores = calculate_similarity(keyword)

#         # Display similarity scores
#         st.write("### Similarity Scores:")
#         for area, score in similarity_scores.items():
#             st.write(f"Similarity between '{keyword}' and '{area}': {score}")

#         # Find the best match
#         best_match = max(similarity_scores, key=similarity_scores.get)
#         best_score = similarity_scores[best_match]
#         st.write(f"\n**Best Match:** '{best_match}' with a score of {best_score}")

# if __name__ == "__main__":
#     main()











# st.write("hello")
# answer = st.text_input(
#     label='Input answer',
#     value=''
# )

# def generate_summary(focus_area, prompt):
#     response = client.chat.completions.create(
#         model='gpt-3.5-turbo',
#         messages=[
#             {'role': 'system',
#              'content':
#              f"Perform the specified tasks based on this focus area:\n\n{focus_area}"},
#             {'role': 'user', 'content': prompt}
#         ]
#     )
#     return response.choices[0].message.content

# def summarize_focus_area(focus_area):
#     prompt = f'Summarize focus_area in easy to understand terms and words'
#     summary = generate_summary(answer, prompt)
#     return summary

# if focus_area:
#     summary = summarize_focus_area(focus_area)
#     st.markdown(summary) 
    
# def generate_response(summary, prompt):
#     response = client.chat.completions.create(
#         model='gpt-3.5-turbo',
#         messages=[
#             {'role': 'system',
#              'content':
#              f"Perform the specified tasks based on this summary:\n\n{summary}"},
#             {'role': 'user', 'content': prompt}
#         ]
#     )
#     return response.choices[0].message.content
    
# def specialty_doctor_recommendation(summary):
#     prompt = f'Which specialty doctor should I consult?:\n\n{summary}'
#     doctor_recommendations = generate_response(summary, prompt)
#     return doctor_recommendation

# if focus_area:
#     summary = summarize_answer(answer)
#     st.markdown(summary)
