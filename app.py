import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import openai
from openai import OpenAI
from wordcloud import WordCloud

# api_key = st.secrets["api_key"]
# SKLLMConfig.set_openai_key(api_key)
# client = OpenAI(api_key=api_key)

# df = pd.read_csv('data/medquad.csv')
### KEYWORD
st.title("Search Questions by Keyword")

# keyword = st.text_input("Enter a keyword to search:")

# if keyword:
#     # Filter questions containing the keyword
#     filtered_df = df[df['question'].str.contains(keyword, case=False, na=False)]
    
#     if not filtered_df.empty:
#         # Create a dropdown with matching questions
#         selected_question = st.selectbox("Select a Question:", filtered_df['question'].tolist())
        
#         # Display the selected question and its answer
#         st.write("Selected Question:", selected_question)
#         selected_answer = filtered_df[filtered_df['question'] == selected_question]['answer'].values[0]
#         st.write("Answer:", selected_answer)
#     else:
#         st.write("No matching questions found.")
# else:
#     st.write("Please enter a keyword to search.")

# # TOKENIZATION
# st.title("Word Cloud of Answers by Focus Area")

# # User input for focus area
# # Make user input box and dropdown side by side
# focus_area_input = st.text_input("Enter a focus area to search:").strip().lower() # input is not case sensitive
# focus_area_dropdown = st.selectbox("Or select a focus area:", [''] + df['focus_area'].str.lower().unique().tolist())

# # Include option of wanting to select specific focuse_area question
# # Add generate word cloud button

# # Determine the focus area to use
# focus_area = focus_area_input if focus_area_input else focus_area_dropdown

# if focus_area:
#     # Filter answers by the selected focus area
#     filtered_df = df[df['focus_area'].str.lower().str.contains(focus_area, case=False, na=False)]
    
#     if not filtered_df.empty:
#         # Concatenate all answers into a single text
#         all_answers_text = " ".join(filtered_df['answer'].dropna().tolist())
        
#         # Generate word cloud of content of summary of answers
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_answers_text)
        
#         # Display the word cloud
#         plt.figure(figsize=(10, 5))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis('off')
#         st.pyplot(plt)
#     else:
#         st.write("No matching focus areas found.")
# else:
#     st.write("Please enter or select a focus area to search.")
    
# ### TEXT SUMMARIZATION
# # Input: from Answers column or dropdown of focus_area
# # Output: Summary of description, symptoms, treaments of selected focus_area

# st.title("Text Summarization")

# def generate_response(focus_area, prompt):
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

# def summarize_answer(focus_area):
#     prompt = f'Summarize the answer in easy to understand terms and words'
#     summary = generate_response(focus_area, prompt)
#     return summary

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
#     doctor_recommendation = generate_response(summary, prompt)
#     return doctor_recommendation

# if focus_area:
#     summary = summarize_answer(focus_area)
#     st.markdown(summary) 

#     doctor_recommendation = specialty_doctor_recommendation(summary)
#     st.markdown(doctor_recommendation)

# ### KEYWORD EXTRACTION
# st.title("Keyword Extraction")

# def extract_keywords(text):
#     system_prompt = 'You are a health professional assistant tasked to extract keywords from medical question answering dataset.'

#     main_prompt = """
#     ###TASK###
    
# Extract the five most crucial keywords from the medical question answering dataset. 
# Extracted keywords must be listed in a comma-separated list. 
# Example: Glaucoma, optic nerve, vision loss, eye, treatment

#     ###HEALTH###
#     """

#     try:
#         response = client.chat.completions.create(
#             model='gpt-3.5-turbo', 
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"{main_prompt}\n{text}"}
#             ]
#         )
#         top_keywords = response.choices[0].message.content
#         return [kw.strip() for kw in top_keywords.split(',')]
        
#     except:
#         return []

# title = st.selectbox(
#     'Select medical question', df['question'], index=None
# )

# if title:
#     health = df[df['question']==title].iloc[0]

#     st.header(f"[{health['question']}]({health['source']})")
#     st.caption(f"Focus Area: {health['focus_area']}")

#     st.caption('TOP KEYWORDS')
#     top_keywords = extract_keywords(health['answer'])

#     highlighted_keywords = ""
#     for i, keyword in enumerate(top_keywords):
#         highlighted_keywords += f"<span style='background-color:#808080;padding: 5px; border-radius: 5px; margin-right: 5px;'>{keyword}</span>"

#     st.markdown(highlighted_keywords, unsafe_allow_html=True) 

#     st.subheader('Full Medical Information')
#     st.write(health['answer'])
