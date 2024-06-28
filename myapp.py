import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from sentence_transformers import SentenceTransformer, util
import openai
from openai import OpenAI
from wordcloud import WordCloud
import subprocess
import time
import numpy as np
# Custom function to download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

###
api_key = st.secrets['api_key']
client = OpenAI(api_key=api_key)
SKLLMConfig.set_openai_key(api_key)

st.set_page_config(layout='wide')

my_page = st.sidebar.radio('Page Navigation', ['MedInfoHub', 'About MedQuAd Dataset', 'Keyword Q'])

# DATA SET
df = pd.read_csv('data/medquad.csv')
# df = df.iloc[:3000]

# Define your focus areas
focus_areas = df['focus_area'].str.lower().unique().tolist()

def generate_response(focus_area, prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system',
             'content':
             f"Perform the specified tasks based on this focus area:\n\n{focus_area}"},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content

def summarize_answer(focus_area):
    prompt = f'Summarize the answer in easy to understand terms and words'
    summary = generate_response(focus_area, prompt)
    return summary

def generate_response(summary, prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system',
             'content':
             f"Perform the specified tasks based on this summary:\n\n{summary}"},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content

def specialty_doctor_recommendation(summary):
    prompt = f'Which specialty doctor should I consult?:\n\n{summary}'
    doctor_recommendation = generate_response(summary, prompt)
    return doctor_recommendation

def initializing():
        msg = st.toast('Getting Ready...')
        time.sleep(1)
        msg.toast('Initializing...')
        time.sleep(1)
        msg.toast('Ready!', icon = "🥞")
        status = 1

# FOR SEMANTIC SIMILARITIES MATCHING
def get_synsets(text):
        tokens = word_tokenize(text)
        synsets = [wn.synsets(token) for token in tokens]
        synsets = [item for sublist in synsets for item in sublist]  # Flatten list
        return synsets
# Compute similarity
def compute_similarity(synsets1, synsets2):
    max_similarity = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity and similarity > max_similarity:
                max_similarity = similarity
    return max_similarity

def search_keyword(keyword, text_list):
    focus_area_embeddings = np.load('data/focus_area_embeddings.npy') # load

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    keyword_embedding = model.encode(keyword)
    similarities = util.pytorch_cos_sim(keyword_embedding, focus_area_embeddings)

    best_match_index = similarities.argmax()

    best_match_focus_area = text_list[int(best_match_index)]

    return best_match_focus_area


def process_keyword(keyword, df, best_match_focus_area):
    column1, column2 = st.columns([1,1])
    column1.header(keyword)
    # keyword_synsets = get_synsets(keyword)
    # focus_area_synsets = {area: get_synsets(area) for area in focus_areas}

    # # Calculate similarities
    # similarities = {}
    # for area, synsets in focus_area_synsets.items():
    #     similarity = compute_similarity(keyword_synsets, synsets)
    #     similarities[area] = similarity


    # # Find the focus area with the highest similarity
    # best_match_focus_area = max(similarities, key=similarities.get)

    column2.header(f"Focus Area - {best_match_focus_area}")
    focus_area = best_match_focus_area

    if focus_area:

        # Filter answers by the selected focus area
        filtered_df = df[df['focus_area'].str.lower().str.contains(focus_area, case=False, na=False)]

        if not filtered_df.empty:
            # Concatenate all answers into a single text
            all_answers_text = " ".join(filtered_df['answer'].dropna().tolist())
            # summary = summarize_answer(all_answers_text)
            # st.session_state['summary'] = summary
            # column1.markdown(summary)
            summary = all_answers_text

            if summary:
                # health = df[df['question']==title].iloc[0]

                # st.header(f"[{health['question']}]({health['source']})")
                # st.caption(f"Focus Area: {health['focus_area']}")

                st.caption('TOP KEYWORDS')
                top_keywords = extract_keywords(all_answers_text)

                highlighted_keywords = ""
                for i, keyword in enumerate(top_keywords):
                    highlighted_keywords += f"<span style='background-color:#808080;padding: 5px; border-radius: 5px; margin-right: 5px;'>{keyword}</span>"

                st.markdown(highlighted_keywords, unsafe_allow_html=True)

                # st.subheader('Full Medical Information')
                # st.write(health['answer'])


            column1.markdown(summary)
            # Generate word cloud of content of summary of answers
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_answers_text)
            st.session_state['wordcloud'] = wordcloud

            # Display the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            column2.pyplot(plt)

        else:
            st.session_state['summary'] = "No matching focus areas found."
            st.session_state['wordcloud'] = None

    return focus_area, summary, filtered_df

def select_questions(filtered_df):
    selected_question = st.selectbox("You may also want to know:", filtered_df['question'].tolist(), index=None)
    if selected_question:
        # Display the selected question and its answer
        st.write("Selected Question:", selected_question)
        selected_answer = filtered_df[filtered_df['question'] == selected_question]['answer'].values[0]
        st.write("Answer:", selected_answer)
    else:
        st.write("No matching questions found.")
def extract_keywords(text):
        system_prompt = 'You are a health professional assistant tasked to extract keywords from medical question answering dataset.'

        main_prompt = """
        ###TASK###

    Extract the five most crucial keywords from the medical question answering dataset.
    Extracted keywords must be listed in a comma-separated list.
    Example: Glaucoma, optic nerve, vision loss, eye, treatment

        ###HEALTH###
        """

        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{main_prompt}\n{text}"}
                ]
            )
            top_keywords = response.choices[0].message.content
            return [kw.strip() for kw in top_keywords.split(',')]

        except:
            return []



if my_page == 'MedInfoHub':

    st.image('data/MIH.png')
    col1, col2 = st.columns([1,1])
    col1.image('data/art.png')
    col2.write("")
    col2.write("")
    col2.write("MedInfoHub is a comprehensive healthcare app designed to provide accessible medical information to patients and healthcare providers. Leveraging the power of the MedQuAD dataset* and advanced AI, MedInfoHub offers reliable answers to medical questions, supports telemedicine consultations, and enhances public health literacy. Whether you’re a patient seeking to understand your health better or a healthcare provider in need of quick, reliable information, MedInfoHub is your go-to resource for trusted medical knowledge.")
    col2.write("*The MedQuAD dataset aggregates content from reputable sources like the National Institutes of Health (NIH), National Library of Medicine (NLM), and other authoritative medical organizations.")
    col2.write("Press the 'Activate MedInfoHub' Button to begin exploring MedInfoHub.")

    # Displaying the button with custom style
    # col_start1, col_start2, col_start3 = st.columns([1,1,1])
    on = col2.toggle("Activate MedInfoHub")

    # START SESSION
    if not on:
        st.session_state['initialized'] = False

    elif on:
          # Check if initializing has been run
        if 'initialized' not in st.session_state:
            st.session_state['initialized'] = False

        if not st.session_state['initialized']:
            initializing()
            st.session_state['initialized'] = True

        # ENTER KEYWORD FOR SEMANTIC SIMILARITIES MATCHING WITH FOCUS AREA
        st.subheader("Search Keyword Focus Area")
        keyword = st.text_input("Enter a keyword to search:")
        if keyword:

            
            def search_by():
                choose_method = st.selectbox(
                        "Choose Keyword Search Method",
                        ("Exact Word", "Best Match"))
                return choose_method

            if choose_method == 'Exact Word':
                filtered_df = df[df['focus_area'].str.lower().str.contains(focus_area, case=False, na=False)]
                focus_area_choose = st.selectbox(
                        "Choose (1) from matched Focus Area/s",
                        filtered_df["focus_area"].str.lower().unique().tolist())
                
                focus_area, summary, filtered_df = process_keyword(keyword, df, focus_area_choose)
                select_questions(filtered_df)
                
            else:
                # # Filter questions containing the keyword
                # filtered_df = df[df['question'].str.contains(keyword, case=False, na=False)]
                best_match_focus_area = search_keyword(keyword, df['focus_area'])
                focus_area, summary, filtered_df = process_keyword(keyword, df, best_match_focus_area)
                select_questions(filtered_df)
            if summary:
                doctor_recommendation = specialty_doctor_recommendation(summary)
                column2.markdown(doctor_recommendation)
        else:
            st.write("Please enter a keyword to search.")











