import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
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
df = df.iloc[:3000]

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
        st.session_state['summarization'] = False
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
            # Filter questions containing the keyword
            filtered_df = df[df['question'].str.contains(keyword, case=False, na=False)]
            column1, column2 = st.columns([1,1])
            column1.header(keyword)
               
            keyword_synsets = get_synsets(keyword)
            focus_area_synsets = {area: get_synsets(area) for area in focus_areas}
                      
            # Calculate similarities
            similarities = {}
            for area, synsets in focus_area_synsets.items():
                similarity = compute_similarity(keyword_synsets, synsets)
                similarities[area] = similarity
            
            
            # Find the focus area with the highest similarity
            best_match_focus_area = max(similarities, key=similarities.get)
            
            column2.header(f"Focus Area - {best_match_focus_area}")
            focus_area = best_match_focus_area
            
            if focus_area:
              
                # Filter answers by the selected focus area
                filtered_df = df[df['focus_area'].str.lower().str.contains(focus_area, case=False, na=False)]
                
                if not filtered_df.empty:
                    # Concatenate all answers into a single text
                    all_answers_text = " ".join(filtered_df['answer'].dropna().tolist())
                    summary = summarize_answer(all_answers_text)
                    column1.markdown(summary) 

                    
                    # Generate word cloud of content of summary of answers
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_answers_text)
                    
                    # Display the word cloud
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    column2.pyplot(plt)

                    st.session_state['summarization'] = True


                    
                else:
                    
                    column2.write("No matching focus areas found.")
        
        else:
            st.write("Please enter a keyword to search.")


        selected_question = st.selectbox("You may also want to know:", filtered_df['question'].tolist(), index=None)
        if selected_question:  
            # Display the selected question and its answer
            st.write("Selected Question:", selected_question)
            selected_answer = filtered_df[filtered_df['question'] == selected_question]['answer'].values[0]
            st.write("Answer:", selected_answer)
            st.write("No matching questions found.")
            
        if summary:
            doctor_recommendation = specialty_doctor_recommendation(summary)
            column2.markdown(doctor_recommendation)

        
        tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
        data = np.random.randn(10, 1)
        
        tab1.subheader("A tab with a chart")
        tab1.line_chart(data)
        
        tab2.subheader("A tab with the data")
        tab2.write(data)

   


if my_page == 'Keyword Q':
    
    ### KEYWORD
    st.title("Search Questions by Keyword")
    
    keyword = st.text_input("Enter a keyword to search:")
    if keyword:
        # Filter questions containing the keyword
        filtered_df = df[df['question'].str.contains(keyword, case=False, na=False)]
        
        if not filtered_df.empty:
            # Create a dropdown with matching questions
            selected_question = st.selectbox("Select a Question:", filtered_df['question'].tolist())
            
            # Display the selected question and its answer
            st.write("Selected Question:", selected_question)
            selected_answer = filtered_df[filtered_df['question'] == selected_question]['answer'].values[0]
            st.write("Answer:", selected_answer)
        else:
            st.write("No matching questions found.")
    else:
        st.write("Please enter a keyword to search.")
    
    

    
    
    if keyword:
        # Filter questions containing the keyword
        filtered_df = df[df['question'].str.contains(keyword, case=False, na=False)]
        
        if not filtered_df.empty:
            # Create a dropdown with matching questions
            selected_question = st.selectbox("Select a Question:", filtered_df['question'].tolist())
            
            # Display the selected question and its answer
            st.write("Selected Question:", selected_question)
            selected_answer = filtered_df[filtered_df['question'] == selected_question]['answer'].values[0]
            st.write("Answer:", selected_answer)
        else:
            st.write("No matching questions found.")
    else:
        st.write("Please enter a keyword to search.")
    
    # TOKENIZATION
    st.title("Word Cloud of Answers by Focus Area")
    
    
    
    
    summary_button = st.button('Summarize')
    
    
    # User input for focus area
    # Make user input box and dropdown side by side
    focus_area_input = st.text_input("Enter a focus area to search:").strip().lower() # input is not case sensitive
    focus_area_dropdown = st.selectbox("Or select a focus area:", [''] + df['focus_area'].str.lower().unique().tolist())
    
    # Include option of wanting to select specific focuse_area question
    # Add generate word cloud button
    
    # Determine the focus area to use
    focus_area = focus_area_input if focus_area_input else focus_area_dropdown
    
    if focus_area:
        # Filter answers by the selected focus area
        filtered_df = df[df['focus_area'].str.lower().str.contains(focus_area, case=False, na=False)]
        
        if not filtered_df.empty:
            # Concatenate all answers into a single text
            all_answers_text = " ".join(filtered_df['answer'].dropna().tolist())
            
            # Generate word cloud of content of summary of answers
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_answers_text)
            
            # Display the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("No matching focus areas found.")
    else:
        st.write("Please enter or select a focus area to search.")
        
    ### TEXT SUMMARIZATION
    # Input: from Answers column or dropdown of focus_area
    # Output: Summary of description, symptoms, treaments of selected focus_area
    
    st.title("Text Summarization")
    

    
    ### KEYWORD EXTRACTION
    st.title("Keyword Extraction")
    
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
    
    title = st.selectbox(
        'Select medical question', df['question'], index=None
    )
    
    if title:
        health = df[df['question']==title].iloc[0]
    
        st.header(f"[{health['question']}]({health['source']})")
        st.caption(f"Focus Area: {health['focus_area']}")
    
        st.caption('TOP KEYWORDS')
        top_keywords = extract_keywords(health['answer'])
    
        highlighted_keywords = ""
        for i, keyword in enumerate(top_keywords):
            highlighted_keywords += f"<span style='background-color:#808080;padding: 5px; border-radius: 5px; margin-right: 5px;'>{keyword}</span>"
    
        st.markdown(highlighted_keywords, unsafe_allow_html=True) 
    
        st.subheader('Full Medical Information')
        st.write(health['answer'])
    
    
