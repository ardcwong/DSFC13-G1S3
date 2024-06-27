import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from skllm.config import SKLLMConfig
# from skllm.models.gpt.text2text.summarization import GPTSummarizer
import openai
from openai import OpenAI
from wordcloud import WordCloud
import subprocess

import spacy

# Install spaCy model
spacy.cli.download("en_core_web_sm")


# Install spaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
