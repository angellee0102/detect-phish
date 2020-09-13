import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from nltk.tokenize import sent_tokenize, word_tokenize
import wordninja
import re
from nltk.stem import PorterStemmer
porter=PorterStemmer()

st.title('Detect phishing emails')

# DATE_COLUMN = 'date/time'
# DATA_URL = ('/Users/lee/Documents/Documents/nctu2020/專題/content_20200906.csv')

# def load_data():
#     data = pd.read_csv(DATA_URL)
#     return data

# Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
# data = load_data()
# Notify the reader that the data was successfully loaded.
# data_load_state.text('Loading data...done!')
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data.head(50))

# num_bins = 100
# n, bins, patches = plt.hist(df_content['lengthDivideWords'], num_bins, facecolor='blue', alpha=0.5)
# n, bins, patches = plt.hist(data[data['phish']==0]['lengthDivideWords'], num_bins, facecolor='green', alpha=0.5)
# n, bins, patches = plt.hist(data[data['phish']==1]['lengthDivideWords'], num_bins, facecolor='red', alpha=0.5)
# plt.xlabel('lengthDivideWords')
# plt.suptitle('lengthDivideWords')
# plt.ylabel('count')

# plt.show()
# st.pyplot()
# functions
def stemSentence(sentence):
    sentence_words=word_tokenize(sentence)
    combined_words=''
    for word in sentence_words:
        combined_words+= ' '+(porter.stem(word))
    return combined_words

def listToString(s):  
    str1 = ""  
    for ele in s:  
        str1 += ele  +' ' 
    return str1

def remove_urls (vTEXT):
    vTEXT = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", vTEXT, flags=re.MULTILINE)
    return(vTEXT)

def oneTextTotfidf(input_content,tfidf_transformer):
    # segmented
    segmented=listToString(wordninja.split(remove_urls(input_content)))
    # and stemmed
    stemmed=stemSentence(segmented)
    count_vector=cv.transform([input_content])
    tf_idf_vector=tfidf_transformer.transform(count_vector)
    df_tf_idf_vector=pd.DataFrame(tf_idf_vector.toarray())
    return df_tf_idf_vector

# tfidf model
loaded_model = pickle.load(open('stemmed_tfidf_xgboost_model_0913.sav', 'rb'))
loaded_transformer = pickle.load(open('stemmed_tfidf_transformer.sav', 'rb'))
cv=pickle.load(open('cv.sav','rb'))

text='hello there how have you been lately i really hope you are having fun and healthy'




inputContent = st.text_input('Email Content:', 'type here')
st.write('The email content is: "', inputContent,'"')
X_test_1=oneTextTotfidf(inputContent,loaded_transformer)
result=loaded_model.predict(X_test_1)[0]
if result ==0:
    st.write('Prediction: Not phish')
if result ==1:
    st.write('Prediction: PHISH')
    