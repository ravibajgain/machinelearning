import streamlit as st
import matplotlib.pyplot as plt 
import streamlit.components.v1 as components
import codecs
import pandas as pd

# Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

# For data split 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Algorithims
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#from analysis.algo1 import *

st.set_option('deprecation.showPyplotGlobalUse', False)
selection = st.sidebar.selectbox("Select items",("blood Preasure", "heart beat", "cholestrol level"))

healthitem=["AGE", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
choice =st.sidebar.selectbox("Health Indicator", healthitem)

user = st.radio("select user type",("Admin","Client"))
if user =="Client":
    st.subheader("AGE")
    age= st.text_input("enter your age", max_chars =20)
    st.subheader("sex")
    sex = st.text_input ("please enter 1 if you are male, 0 if you are female",max_chars = 20)
    st.subheader("trestbps")
    trestbps = st.text_input(" enter resting blood pressure in mm Hg ", max_chars =20)
    st.subheader("chol")
    chol = st.text_input (" Please enter erum cholestral in mg/dl", max_chars = 20)
    st.subheader("fbs")
    fbs = st.text_input( " please enter fasting sugar  sugar > 120 mg/dl  ( 1 = true ; 0 = false)", max_chars = 20)
    st.subheader("restecg")
    #restecg = st.text_input(" resting electrocardiographic results ( 0 = normal; 1 = having ST-T; 2 = hypertrophy", max_chars = 20)
    restecg = st.slider("what is your result ", 0,2)
    st.write(restecg)
    st.subheader("thalacg")
    thalacg = st.text_input(" please enter value for thalacg ( maximum heart rate achieved", max_chars = 20)
    st.subheader("exang")
    exang = st.text_input("please enter exercise induced angina ( 0 = no 1 = yes", max_chars = 20)
    st.subheader("oldpeak")
    oldpeak = st.text_input("please enter value for (ST depression induced by exercise which is relative to rest", max_chars = 20)
    st.subheader("slope")
    slope = st.text_input(" enter value for slope ( slope of peak exercise ST segment (1 = upsoloping, 2 = flat, 3 = downsloping), max_chars = 20")
    st.subheader("ca", )
    ca = st.text_input(" please enter value for number of major vessels ( 0 - 3 )  colored by flourosopy", max_chars = 20)
    st.subheader("thal")
    thal = st.text_input('please enter value ( 3 = normal, 6 = fixed defect 7 = reversable defect', max_chars = 20)
else:
    st.header("select your data")
    df = pd.read_csv("data_sets/heart2.csv")
    status = st.radio( "What is your status",("show Dataset", "describe dataset"))
    if status == "show Dataset":
        st.dataframe(df)
    else:
        describe = df.describe()
        st.dataframe(describe)
        


# Sidebar
#st.sidebar.header(selection)


# Body st.pyplot((data['data_histo']))



#st.checkbox("Data, Dataframe")


'''
age1 = df['age']
st.line_chart(age1,df, data=None, width=0, height=0, use_container_width=True)
'''

if st.button("Data"):

    st.dataframe(data['data_set'])



if st.button("Data Describe"):
    st.dataframe(data['data_describe'])
    
if st.button("Correlation Matrix"):
    st.pyplot((data['data_histo']))
        #st.image("/Users/iamsan/Desktop/py_data/images/plot1.png")
if st.button("Histogram"):
    st.image("/Users/iamsan/Desktop/py_data/images/histo.png", width=850)
    


