import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
#===========================================================================================================================================
                                                      # Let's Begin !
#Config:
st.set_page_config(
    page_title="ModelForge",
    layout="wide",  # This sets full-width layout
    initial_sidebar_state="expanded"  # or "collapsed"
)
#Title:
st.title('ModelForge AI — Your Smart ML Builder')
#Begin :
st.markdown("""
**ModelForge** automates your ML workflow — upload data, choose a task, and let the magic happen.  
Train, evaluate, and predict — all from one clean, smart interface.
""")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx",'xls'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        readed_data = pd.read_csv(uploaded_file)
    else:
        readed_data = pd.read_excel(uploaded_file)
    TARGET_COLUMN =  '--TARGET COLUMN--'   
    columns_list = [col for col in readed_data.columns]
    columns_list.insert(0,TARGET_COLUMN)

    #Asking the target column :
    target = st.selectbox('Select your `TARGET COLUMN` :',options=columns_list)  
    if target==TARGET_COLUMN:
        st.warning('Please select the target column above  ^-^ ')
    readed_data.to_csv(os.path.join('artifact','raw_data.csv'))
    with open(os.path.join('artifact','target.txt'),'w') as fp :
        fp.write(target)
    st.write("Preview of uploaded data:")
    st.dataframe(readed_data)

else:
    st.warning("Please upload a file to proceed.")
