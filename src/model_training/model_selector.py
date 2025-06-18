import streamlit as st

def select_task_type():
    return st.radio("Select Task Type:", ['Classification', 'Regression'])

def select_model():
    return st.selectbox(
        "Select Model:",
        ['Linear Regression', 'Logistic Regression', 'Decision Trees', 'Random Forest',
         'Support Vector Machine', 'Naive Bayes', 'KNN']
    )