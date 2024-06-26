import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("No data found in the file.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing the file.")
        st.stop()

# Function to clean data
def clean_data(data, drop_threshold=0.5):
    if drop_threshold > 0:
        data = data.dropna(thresh=data.shape[0] * drop_threshold, axis=1)
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

# Function to preprocess data
def preprocess_data(data):
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = pd.factorize(data[column])[0]
    return data

# Function to perform EDA
def explore_data(data, compare_columns=None):
    st.subheader("Data Description")
    st.write(data.describe())
    
    numeric_cols = data.select_dtypes(include=np.number).columns
    
    fig, ax = plt.subplots()
    data[numeric_cols].hist(ax=ax, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.tight_layout()
    st.pyplot(fig)
    
    if len(numeric_cols) > 1:
        sns.pairplot(data[numeric_cols])
        plt.suptitle('Pairplot of Numeric Columns', y=1.02)
        plt.tight_layout()
        st.pyplot()
    
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    
    num_cols = data.shape[1]
    num_rows = (num_cols - 1) // 3 + 1
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    
    for i, column in enumerate(data.columns):
        if pd.api.types.is_numeric_dtype(data[column]):
            data.boxplot(column=column, ax=axes.flatten()[i], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
            axes.flatten()[i].set_title(f'Boxplot - {column}')
    
    for j in range(i + 1, num_rows * 3):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    if compare_columns and 2 <= len(compare_columns) <= 10:
        for col in compare_columns:
            if col in data.columns:
                if col in numeric_cols:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    data[col].plot.hist(bins=20, ax=ax, color='salmon', edgecolor='black')
                    ax.set_title(f'Histogram - {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                else:
                    st.warning(f"Column '{col}' is not numeric and cannot be compared.")
            else:
                st.warning(f"Column '{col}' not found in the dataset.")
    elif compare_columns:
        st.warning("Please select between 2 to 10 columns for comparison.")
    else:
        st.info("No valid columns selected for comparison.")

# Streamlit app with enhanced UI and error handling
def main():
    st.set_page_config(page_title="Dynamic Data Analysis App", layout="wide")
    st.title('Dynamic Data Analysis App')
    
    st.markdown("""
    Welcome to the Dynamic Data Analysis App! Upload a CSV file to explore and analyze your data. 
    You can clean the data, perform exploratory data analysis (EDA), and compare multiple columns.
    """)
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            st.subheader("Columns in the dataset:")
            st.write(list(data.columns))
            
            st.write("Sample data:")
            st.write(data.head(10))
            
            clean_choice = st.radio("Do you want to clean the data?", ('Yes', 'No'))
            if clean_choice == 'Yes':
                drop_threshold = st.slider("Enter the drop threshold for columns with missing values (0 to 1):", 0.0, 1.0, 0.5)
                data = clean_data(data, drop_threshold=drop_threshold)
            
            data = preprocess_data(data)
            
            compare_choice = st.radio("Do you want to compare multiple columns?", ('Yes', 'No'))
            if compare_choice == 'Yes':
                num_compare = st.number_input("How many columns do you want to compare (2 to 10)?", min_value=2, max_value=10)
                if num_compare >= 2:
                    compare_columns = []
                    for i in range(num_compare):
                        col_options = [col for col in data.columns if col not in compare_columns]
                        if col_options:
                            col = st.selectbox(f"Select column {i + 1}:", col_options)
                            compare_columns.append(col)
                        else:
                            st.warning("No more columns available for comparison.")
                            break
                    if len(compare_columns) >= 2:
                        explore_data(data, compare_columns=compare_columns)
                else:
                    st.warning("Please enter a number between 2 and 10 for comparison.")
            else:
                explore_data(data)
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
