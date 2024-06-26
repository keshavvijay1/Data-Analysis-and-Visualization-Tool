import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        exit()
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        exit()
    except pd.errors.ParserError:
        print("Error parsing the file.")
        exit()

def clean_data(data, drop_threshold=0.5):
    """
    Clean data by handling missing values and dropping columns with excessive missing values.
    """
    # Drop columns with more than drop_threshold (default 50%) missing values
    if drop_threshold > 0:
        data = data.dropna(thresh=data.shape[0] * drop_threshold, axis=1)
    
    # Fill missing values with median for numeric columns and mode for categorical columns
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

def preprocess_data(data):
    """
    Preprocess data by encoding categorical variables.
    """
    # Encode categorical variables
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = pd.factorize(data[column])[0]
    
    return data

def explore_data(data, compare_columns=None):
    """
    Perform exploratory data analysis (EDA) on the data.
    """
    print("Data Description:")
    print(data.describe())
    
    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=np.number).columns
    categorical_cols = data.select_dtypes(include='object').columns
    
    # Histograms for numeric columns
    data[numeric_cols].hist(figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    # Pairplot for numeric columns (if more than one numeric column)
    if len(numeric_cols) > 1:
        sns.pairplot(data[numeric_cols])
        plt.suptitle('Pairplot of Numeric Columns', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # Correlation heatmap for numeric columns (if more than one numeric column)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()
    
    # Boxplots for numeric columns
    num_cols = data.shape[1]
    num_rows = (num_cols - 1) // 3 + 1
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    
    for i, column in enumerate(data.columns):
        ax = axes.flatten()[i]
        if pd.api.types.is_numeric_dtype(data[column]):
            data.boxplot(column=column, ax=ax)
            ax.set_title(f'Boxplot - {column}')
            ax.set_ylabel(column)
    
    for j in range(i + 1, num_rows * 3):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()
    plt.show()
    
    # Pie charts for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title(f'Pie Chart - {col}')
        plt.ylabel('')
        plt.show()
    
    # Compare selected columns (if provided)
    if compare_columns and 2 <= len(compare_columns) <= 10:
        for col in compare_columns:
            if col in data.columns:
                if col in categorical_cols:
                    plt.figure(figsize=(8, 6))
                    data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
                    plt.title(f'Pie Chart - {col}')
                    plt.ylabel('')
                    plt.show()
                elif col in numeric_cols:
                    plt.figure(figsize=(8, 6))
                    data[col].plot.hist(bins=20)
                    plt.title(f'Histogram - {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    print(f"Column '{col}' is not numeric or categorical and cannot be compared.")
            else:
                print(f"Column '{col}' not found in the dataset.")
    elif compare_columns:
        print("Please select between 2 to 10 columns for comparison.")
    else:
        print("No valid columns selected for comparison.")

def suggest_columns_for_comparison(data):
    """
    Suggest columns for comparison based on uniqueness and type.
    """
    # Calculate number of unique values for each column
    unique_counts = data.nunique()
    
    # Rank columns based on uniqueness (higher unique counts are better)
    unique_ranking = unique_counts.sort_values(ascending=False).index.tolist()
    
    # Print suggestion based on rank
    print("Suggested columns for comparison (based on uniqueness and type):")
    for idx, col in enumerate(unique_ranking):
        print(f"{idx + 1}. {col} (Unique values: {unique_counts[col]})")

if __name__ == '__main__':
    # Ask user for the path to the CSV dataset
    file_path = input("Enter the path to your CSV dataset: ").strip('"')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print("The specified file does not exist.")
        exit()
    
    # Load the data
    data = load_data(file_path)
    
    # Display columns in the dataset
    print("\nColumns in the dataset:")
    for idx, column in enumerate(data.columns):
        print(f"{idx + 1}. {column}")
    
    # Ask user if they want to clean the data
    clean_choice = input("Do you want to clean the data? (yes/no): ").lower()
    if clean_choice == 'yes':
        try:
            drop_threshold = float(input("Enter the drop threshold for columns with missing values (0 to 1): "))
            if 0 <= drop_threshold <= 1:
                data = clean_data(data, drop_threshold=drop_threshold)
            else:
                print("Invalid threshold entered. Using default cleaning (drop_threshold=0.5).")
                data = clean_data(data)
        except ValueError:
            print("Invalid input. Using default cleaning (drop_threshold=0.5).")
            data = clean_data(data)
    
    # Preprocess the data (optional, if needed for handling missing values)
    data = preprocess_data(data)
    
    # Suggest columns for comparison
    suggest_columns_for_comparison(data)
    
    # Ask user if they want to compare multiple columns
    compare_choice = input("Do you want to compare multiple columns? (yes/no): ").lower()
    if compare_choice == 'yes':
        compare_columns = []
        try:
            num_compare = int(input("How many columns do you want to compare (2 to 10)? "))
            if 2 <= num_compare <= 10:
                for _ in range(num_compare):
                    col = input(f"Enter the column name for comparison {_ + 1}: ")
                    compare_columns.append(col)
                explore_data(data, compare_columns=compare_columns)
            else:
                print("Please enter a number between 2 and 10 for comparison.")
        except ValueError:
            print("Invalid number of columns entered.")
    else:
        # Perform exploratory data analysis (EDA) without comparison
        explore_data(data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        exit()
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        exit()
    except pd.errors.ParserError:
        print("Error parsing the file.")
        exit()

def clean_data(data, drop_threshold=0.5):
    """
    Clean data by handling missing values and dropping columns with excessive missing values.
    """
    # Drop columns with more than drop_threshold (default 50%) missing values
    if drop_threshold > 0:
        data = data.dropna(thresh=data.shape[0] * drop_threshold, axis=1)
    
    # Fill missing values with median for numeric columns and mode for categorical columns
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

def preprocess_data(data):
    """
    Preprocess data by encoding categorical variables.
    """
    # Encode categorical variables
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = pd.factorize(data[column])[0]
    
    return data

def explore_data(data, compare_columns=None):
    """
    Perform exploratory data analysis (EDA) on the data.
    """
    print("Data Description:")
    print(data.describe())
    
    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=np.number).columns
    categorical_cols = data.select_dtypes(include='object').columns
    
    # Histograms for numeric columns
    data[numeric_cols].hist(figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    # Pairplot for numeric columns (if more than one numeric column)
    if len(numeric_cols) > 1:
        sns.pairplot(data[numeric_cols])
        plt.suptitle('Pairplot of Numeric Columns', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # Correlation heatmap for numeric columns (if more than one numeric column)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()
    
    # Boxplots for numeric columns
    num_cols = data.shape[1]
    num_rows = (num_cols - 1) // 3 + 1
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    
    for i, column in enumerate(data.columns):
        ax = axes.flatten()[i]
        if pd.api.types.is_numeric_dtype(data[column]):
            data.boxplot(column=column, ax=ax)
            ax.set_title(f'Boxplot - {column}')
            ax.set_ylabel(column)
    
    for j in range(i + 1, num_rows * 3):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()
    plt.show()
    
    # Pie charts for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title(f'Pie Chart - {col}')
        plt.ylabel('')
        plt.show()
    
    # Compare selected columns (if provided)
    if compare_columns and 2 <= len(compare_columns) <= 10:
        for col in compare_columns:
            if col in data.columns:
                if col in categorical_cols:
                    plt.figure(figsize=(8, 6))
                    data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
                    plt.title(f'Pie Chart - {col}')
                    plt.ylabel('')
                    plt.show()
                elif col in numeric_cols:
                    plt.figure(figsize=(8, 6))
                    data[col].plot.hist(bins=20)
                    plt.title(f'Histogram - {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    print(f"Column '{col}' is not numeric or categorical and cannot be compared.")
            else:
                print(f"Column '{col}' not found in the dataset.")
    elif compare_columns:
        print("Please select between 2 to 10 columns for comparison.")
    else:
        print("No valid columns selected for comparison.")

def suggest_columns_for_comparison(data):
    """
    Suggest columns for comparison based on uniqueness and type.
    """
    # Calculate number of unique values for each column
    unique_counts = data.nunique()
    
    # Rank columns based on uniqueness (higher unique counts are better)
    unique_ranking = unique_counts.sort_values(ascending=False).index.tolist()
    
    # Print suggestion based on rank
    print("Suggested columns for comparison (based on uniqueness and type):")
    for idx, col in enumerate(unique_ranking):
        print(f"{idx + 1}. {col} (Unique values: {unique_counts[col]})")

if __name__ == '__main__':
    # Ask user for the path to the CSV dataset
    file_path = input("Enter the path to your CSV dataset: ").strip('"')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print("The specified file does not exist.")
        exit()
    
    # Load the data
    data = load_data(file_path)
    
    # Display columns in the dataset
    print("\nColumns in the dataset:")
    for idx, column in enumerate(data.columns):
        print(f"{idx + 1}. {column}")
    
    # Ask user if they want to clean the data
    clean_choice = input("Do you want to clean the data? (yes/no): ").lower()
    if clean_choice == 'yes':
        try:
            drop_threshold = float(input("Enter the drop threshold for columns with missing values (0 to 1): "))
            if 0 <= drop_threshold <= 1:
                data = clean_data(data, drop_threshold=drop_threshold)
            else:
                print("Invalid threshold entered. Using default cleaning (drop_threshold=0.5).")
                data = clean_data(data)
        except ValueError:
            print("Invalid input. Using default cleaning (drop_threshold=0.5).")
            data = clean_data(data)
    
    # Preprocess the data (optional, if needed for handling missing values)
    data = preprocess_data(data)
    
    # Suggest columns for comparison
    suggest_columns_for_comparison(data)
    
    # Ask user if they want to compare multiple columns
    compare_choice = input("Do you want to compare multiple columns? (yes/no): ").lower()
    if compare_choice == 'yes':
        compare_columns = []
        try:
            num_compare = int(input("How many columns do you want to compare (2 to 10)? "))
            if 2 <= num_compare <= 10:
                for _ in range(num_compare):
                    col = input(f"Enter the column name for comparison {_ + 1}: ")
                    compare_columns.append(col)
                explore_data(data, compare_columns=compare_columns)
            else:
                print("Please enter a number between 2 and 10 for comparison.")
        except ValueError:
            print("Invalid number of columns entered.")
    else:
        # Perform exploratory data analysis (EDA) without comparison
        explore_data(data)

