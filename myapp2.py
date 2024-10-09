import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'Salary_Data.csv'  # Ensure the CSV file exists in this path
try:
    data = pd.read_csv(file_path)
    # Streamlit app UI
    st.title('Salary Prediction Based on Years of Experience')
    st.write("## By : Huda Alzaharni & Layan Alghamdi")
    st.write("##### First Five Rows of the Data")
    st.write(data.head())  # Display the data in Streamlit to ensure it's loaded
except Exception as e:
    st.error(f"Error loading data: {e}")

# Check if data is loaded before proceeding
if 'data' in locals():
    # Split data into input and output variables
    X = data['YearsExperience'].values.reshape(-1, 1)  # Input variable: Years of experience
    y = data['Salary'].values  # Output variable: Salary

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)



    # User input for years of experience using text_input
    years_experience = st.text_input('Enter Years of Experience:')

    # Prediction button
    if st.button('Predict Salary'):
        try:
            # Convert input to float and make sure it's a valid number
            years_experience = float(years_experience)
            prediction = model.predict([[years_experience]])[0]
            st.success(f'Predicted Salary: {round(prediction, 2)}')
        except ValueError:
            st.error("Please enter a valid number for years of experience.")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

