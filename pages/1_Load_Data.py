import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Plotting Demo", page_icon="📈", layout="wide")

def generate_countplots(data, columns, target):
    plt.figure(figsize=(20, 25))
    index = 1
    temp = data.drop(target, axis=1)

    for i in columns:
        plt.subplot(5, 4, index)
        sns.countplot(data=data, x=i, hue=target)
        index += 1
    plt.show()

# Define correlation matrix
def correlation_matrix(data, y_column):
    # Separate the variables from the label
    X = data.drop(column, axis=1)
    y = data[column]
    # Calculate the correlation matrix
    corr_matrix = X.corr()
    # Create a heatmap plot of the correlation matrix
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')

    return fig

# Set up the left-side panel
st.sidebar.title('Select Options')
# Create a file uploader widget
uploaded_file = st.sidebar.file_uploader('1 - Upload CSV or Excel file', type=['csv', 'xlsx'])

if not uploaded_file:
    st.write('# Please Upload the CSV or Excel file')

if uploaded_file:
    # Read the uploaded file into a Pandas DataFrame
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    data.dropna(inplace=True)
    st.session_state['data'] = data
    pr = data.profile_report()
    # Display the table
    st.write(st_profile_report(pr))
