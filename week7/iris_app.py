# iris_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Iris Flower Classifier", layout="wide")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for EDA
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species'] = df['species'].map({i: name for i, name in enumerate(target_names)})

# Train the Random Forest model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel
st.title("Iris Flower Classification App")
st.subheader("Using Random Forest Classifier")

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write(f"Predicted Species: **{target_names[prediction[0]]}**")

st.subheader('Prediction Probability')
st.write(f"Setosa: {prediction_proba[0][0]:.2f}")
st.write(f"Versicolor: {prediction_proba[0][1]:.2f}")
st.write(f"Virginica: {prediction_proba[0][2]:.2f}")

# Visualizations
st.header("Data Visualizations")

# Feature distributions
st.subheader("Feature Distributions by Species")
fig1, ax1 = plt.subplots(2, 2, figsize=(12, 10))
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.boxplot(x='species', y=feature, data=df, ax=ax1[row, col])
    ax1[row, col].set_title(f'{feature} distribution')

plt.tight_layout()
st.pyplot(fig1)

# Feature correlations
st.subheader("Feature Correlations")
fig2 = plt.figure(figsize=(8, 6))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
st.pyplot(fig2)

# Feature importances
st.subheader("Feature Importances")
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

fig3 = plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importances')
st.pyplot(fig3)
