import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Linear Regression App", layout="centered")

st.title("📈 Linear Regression using Streamlit")
st.write("Linear Regression implemented **without any dataset file**.")

# Generate sample data
np.random.seed(42)
X = np.linspace(1, 100, 50).reshape(-1, 1)
y = 3 * X + np.random.randn(50, 1) * 15

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction line
X_line = np.linspace(1, 100, 100).reshape(-1, 1)
y_line = model.predict(X_line)

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X_line, y_line)
ax.set_xlabel("X Value")
ax.set_ylabel("Y Value")
ax.set_title("Linear Regression Graph")

st.pyplot(fig, clear_figure=True)

# User input
st.subheader("🔢 Predict Y for a given X")
x_input = st.number_input("Enter X value", min_value=0.0, step=1.0)

if st.button("Predict"):
    y_pred = model.predict([[x_input]])
    st.success(f"Predicted Y value: {y_pred[0][0]:.2f}")

