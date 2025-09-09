import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load data
df = pd.read_csv("student_performance_data.csv")
df['math_score'] = df.groupby('major')['math_score'].transform(lambda x: x.fillna(x.median()))
df['attendance_rate'] = df['attendance_rate'].fillna(df['attendance_rate'].mean())
df['average_score'] = df[['math_score','science_score','english_score']].mean(axis=1)

# Train GPA model
X = df[["math_score","science_score","english_score","attendance_rate","study_hours_per_week","average_score"]]
y = df["gpa"]
model = LinearRegression().fit(X,y)

# Streamlit App
st.title("📊 Student Performance Dashboard")

# GPA Distribution
st.subheader("GPA Distribution")
fig = px.histogram(df, x="gpa", nbins=10, color="major", marginal="box")
st.plotly_chart(fig)

# Average Score by Major
st.subheader("Average Score by Major")
fig = px.bar(df, x="major", y="average_score", color="major")
st.plotly_chart(fig)

# Study Hours vs GPA
st.subheader("Study Hours vs GPA")
fig = px.scatter(df, x="study_hours_per_week", y="gpa", color="major", size="average_score")
st.plotly_chart(fig)

# Attendance vs GPA
st.subheader("Attendance vs GPA")
fig = px.scatter(df, x="attendance_rate", y="gpa", color="major", size="study_hours_per_week")
st.plotly_chart(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = X.join(y).corr()
fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1))
st.plotly_chart(fig)

# GPA Predictor
st.subheader("🎯 Predict GPA")
math = st.slider("Math Score", 50, 100, 70)
science = st.slider("Science Score", 50, 100, 70)
english = st.slider("English Score", 50, 100, 70)
attendance = st.slider("Attendance Rate", 50, 100, 80)
study = st.slider("Study Hours per Week", 1, 40, 10)

avg_score = (math + science + english) / 3
pred = model.predict([[math, science, english, attendance, study, avg_score]])[0]
st.success(f"Predicted GPA: {pred:.2f}")
