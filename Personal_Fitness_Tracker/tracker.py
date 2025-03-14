import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🔥", layout="centered")
st.title("🏋️‍♂️ Personal Fitness Tracker")

st.write("In this WebApp, you can **predict the calories burned** based on parameters like `Age`, `Gender`, `BMI`, etc. 🏃‍♀️💪")
st.sidebar.header("⚙️ User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("🎂 Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("⚖️ BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("⏳ Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("❤️ Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("🌡️ Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("🚻 Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0
    
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

st.write("---")
st.header("📊 Your Parameters")
st.write(df)

st.write("🔄 Processing your input...")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Load Data
calories = pd.read_csv("/Users/noorshaik/Downloads/untitled folder 5/calories.csv")
exercise = pd.read_csv("/Users/noorshaik/Downloads/untitled folder 5/exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

st.write("---")
st.header("🔥 Prediction")
st.success(f"💥 You burned **{round(prediction[0], 2)} kilocalories** during your workout! 🏃‍♂️💨")

st.write("---")
st.header("🔍 Similar Results")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("📈 General Information")
st.info(f"🆘 You are older than **{round(sum((exercise_df['Age'] < df['Age'].values[0])) / len(exercise_df), 2) * 100}%** of other users.")
st.warning(f"⏱️ Your exercise duration is longer than **{round(sum((exercise_df['Duration'] < df['Duration'].values[0])) / len(exercise_df), 2) * 100}%** of users.")
st.success(f"❤️ Your heart rate is higher than **{round(sum((exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0])) / len(exercise_df), 2) * 100}%** of users during exercise.")
st.error(f"🌡️ Your body temperature is higher than **{round(sum((exercise_df['Body_Temp'] < df['Body_Temp'].values[0])) / len(exercise_df), 2) * 100}%** of users.")

st.write("---")
st.subheader("Thank you for using Personal Fitness Tracker! 🚀")
