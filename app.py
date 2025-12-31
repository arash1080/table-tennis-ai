import streamlit as st
import joblib
import pandas as pd
from pose_analysis import analyze_pose

st.set_page_config(page_title="استعداد‌یابی پینگ‌پنگ", layout="wide")

st.title("🏓 سیستم هوشمند استعداد‌یابی پینگ‌پنگ")

text_model = joblib.load("models/text_model.pkl")
match_model = joblib.load("models/match_model.pkl")

st.header("مرحله ۱: اطلاعات فردی")

age = st.number_input("سن", 10, 30)
height = st.number_input("قد (cm)", 140, 200)
weight = st.number_input("وزن (kg)", 40, 120)
reaction = st.number_input("زمان واکنش (ms)", 200, 400)
training = st.slider("ساعت تمرین هفتگی", 0, 20)
hand = st.selectbox("دست غالب", ["راست", "چپ"])
exp = st.slider("سابقه ورزشی (سال)", 0, 10)

if st.button("تحلیل استعداد"):
    X = pd.DataFrame([[age, height, weight, reaction, training, 1 if hand=="راست" else 0, exp]],
                     columns=["age","height","weight","reaction_time","training_hours","dominant_hand","experience"])
    res = text_model.predict(X)[0]
    st.success(f"سطح پیشنهادی: {res}")

st.header("مرحله ۲: تحلیل حرکت (Pose Detection)")
video = st.file_uploader("ویدئوی تمرین", type=["mp4"])

if video:
    with open("temp.mp4", "wb") as f:
        f.write(video.read())

    score = analyze_pose("temp.mp4")
    st.info(f"امتیاز تکنیک: {score}")

st.header("مرحله ۳: پیش‌بینی مسابقه")
if st.button("پیش‌بینی نتیجه"):
    prob = match_model.predict_proba([[score, exp]])[0][1]
    st.success(f"احتمال برد: {int(prob*100)}٪")