import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Women's Cervical Cancer AI System",
    page_icon="🎀",
    layout="wide"
)

# ---------------- LOAD CSS ----------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if "users" not in st.session_state:
    st.session_state.users = {}

# -------- DEFAULT 10 PATIENT RECORDS --------
if "results" not in st.session_state:
    st.session_state.results = [
        {"User":"Anita","Risk":"Low","Time":"10:00"},
        {"User":"Priya","Risk":"Medium","Time":"10:10"},
        {"User":"Divya","Risk":"Low","Time":"10:20"},
        {"User":"Meena","Risk":"High","Time":"10:30"},
        {"User":"Kavya","Risk":"Medium","Time":"10:40"},
        {"User":"Nisha","Risk":"Low","Time":"10:50"},
        {"User":"Latha","Risk":"Medium","Time":"11:00"},
        {"User":"Rekha","Risk":"Low","Time":"11:10"},
        {"User":"Sangeetha","Risk":"High","Time":"11:20"},
        {"User":"Deepa","Risk":"Medium","Time":"11:30"}
    ]

# ---------------- SIDEBAR ----------------
if st.session_state.logged_in:
    menu = st.sidebar.selectbox("🎀 Navigation",["Home","Prediction","Analysis","Logout"])
else:
    menu = st.sidebar.selectbox("🎀 Navigation",["Home","Register","Login"])

# ---------------- HOME PAGE ----------------
if menu == "Home":

    st.markdown("""
    <div class="banner">
    👩‍⚕ AI Powered Cervical Cancer Risk Prediction System
    </div>
    """, unsafe_allow_html=True)

    col1,col2 = st.columns([1,2])

    with col1:
        st.image("cervical.jpg",width=350)

    with col2:
        st.markdown("""
### What is Cervical Cancer?
Cervical cancer occurs in the cells of the cervix.

Most cervical cancers are caused by HPV infection.

Early detection through Pap smear tests can prevent complications.

### Prevention
✔ HPV Vaccination  
✔ Regular screening  
✔ Healthy lifestyle  
""")

# ---------------- REGISTER ----------------
elif menu == "Register":

    st.title("👩 User Registration")

    username = st.text_input("Username")
    password = st.text_input("Password",type="password")

    if st.button("Register"):

        if username in st.session_state.users:
            st.error("User already exists")
        else:
            st.session_state.users[username]=password
            st.success("Registration Successful")

# ---------------- LOGIN ----------------
elif menu == "Login":

    st.title("🔐 User Login")

    username = st.text_input("Username")
    password = st.text_input("Password",type="password")

    if st.button("Login"):

        if username in st.session_state.users and st.session_state.users[username]==password:
            st.session_state.logged_in=True
            st.session_state.current_user=username
            st.success("Login Successful")
        else:
            st.error("Invalid Login")

# ---------------- LOGOUT ----------------
elif menu == "Logout":

    st.session_state.logged_in=False
    st.session_state.current_user=""
    st.success("Logged Out Successfully")

# ---------------- PREDICTION ----------------
elif menu == "Prediction":

    if not st.session_state.logged_in:
        st.warning("Please login first")
        st.stop()

    st.title("👩‍⚕ Cervical Cancer Risk Prediction")

    age = st.number_input("Age",18,80)
    partners = st.number_input("Number of Partners",0,20)
    first_sex = st.number_input("Age at First Intercourse",10,50)
    pregnancies = st.number_input("Number of Pregnancies",0,10)
    smoke = st.selectbox("Do you Smoke?",["Yes","No"])
    hormonal = st.number_input("Years of Hormonal Contraceptives",0,20)
    iud = st.number_input("Years of IUD Usage",0,20)
    hinselmann = st.number_input("Hinselmann Test Result (0 or 1)",0,1)
    schiller = st.number_input("Schiller Test Result (0 or 1)",0,1)
    citology = st.number_input("Citology Test Result (0 or 1)",0,1)
    lifetime = st.number_input("Lifetime Exposure Index",0,50)
    cancer_load = st.number_input("Family Cancer History Level (0-5)",0,5)
    stress = st.number_input("Stress Level (1-10)",1,10)

    if st.button("Predict Risk"):

        smoke_val = 5 if smoke=="Yes" else 0

        risk_score = (
            age*0.5 +
            partners*2 +
            (50-first_sex)*0.3 +
            pregnancies*1.5 +
            smoke_val +
            hormonal*1.2 +
            iud +
            hinselmann*10 +
            schiller*10 +
            citology*10 +
            lifetime*2 +
            cancer_load*3 +
            stress*2
        )

        risk_percentage=min((risk_score/200)*100,100)

        if risk_percentage<30:
            risk_label="Low"
            st.success("Low Risk")
        elif risk_percentage<60:
            risk_label="Medium"
            st.warning("Medium Risk")
        else:
            risk_label="High"
            st.error("High Risk")

        # -------- SAVE RESULT WITH USERNAME --------
        st.session_state.results.append({
            "User":st.session_state.current_user,
            "Risk":risk_label,
            "Time":datetime.now().strftime("%H:%M")
        })

        st.markdown(f"""
        <div class="card">
        <h2>Predicted Risk</h2>
        <h1>{risk_percentage:.2f}%</h1>
        </div>
        """,unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
elif menu == "Analysis":

    if not st.session_state.logged_in:
        st.warning("Please login first")
        st.stop()

    st.title("📊 Prediction Analytics Dashboard")

    df=pd.DataFrame(st.session_state.results)

    total=len(df)
    low=len(df[df["Risk"]=="Low"])
    medium=len(df[df["Risk"]=="Medium"])
    high=len(df[df["Risk"]=="High"])

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Total Screenings",total)
    col2.metric("Low Risk",low)
    col3.metric("Medium Risk",medium)
    col4.metric("High Risk",high)

    st.markdown("---")

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        fig,ax=plt.subplots()
        ax.pie([low,medium,high],labels=["Low","Medium","High"],autopct="%1.1f%%")
        st.pyplot(fig)

    with col2:

        st.subheader("Risk Count")

        fig2,ax2=plt.subplots()
        ax2.bar(["Low","Medium","High"],[low,medium,high])
        st.pyplot(fig2)

    st.markdown("---")

    # -------- TIMELINE --------
    st.subheader("📅 Prediction Timeline")

    for index,row in df[::-1].iterrows():

        if row["Risk"]=="Low":
            color="🟢"
        elif row["Risk"]=="Medium":
            color="🟡"
        else:
            color="🔴"

        st.markdown(f"""
        **{row['Time']}**  
        👩 User: **{row['User']}**  
        Risk Level: {color} **{row['Risk']}**
        """)

    st.markdown("---")

    st.subheader("Prediction Records")

    st.dataframe(df)

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
'<div class="footer">🎀 Women\'s AI Cervical Cancer Prediction System</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="footer">Developed for Healthcare Risk Analysis</div>',
unsafe_allow_html=True
)