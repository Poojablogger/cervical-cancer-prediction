import streamlit as st

def app():

    # ---------- BANNER ----------

    st.markdown(
    '<div class="banner">🎀 AI Powered Cervical Cancer Prediction System</div>',
    unsafe_allow_html=True
    )

    # ---------- INTRO SECTION ----------

    col1, col2 = st.columns([1,1.5])

    with col1:
        st.image("cervical.jpg")

    with col2:

        st.subheader("What is Cervical Cancer?")

        st.write(
        "Cervical cancer occurs in the cells of the cervix. "
        "Most cases are caused by HPV infection."
        )

        st.write(
        "Early screening methods like Pap smear and HPV testing "
        "can help detect cervical cancer at an early stage."
        )

        st.subheader("Prevention")

        st.write("✔ HPV Vaccination")
        st.write("✔ Regular Pap Smear Screening")
        st.write("✔ Healthy Lifestyle")

    st.markdown("---")

    # ---------- INFORMATION CARDS ----------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
        '''
        <div class="card">
        <h3>Symptoms</h3>
        <p>
        Bleeding between periods<br>
        Pelvic pain<br>
        Unusual discharge
        </p>
        </div>
        ''',
        unsafe_allow_html=True
        )

    with c2:
        st.markdown(
        '''
        <div class="card">
        <h3>Causes</h3>
        <p>
        HPV Infection<br>
        Smoking<br>
        Weak Immunity
        </p>
        </div>
        ''',
        unsafe_allow_html=True
        )

    with c3:
        st.markdown(
        '''
        <div class="card">
        <h3>Prevention</h3>
        <p>
        HPV Vaccine<br>
        Pap Test<br>
        Healthy Lifestyle
        </p>
        </div>
        ''',
        unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------- AI SYSTEM FEATURES ----------

    st.subheader("🤖 AI System Features")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
        '''
        <div class="card">
        <h3>Risk Prediction</h3>
        <p>AI model predicts cervical cancer risk based on patient health data.</p>
        </div>
        ''',
        unsafe_allow_html=True
        )

    with f2:
        st.markdown(
        '''
        <div class="card">
        <h3>Data Analysis</h3>
        <p>Dashboard provides risk distribution and prediction analytics.</p>
        </div>
        ''',
        unsafe_allow_html=True
        )

    with f3:
        st.markdown(
        '''
        <div class="card">
        <h3>Explainable AI</h3>
        <p>Feature importance helps doctors understand prediction reasons.</p>
        </div>
        ''',
        unsafe_allow_html=True
        )