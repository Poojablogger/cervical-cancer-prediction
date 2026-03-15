import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# -------- LOAD MODEL --------
model = pickle.load(open("model.pkl","rb"))

def app():

    st.markdown("## 🎀 Cervical Cancer AI Prediction")

    # session storage
    if "results" not in st.session_state:
        st.session_state.results = []

    # ---------- INPUT FORM ----------
    col1,col2 = st.columns(2)

    with col1:

        age = st.slider("Age",18,80)
        partners = st.slider("Sexual Partners",0,10)
        pregnancies = st.slider("Pregnancies",0,10)

    with col2:

        smoke = st.selectbox("Smoking",["No","Yes"])
        stress = st.slider("Stress Level",1,10)
        immunity = st.slider("Immunity",1,10)

    smoke_val = 1 if smoke=="Yes" else 0

    # ---------- PREDICTION ----------
    if st.button("🔍 Predict Risk"):

        data = pd.DataFrame(
        [[age,partners,pregnancies,smoke_val,stress,immunity]],
        columns=["Age","Partners","Pregnancies","Smoking","Stress","Immunity"]
        )

        prediction = model.predict(data)[0]

        prob = model.predict_proba(data)[0]

        risk = max(prob)*100

        st.subheader(f"Risk Probability : {risk:.2f}%")

        st.progress(int(risk))

        # ---------- RISK LEVEL ----------
        if prediction == 0:
            level = "Low"
            st.success("🟢 Low Risk")

        elif prediction == 1:
            level = "Medium"
            st.warning("🟡 Medium Risk")

        else:
            level = "High"
            st.error("🔴 High Risk")

        # store result
        st.session_state.results.append({
            "Age":age,
            "Partners":partners,
            "Risk":level
        })

        # ---------- EXPLAINABLE AI ----------
        st.markdown("### 🔎 Explainable AI")

        importance = model.feature_importances_

        features = [
        "Age",
        "partners",
        "pregnancies",
        "smoke",
        "hormonal",
        "iud",
        "hinselmann",
        "schiller",
        "citology",
        "lifetime",
        "cancer_load",
        "stress"
        ]

        fig,ax = plt.subplots()

        ax.barh(features,importance)

        ax.set_title("Feature Importance")

        st.pyplot(fig)

        # Save chart for PDF
        chart_path = "chart.png"
        fig.savefig(chart_path)

        # ---------- WHY THIS RISK ----------
        st.markdown("### 🧠 Why this Risk Level?")

        reasons = []

        if age > 45:
            reasons.append("Higher age increases cervical cancer risk")

        if partners > 3:
            reasons.append("Multiple sexual partners increase HPV exposure risk")

        if pregnancies > 3:
            reasons.append("Multiple pregnancies may increase cervical stress")

        if smoke == "Yes":
            reasons.append("Smoking weakens immune response to HPV infection")

        if stress > 7:
            reasons.append("High stress may weaken immune system")

        if immunity < 4:
            reasons.append("Low immunity increases infection vulnerability")

        if len(reasons) == 0:
            st.info("No major risk factors detected.")
        else:
            for r in reasons:
                st.write("•", r)

        # ---------- BASIC MEDICAL ADVICE ----------
        st.markdown("### 👩‍⚕ Basic Health Recommendation")

        if level == "Low":
            st.success("""
Your risk appears LOW.

Recommendations:
• Maintain healthy lifestyle  
• Regular Pap smear screening  
• HPV vaccination if not taken  
• Balanced diet and exercise
""")
        elif level == "Medium":
            st.warning("""
Your risk appears MODERATE.

Recommendations:
• Consult a gynecologist  
• Perform HPV test and Pap smear  
• Reduce stress and smoking  
• Regular medical checkups
""")
        else:
            st.error("""
Your risk appears HIGH.

Recommendations:
• Immediate medical consultation  
• Cervical screening (Pap test / HPV test)  
• Avoid smoking  
• Follow doctor’s advice strictly
""")

        # -------- PDF REPORT --------
        pdf_file = "report_prediction.pdf"
        doc = SimpleDocTemplate(pdf_file)

        styles = getSampleStyleSheet()
        elements = []

        # Hospital Logo
        elements.append(Image("hospital_logo.png", width=120, height=80))
        elements.append(Spacer(1, 20))

        # Title
        elements.append(Paragraph("Cervical Cancer AI Prediction Report", styles["Title"]))
        elements.append(Spacer(1, 20))

        # Risk Probability
        elements.append(Paragraph(f"Risk Probability: {risk:.2f}%", styles["Normal"]))
        elements.append(Spacer(1, 20))

        # Disclaimer
        disclaimer_text = """
<b>Disclaimer:</b> This report is AI-generated for informational purposes only.
It does not replace professional medical advice. Consult a qualified healthcare provider.
"""
        elements.append(Paragraph(disclaimer_text, styles["Normal"]))
        elements.append(Spacer(1, 20))

        # Input Factors Table
        data = [["Factor", "Value"]]
        data.append(["Age", age])
        data.append(["Partners", partners])
        data.append(["Pregnancies", pregnancies])
        data.append(["Smoking", smoke])
        data.append(["Stress", stress])
        data.append(["Immunity", immunity])

        table = Table(data)
        table.setStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.pink),
            ("GRID", (0,0), (-1,-1), 1, colors.black)
        ])
        elements.append(table)
        elements.append(Spacer(1, 30))

        # Chart
        elements.append(Image(chart_path, width=400, height=250))
        elements.append(Spacer(1, 40))

        # Doctor Approval
        elements.append(Paragraph("Doctor Approval", styles["Heading3"]))
        elements.append(Image("doctor_sign.png", width=150, height=80))

        # Build PDF
        doc.build(elements)

        # Provide download button
        with open(pdf_file, "rb") as f:
            st.download_button(
                "Download Prediction Report",
                f,
                file_name="Cervical_Cancer_Prediction_Report.pdf"
            )