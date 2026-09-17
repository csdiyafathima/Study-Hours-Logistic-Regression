import streamlit as st
import joblib
model=joblib.load("logistic_regression_student_studyhours_attendance_model.pkl")
st.title("🎓 Student Pass Prediction")
st.write("Enter the student's study hours  and attendance to predict the result.")

hours=st.number_input("Enter Study Hours",min_value=0.0,max_value=24.0,value=5.0,step=0.5)
attendance=st.number_input("Enter the Attendance",min_value=0.0,max_value=100,value=69.0,step=1.0)
if st.button("🔮 Predict Result"):
    input_data = [[hours,attendance]]
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    pass_probability = probabilities[0][1] * 100
    fail_probability = probabilities[0][0] * 100

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Student is predicted to PASS")
    else:
        st.error("❌ Student is predicted to FAIL")

    st.metric(
        "🎯 Probability of Passing",
        f"{pass_probability:.2f}%"
    )

    st.progress(pass_probability / 100)

    
    st.write(
        f"❌ Probability of Failing: **{fail_probability:.2f}%**"
    )
