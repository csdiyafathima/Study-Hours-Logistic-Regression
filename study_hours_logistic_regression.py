
import gradio as gr
import joblib
model = joblib.load(
    "logistic_regression_student_studyhours_attendance_model.pkl"
)
def predict_result(hours, attendance):

    
    input_data = [[hours, attendance]]

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probabilities = model.predict_proba(input_data)

    pass_probability = probabilities[0][1] * 100
    fail_probability = probabilities[0][0] * 100

    # Result
    if prediction[0] == 1:
        result = "✅ Student is predicted to PASS"
    else:
        result = "❌ Student is predicted to FAIL"

    return (
        result,
        f"{pass_probability:.2f}%",
        f"{fail_probability:.2f}%"
    )


# Create Gradio interface
app = gr.Interface(
    fn=predict_result,

    inputs=[
        gr.Number(
            label="Enter Study Hours",
            value=5.0
        ),

        gr.Number(
            label="Enter Attendance (%)",
            value=69.0
        )
    ],

    outputs=[
        gr.Textbox(
            label="📊 Prediction Result"
        ),

        gr.Textbox(
            label="🎯 Probability of Passing"
        ),

        gr.Textbox(
            label="❌ Probability of Failing"
        )
    ],

    title="🎓 Student Pass Prediction",

    description=(
        "Enter the student's study hours and attendance "
        "to predict the result."
    )
)


# Launch the app
app.launch(erver_name="0.0.0.0", server_port=7860)
