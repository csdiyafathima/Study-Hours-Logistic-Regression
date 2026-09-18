
import gradio as gr
import joblib
import pandas as pd
import os

# Load model
model = joblib.load(
    "logistic_regression_student_studyhours_attendance_model.pkl"
)


def predict_result(study_hours, attendance):

    # Create input data
    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "Attendance": [attendance]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        result = "PASS"
        confidence = probability[1] * 100
    else:
        result = "FAIL"
        confidence = probability[0] * 100

    return (
        f"Student Result: {result}\n"
        f"Probability: {confidence:.2f}%"
    )


# Gradio Interface
demo = gr.Interface(
    fn=predict_result,

    inputs=[
        gr.Number(
            label="Enter Study Hours",
            minimum=0,
            maximum=24,
            value=5
        ),

        gr.Number(
            label="Enter Attendance (%)",
            minimum=0,
            maximum=100,
            value=69
        )
    ],

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="🎓 Student Result Prediction",

    description=(
        "Enter the student's Study Hours and Attendance "
        "to predict Pass or Fail."
    )
)


# Run on Render
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
