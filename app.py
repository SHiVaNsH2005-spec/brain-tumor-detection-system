from report_generator import generate_report
from gradcam import make_gradcam_heatmap
from gradcam import overlay_heatmap
import cv2
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

model = load_model()

# Class Labels
classes = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Tumor Information
tumor_info = {
    "glioma":
    """
    The AI model detected features consistent with a Glioma tumor.
    Gliomas originate from glial cells within the brain and can vary in severity.
    The highlighted Grad-CAM region indicates the area that contributed most strongly to the prediction.
    Further clinical evaluation and radiological assessment are recommended for confirmation.
    """,

    "meningioma":
    """
    The AI model identified imaging characteristics consistent with a Meningioma.
    Meningiomas develop from the meninges, the protective membranes surrounding the brain and spinal cord.
    These tumors are commonly slow-growing and are often benign, although their location may cause neurological symptoms.
    The Grad-CAM visualization highlights the region that most influenced the model's decision.
    Clinical correlation and specialist review are recommended for final diagnosis.
    """,

    "pituitary":
    """
    The AI model detected features associated with a Pituitary tumor.
    Pituitary tumors arise in the pituitary gland and may affect hormone production and endocrine function.
    The highlighted area in the Grad-CAM visualization corresponds to the region most relevant to the prediction.
    Additional endocrinological and radiological evaluation may be required to determine clinical significance.
    """,

    "no tumor":
    """
    The AI model did not identify significant imaging features associated with a brain tumor.
    The MRI scan appears consistent with normal brain anatomy based on the learned patterns of the model.
    No suspicious tumor-related regions were strongly activated in the classification process.
    Despite this result, clinical interpretation by a qualified medical professional remains essential.
    """
}

# Sidebar
st.sidebar.title("🧠 Brain Tumor AI")
st.sidebar.success("Model Accuracy: 94.13%")

st.title("🧠 Brain Tumor Detection & Analysis System")

uploaded_file = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    with col2:

        st.subheader("Prediction")

        if predicted_class == "notumor":
            st.success(f"✅ {predicted_class.upper()}")
        else:
            st.error(f"⚠️ {predicted_class.upper()}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )
        

        st.info(tumor_info[predicted_class])
        if confidence > 90:
            risk = "High Confidence"
        elif confidence > 75:
            risk = "Moderate Confidence"
        else:
            risk = "Low Confidence"

        st.metric("Risk Assessment", risk)

        if confidence < 70:
           st.warning(
           f"Low confidence prediction ({confidence:.2f}%). "
           "Please verify with additional scans."
           )

        st.markdown("---")
        img_for_gradcam = np.expand_dims(
        np.array(img),
        axis=0
        )

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            "top_conv"
        )

        overlay = overlay_heatmap(
            np.array(img),
            heatmap
        )

        st.subheader("🔥 Tumor Localization (Grad-CAM)")
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original MRI",
                use_container_width=True
            )

        with col2:
            st.image(
                overlay,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        st.markdown("---")
            # SAVE HISTORY HERE
        history_file = "prediction_history.csv"

        new_entry = pd.DataFrame({
            "Date":[datetime.now()],
            "Prediction":[predicted_class],
            "Confidence":[confidence]
        })

        if os.path.exists(history_file):
            new_entry.to_csv(
                history_file,
                mode="a",
                header=False,
                index=False
            )
        else:
            new_entry.to_csv(
                history_file,
                index=False
            )

    st.markdown("---")  
    report_file = "brain_tumor_report.pdf"

    # Save original MRI
    image.save("original_mri.png")

    # Save Grad-CAM image
    cv2.imwrite(
        "gradcam.png",
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    generate_report(
        report_file,
        predicted_class,
        confidence,
        risk,
        tumor_info[predicted_class],
        "original_mri.png",
        "gradcam.png"
    )
    

    with open(report_file, "rb") as pdf_file:

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="Brain_Tumor_Report.pdf",
            mime="application/pdf"
        )
    st.markdown("---") 
    st.subheader("Class Probabilities")

    for cls, prob in zip(classes, prediction[0]):
       st.progress(float(prob))
       st.write(f"{cls}: {prob*100:.2f}%")
    
    st.markdown("---")
    st.subheader("Prediction Probabilities")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=classes,
            y=prediction[0]
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    if st.button("Analyze MRI"):
        st.markdown("---")
        st.subheader("📜 Prediction History")

        history_file = "prediction_history.csv"

    if st.button("🗑️ Clear History"):
        if os.path.exists(history_file):
            os.remove(history_file)
            st.success("History Cleared!")
            st.rerun()

    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        st.dataframe(history.tail(10), use_container_width=True)
        