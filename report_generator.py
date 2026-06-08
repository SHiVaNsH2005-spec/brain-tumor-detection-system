from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

def generate_report(
    filename,
    prediction,
    confidence,
    risk,
    description,
    original_image_path,
    gradcam_image_path
):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    content = []

    title = Paragraph(
        "🏥 AI Brain Tumor Detection Report",
        styles["Title"]
    )

    content.append(title)

    content.append(Spacer(1, 20))

    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )
    
    import random

    report_id = f"BT-{random.randint(100000,999999)}"

    Paragraph(
    f"<b>Report ID:</b> {report_id}",
    styles["BodyText"]
    ) 
    
    content.append(
        Paragraph(
            f"<b>Date & Time:</b> {current_time}",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 15))

    if prediction.lower() == "notumor":
        color = "green"
        display_prediction = "NO TUMOR"
    else:
        color = "red"
        display_prediction = prediction.upper()

    content.append(
        Paragraph(
            f"<b>Tumor Type:</b> <font color='{color}'>{display_prediction}</font>",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence Score:</b> {confidence:.2f}%",
            styles["BodyText"]
        )
    )
    if confidence >= 95:
        confidence_text = "Very High Confidence"
    elif confidence >= 80:
        confidence_text = "High Confidence"
    elif confidence >= 60:
        confidence_text = "Moderate Confidence"
    else:
        confidence_text = "Low Confidence"

    content.append(
        Paragraph(
            f"<b>Model Confidence Level:</b> {confidence_text}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Risk Assessment:</b> {risk}",
            styles["BodyText"]
        )
    )

    from reportlab.platypus import Table, TableStyle, Image

    content.append(
        Paragraph(
            "MRI Analysis",
            styles["Heading2"]
        )
    )

    content.append(Spacer(1, 10))

    original_img = Image(
        original_image_path,
        width=180,
        height=180
    )

    gradcam_img = Image(
        gradcam_image_path,
        width=180,
        height=180
    )

    table = Table(
        [[original_img, gradcam_img]],
        colWidths=[220, 220]
    )

    table.setStyle(
        TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER')
        ])
    )

    content.append(table)

    content.append(Spacer(1, 10))

    content.append(
        Paragraph(
            "<b>Left:</b> Original MRI &nbsp;&nbsp;&nbsp;&nbsp; "
            "<b>Right:</b> Grad-CAM Localization",
            styles["BodyText"]
        )
    )  

    content.append(
        Paragraph(
            "<b>AI Analysis</b>",
            styles["Heading2"]
        )
    )
    content.append(
        Paragraph(
            description,
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 10))

    content.append(
        Paragraph(
            "<b>Conclusion</b>",
            styles["Heading2"]
        )
    )

    if prediction.lower() == "notumor":
        conclusion = (
            "No tumor detected by the AI system. "
            "The scan appears normal with high confidence."
        )
    else:
        conclusion = (
            f" Based on the uploaded MRI scan, the AI model identified imaging patterns consistent with {prediction.upper()}."
            f" The prediction was generated with {confidence:.2f}% confidence. The Grad-CAM visualization indicates that the highlighted region significantly contributed to the model's decision. "
        )

    content.append(
        Paragraph(
            conclusion,
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 25))
    content.append(
        Paragraph(
            "<b>Medical Disclaimer</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            "This report is generated using an Artificial Intelligence model for educational and research purposes. Clinical decisions should always be made by qualified medical professionals after reviewing all relevant patient information.",
            styles["Italic"]
        )
    )

    doc.build(content)