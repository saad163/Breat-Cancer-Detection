# from flask import Flask, render_template, request, send_file
# import os
# from werkzeug.utils import secure_filename
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.platypus import Table, TableStyle
# from reportlab.lib import colors

# app = Flask(__name__)

# UPLOAD_FOLDER = 'C:/Users/Ahmad Hanif/Downloads/Breast Cancer/static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load trained model
# model = tf.keras.models.load_model('BreastCancer95%_model.h5')

# # Function to preprocess image
# def preprocess_image(img_path):
#     img = Image.open(img_path)  # Open image
#     img = img.convert("RGB")  # Convert grayscale to RGB
#     img = img.resize((224, 224))  # Resize to model's input size
#     img_array = np.array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         patient_name = request.form.get("patient_name")
#         if 'file' not in request.files:
#             return render_template('index.html', error='No file uploaded')
#         file = request.files['file']
#         if file.filename == '':
#             return render_template('index.html', error='No file selected')
#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#             # Preprocess and predict
#             img = preprocess_image(filepath)
#             prediction = model.predict(img)[0][0]
#             result = "Cancer Detected" if prediction > 0.5 else "No Cancer Detected"
            
#             # Dynamic Comments
#             comments = "Consult a doctor immediately and seek treatment as soon as possible!" if prediction > 0.5 else "You don't need to worry at all, you are completely safe!"
            
#             return render_template('index.html', filename=filename, result=result, patient_name=patient_name, Comments=comments)
    
#     return render_template('index.html')

# @app.route('/download_report/<filename>/<patient_name>/<result>')
# def download_report(filename, patient_name, result):
#     pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Medical_Report.pdf')
#     c = canvas.Canvas(pdf_path, pagesize=letter)
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(200, 750, "Mammogram Medical Report")
#     c.setFont("Helvetica", 12)
#     c.drawString(50, 700, f"Patient Name: {patient_name}")
#     c.drawString(50, 680, f"Diagnosis: {result}")
    
#     # Dynamic Comments
#     comments = "Consult a doctor immediately and seek treatment as soon as possible!" if "Cancer Detected" in result else "You don't need to worry at all, you are completely safe!"
#     c.drawString(50, 660, f"Comments: {comments}")
    
#     # Table Data
#     data = [['Attribute', 'Value'],
#             ['Patient Name', patient_name],
#             ['Diagnosis', result],
#             ['Comments', comments]]
    
#     table = Table(data, colWidths=[200, 200])
#     table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#         ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black),
#     ]))
#     table.wrapOn(c, 50, 500)
#     table.drawOn(c, 50, 600)
    
#     # Insert Image
#     img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     c.drawImage(img_path, 50, 350, width=200, height=200)
    
#     c.save()
    
#     return send_file(pdf_path, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)

import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Users/Ahmad Hanif/Downloads/Breast Cancer/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model('BreastCancer95%_model.h5')

# Function to preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path)  # Open image
    img = img.convert("RGB")  # Convert grayscale to RGB
    img = img.resize((224, 224))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to highlight tumor (bounding box)
def highlight_tumor(image_path):
    img = cv2.imread(image_path)  
    height, width, _ = img.shape  

    # 🟡 Dummy bounding box coordinates (Tumor model required for actual detection)
    x, y, w, h = width//3, height//3, width//4, height//4  
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)  # 🔴 Draw red box

    # Save updated image
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_path, img)
    return processed_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        patient_name = request.form.get("patient_name")
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            img = preprocess_image(filepath)
            prediction = model.predict(img)[0][0]
            result = "Cancer Detected" if prediction > 0.5 else "No Cancer Detected"

            # Highlight tumor if detected
            processed_image = highlight_tumor(filepath) if prediction > 0.5 else filepath
            
            # Dynamic Comments
            comments = "Consult a doctor immediately and seek treatment as soon as possible!" if prediction > 0.5 else "You don't need to worry at all, you are completely safe!"
            
            return render_template('index.html', filename=os.path.basename(processed_image), result=result, patient_name=patient_name, Comments=comments)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
