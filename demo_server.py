from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import os
from demo_test import load_demo_model, preprocess_image, predict_image_demo, class_labels, IMG_HEIGHT, IMG_WIDTH
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the demo model
model = load_demo_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perform_prediction(file_path):
    """
    Perform prediction on the uploaded image and return result.
    """
    # Preprocess the image
    img_array = preprocess_image(file_path, IMG_HEIGHT, IMG_WIDTH)
    
    # Generate random prediction for demo
    import random
    prediction = random.uniform(0.1, 0.9)
    
    # Determine class and confidence
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction
    
    return class_labels[predicted_class], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Perform prediction
        prediction, confidence = perform_prediction(file_path)
        
        # Create a simple report
        report_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Report</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 min-h-screen flex items-center justify-center">
            <div class="bg-white p-8 rounded-lg shadow-lg max-w-md w-full">
                <h1 class="text-2xl font-bold text-center mb-6 text-gray-800">
                    Melanocytic Nevi Diagnosis Result
                </h1>
                
                <div class="mb-6">
                    <img src="/uploads/{filename}" alt="Uploaded image" class="w-full h-64 object-cover rounded-lg">
                </div>
                
                <div class="text-center">
                    <h2 class="text-xl font-semibold mb-2">Prediction:</h2>
                    <p class="text-lg text-blue-600 font-medium mb-2">{prediction}</p>
                    <p class="text-sm text-gray-600 mb-4">Confidence: {confidence:.2f}</p>
                    
                    <div class="mt-6">
                        <a href="/" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                            Upload Another Image
                        </a>
                    </div>
                </div>
                
                <div class="mt-6 text-xs text-gray-500 text-center">
                    <p>This is a demo version with random predictions.</p>
                    <p>For actual diagnosis, use a trained model.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open("templates/report.html", "w") as f:
            f.write(report_content)
        
        return render_template('report.html')
    
    else:
        flash('Please upload a valid image file (PNG, JPG, JPEG, GIF)')
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
