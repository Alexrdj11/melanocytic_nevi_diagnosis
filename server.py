from flask import Flask, request, render_template, redirect, url_for
import os
from test import load_trained_model, preprocess_image, predict_image, class_labels, IMG_HEIGHT, IMG_WIDTH
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = "best_model.keras"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model once at the start
model = load_trained_model(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform prediction
        prediction, confidence = perform_prediction(file_path)

        # Save results for display in report.html
        with open("templates/report.html", "w") as report_file:
            report_file.write(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Prediction Report</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link
                    href="https://fonts.googleapis.com/css2?family=Caveat:wght@400..700&family=Funnel+Display:wght@300..800&family=Hubot+Sans:ital,wght@0,200..900;1,200..900&family=Press+Start+2P&display=swap"
                rel="stylesheet">
                <style>
                    .font-hubot {{
                        font-family: 'Funnel Display', serif;
                    }}
                </style>
            </head>
            <body>
                <div class="bg-slate-950 container min-h-screen">
                    <h1 class="text-stone-300 text-5xl text-center font-hubot p-24">
                        Skin Disease Detection using Deep Learning
                        <span class="mt-5 before:block before:absolute before:-inset-1 before:-skew-y-3 before:bg-slate-500 relative inline-block">
                            <span class="relative text-slate-950">Melanocytic Nevi</span>
                        </span>
                    </h1>
                    <div class="w-4/12 h-2/5 bg-slate-500 rounded-lg mx-auto flex flex-col items-center justify-center text-center">
                    <h1 class="text-slate-950 font-hubot text-4xl mb-10 mt-10">Prediction Report</h1>
                    <p class="text-stone-300 font-hubot text-2xl mb-2 mt-2"><b class="text-slate-950">Predicted Class:</b> {prediction}</p>
                    <p class="text-stone-300 font-hubot text-2xl mb-2 mt-2"><b class="text-slate-950">Confidence:</b> {confidence:.2f}</p>
                    <a class="bg-slate-950 rounded-full px-10 py-4 mb-10 mt-10 text-stone-300" href="/">Upload Another Image</a>
                </div>
            </body>
            </html>
            """)

        # Redirect to report page
        return redirect(url_for('report'))


@app.route('/report')
def report():
    return render_template('report.html')


def perform_prediction(image_path):
    """
    Perform prediction using the model and return the result.
    """
    img_array = preprocess_image(image_path, IMG_HEIGHT, IMG_WIDTH)
    prediction = model.predict(img_array)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction
    return class_labels[predicted_class], confidence


if __name__ == "__main__":
    app.run(debug=True)
