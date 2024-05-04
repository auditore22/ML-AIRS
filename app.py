import os

from flask import Flask, request, jsonify, render_template, url_for, send_file
from keras_tuner.src.backend.io import tf
from werkzeug.utils import secure_filename, redirect
from Model.Training.use_model import predict_image

app = Flask(__name__, template_folder='View/templates', static_folder='View/static')

UPLOAD_FOLDER = 'View/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
model = tf.saved_model.load('Model/Training/Models/Stable/run_final')

# Ensure the folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = predict_image(model, file_path)
        # Redirect to the result.html template
        return jsonify({'redirectTo': url_for('result', prediction=prediction, image_filename=filename)})
    return jsonify({'error': 'Invalid file type'}), 400


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/result/<prediction>/<image_filename>')
def result(prediction, image_filename):
    # You can use the prediction and image_filename parameters here to customize the rendering of result.html
    # For example, you can pass them to the render_template function
    return render_template('result.html', prediction=prediction, image_filename=image_filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('index'))  # Assuming 'index' is the view function for your main page


if __name__ == '__main__':
    app.run(debug=True)
