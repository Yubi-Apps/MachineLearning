from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import prediction
from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Set allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check valid image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if value:
                return {'Post Values': value}, 201
            return {"error": "Invalid format."}
        except Exception as error:
            return {'error': str(error)}

class GetPredictionOutput(Resource):
    def get(self):
        return {"error": "Invalid Method."}

    def post(self):
        try:
            # Check if an image was uploaded
            if 'file' not in request.files:
                return {'error': 'No file part'}
            
            file = request.files['file']
            
            # If no file is selected
            if file.filename == '':
                return {'error': 'No selected file'}

            # Check if the file is allowed
            if file and allowed_file(file.filename):
                # Save the file to a temporary location
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                file.save(file_path)

                # Preprocess image and make a prediction
                result = prediction.predict_image_class(file_path, prediction.model, prediction.class_names, prediction.img_height, prediction.img_width)
                
                # Return prediction output
                return jsonify({'predicted_class': result})

            return {'error': 'Invalid file format. Only png, jpg, and jpeg are allowed.'}

        except Exception as error:
            return {'error': str(error)}

api.add_resource(Test, '/')
api.add_resource(GetPredictionOutput, '/getPredictionOutput')

if __name__ == "__main__":
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
