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
                predicted_class = prediction.predict_image_class(
                    file_path, 
                    prediction.model, 
                    prediction.class_names, 
                    prediction.img_height, 
                    prediction.img_width
                )

                # Generate description based on predicted class
                if predicted_class == 'SL17 Phthorimaea operculella (Zeller)':
                    description = "Phthorimaea operculella, commonly known as the potato tuber moth, is a pest of potato crops worldwide."
                elif predicted_class == 'SL15 Myzus persicae (Sulzer)':
                    description = "Myzus persicae, the green peach aphid, is a major agricultural pest affecting various plants and crops."
                elif predicted_class == 'SL01 Agrotis ipsilon (Hufnagel)':
                    description = "Agrotis ipsilon, or the black cutworm, is a significant pest of young plants, especially maize and wheat."
                elif predicted_class == 'SL05 Bemisia tabaci (Gennadius)':
                    description = "Bemisia tabaci, the whitefly, is a sap-sucking insect known for transmitting plant viruses."
                elif predicted_class == 'SL10 Epilachna vigintioctopunctata (Fabricius)':
                    description = "Epilachna vigintioctopunctata, or the 28-spotted ladybird, feeds on solanaceous crops, particularly potatoes."
                elif predicted_class == 'SL03 Aphis gossypii Glover':
                    description = "Aphis gossypii, also called the cotton aphid, is a common pest in cotton and melon crops."
                elif predicted_class == 'SL06 Brachytrypes portentosus Lichtenstein':
                    description = "Brachytrypes portentosus, known as the large cricket, is found in agricultural and forested habitats."
                elif predicted_class == 'SL02 Amrasca devastans (Distant)':
                    description = "Amrasca devastans, or the cotton leafhopper, severely impacts cotton and vegetable crops."
                else:
                    description = "No description available for this class."

                # Return prediction output with description
                return jsonify({'predicted_class': predicted_class, 'description': description})

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
