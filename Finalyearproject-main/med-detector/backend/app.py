from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import jwt
import datetime
from functools import wraps
import bcrypt
from dotenv import load_dotenv
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load the trained model
MODEL_PATH = "models/Lung_model.keras"  # Ensure you save the trained model as model.h5
model = load_model(MODEL_PATH)
app = Flask(__name__)
CORS(app)

# Class labels from your model training
LABELS = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']



def preprocess_image(image_path):
    """Preprocess the image before feeding into the model."""
    img_height, img_width = 128, 128  # Ensure these match your trained model input size

    img = load_img(image_path, target_size=(img_height, img_width))  # Use same method as manual test
    img = img_to_array(img)  # Convert to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension]

    return img




load_dotenv()


# MongoDB Connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client.med_detector
users_collection = db.users

# JWT Secret Key
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "supersecretkey")

# Admin Credentials
ADMIN_USERS = [
    {"username": "ragu", "password": "ragu123"},
    {"username": "ji", "password": "ji"},
    {"username": "ai", "password": "ai123"}
]

# Initialize admin users in the database
def initialize_admins():
    for admin in ADMIN_USERS:
        existing_admin = users_collection.find_one({"username": admin["username"]})
        if not existing_admin:
            hashed_password = bcrypt.hashpw(admin["password"].encode('utf-8'), bcrypt.gensalt())
            users_collection.insert_one({
                "username": admin["username"],
                "password": hashed_password,
                "role": "admin",
                "email": f"{admin['username']}@meddetector.com"
            })

# JWT Token Required Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"message": "Token is missing!"}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = users_collection.find_one({"username": data['username']})
            if not current_user:
                return jsonify({"message": "User not found!"}), 401
        except:
            return jsonify({"message": "Token is invalid!"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Routes
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        token = jwt.encode({
            'username': username,
            'role': user['role'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, app.config['SECRET_KEY'])
        return jsonify({
            "token": token, 
            "role": user['role'],
            "username": user['username'],
            "approved": user.get('approved', True)  # Admins are always approved
        })
    return jsonify({"message": "Invalid credentials!"}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    
    # Check if username already exists
    existing_user = users_collection.find_one({"username": data['username']})
    if existing_user:
        return jsonify({"message": "Username already exists!"}), 400
    
    # Check if email already exists
    existing_email = users_collection.find_one({"email": data['email']})
    if existing_email:
        return jsonify({"message": "Email already exists!"}), 400
    
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    user = {
        "username": data['username'],
        "email": data['email'],
        "password": hashed_password,
        "role": "doctor",
        "hospital_name": data['hospital_name'],
        "contact_number": data['contact_number'],
        "approved": False,
        "created_at": datetime.datetime.utcnow()
    }
    users_collection.insert_one(user)
    return jsonify({"message": "Doctor registered successfully! Please wait for admin approval."})

@app.route('/admin/doctors', methods=['GET'])
@token_required
def get_doctors(current_user):
    if current_user['role'] != 'admin':
        return jsonify({"message": "Unauthorized!"}), 403
        
    doctors = list(users_collection.find({"role": "doctor"}, {"password": 0}))
    for doctor in doctors:
        doctor['_id'] = str(doctor['_id'])
    return jsonify(doctors)

@app.route('/admin/approve/<doctor_id>', methods=['PUT'])
@token_required
def approve_doctor(current_user, doctor_id):
    if current_user['role'] != 'admin':
        return jsonify({"message": "Unauthorized!"}), 403
        
    result = users_collection.update_one(
        {"_id": ObjectId(doctor_id), "role": "doctor"}, 
        {"$set": {"approved": True}}
    )
    
    if result.modified_count > 0:
        return jsonify({"message": "Doctor approved successfully!"})
    return jsonify({"message": "Doctor not found or already approved!"}), 404

@app.route('/admin/delete/<doctor_id>', methods=['DELETE'])
@token_required
def delete_doctor(current_user, doctor_id):
    if current_user['role'] != 'admin':
        return jsonify({"message": "Unauthorized!"}), 403
        
    result = users_collection.delete_one({"_id": ObjectId(doctor_id), "role": "doctor"})
    
    if result.deleted_count > 0:
        return jsonify({"message": "Doctor deleted successfully!"})
    return jsonify({"message": "Doctor not found!"}), 404

@app.route('/doctor/profile', methods=['GET'])
@token_required
def get_doctor_profile(current_user):
    if current_user['role'] != 'doctor':
        return jsonify({"message": "Unauthorized!"}), 403
    
    doctor = {
        "username": current_user['username'],
        "email": current_user['email'],
        "hospital_name": current_user.get('hospital_name', ''),
        "contact_number": current_user.get('contact_number', ''),
        "approved": current_user.get('approved', False)
    }
    
    return jsonify(doctor)

@app.route('/doctor/upload', methods=['POST'])
@token_required
def upload_file(current_user):
    if current_user['role'] != 'doctor':
        return jsonify({"message": "Unauthorized!"}), 403
    
    if not current_user.get('approved', False):
        return jsonify({"message": "Your account is not approved yet!"}), 403
    
    # Here you would handle file upload logic
    # For now, we'll just return a success message
    return jsonify({"message": "File uploaded successfully!"})
@app.route('/doctor/predict', methods=['POST'])
@token_required
def predict_disease(current_user):
    if current_user['role'] != 'doctor':
        return jsonify({"message": "Unauthorized!"}), 403

    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded!"}), 400

    file = request.files['file']
    file_path = f"static/uploads/{file.filename}"
    file.save(file_path)

    # Preprocess and predict
    img = preprocess_image(file_path)
    prediction = model.predict(img)

    print(f"Raw model output: {prediction}")  # DEBUGGING LINE

    class_index = np.argmax(prediction)
    disease = LABELS[class_index]

    return jsonify({
        "message": "Prediction successful!",
        "disease": disease,
        "confidence": float(prediction[0][class_index]),
        "confidence_values": prediction[0].tolist()  # Send full probability distribution
    })


# Initialize server
# Using with_app_context pattern instead of before_first_request
with app.app_context():
    initialize_admins()

if __name__ == '__main__':
    app.run(debug=True)