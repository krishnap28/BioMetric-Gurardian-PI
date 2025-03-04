# Bio-Pi V 1.0
# Author: Krishna Prasad PM
# Version: 1.0.0
# Date: November 31, 2024
# Description: Handle Login/Register/Fingerprint operations

import base64
import os
import random
import secrets
import time
import cv2
from flask import Flask, redirect, request, jsonify, render_template, url_for
import firebase_admin
from firebase_admin import credentials, auth, firestore
import requests
from flask import session
from threading import Timer

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 
# Initialize Firebase Admin SDK with service account credentials
cred_path = r'C:\Users\Krishna\Downloads\biopi4-firebase-adminsdk-y9avm-1b6e905af3.json'  
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore client
try:
    db = firestore.client()
    firestore_initialized = True
except Exception as e:
    print(f"Error initializing Firestore: {str(e)}")
    firestore_initialized = False


PI_IP = "http://192.168.137.55:5001/get_gps_data"
fp_ip = "http://192.168.137.55:5001/fp_validate"
en_ip = "http://192.168.137.55:5001/fp_enroll"
ECG_IP = "http://192.168.137.55:5001/start_ecg"
alert_url="http://192.168.137.55:5001/send_sms"

person = {"is_logged_in": True, "name": "", "email": "", "uid": ""}

UPLOAD_FOLDER = r'C:\Users\Krishna\Documents\Bio_Pi'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/signup")
def signup():
    return render_template("signup.html")

def send_to_pi(data):
    print(data)
    try:
        response = requests.post(alert_url, json=data, timeout=5)
        print(f"Sent data to Pi. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to Pi: {e}")


      

def get_combined_data_from_pi():
    try:
        response = requests.get(PI_IP, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to fetch data"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout fetching data from Raspberry Pi"}
    except Exception as e:
        return {"error": str(e)}

    
@app.route("/get_gps_data")
def get_gps_data():
    data = get_combined_data_from_pi()
    if "gps" in data:
        data["gps"]["timestamp"] = time.time()  
    return jsonify(data)

@app.route("/welcome")
def welcome():
    if 'user_email' in session:
        # Retrieve user data from the session
        email = session['user_email']
        name = session['user_name']
        phone = session['user_phone']

        #  fetch GPS data, ECG data
        data = get_combined_data_from_pi()
        latitude = data.get("gps", {}).get("latitude", "N/A")
        longitude = data.get("gps", {}).get("longitude", "N/A")
        heart_rate = data.get("ecg", {}).get("heart_rate", "N/A")
        num_peaks = data.get("ecg", {}).get("num_peaks", "N/A")
        status = data.get("ecg", {}).get("status", "N/A")
        ecg_thrsh = data.get("ecg", {}).get("ecg_threshold", "N/A")

    if data is not None:
        if heart_rate <50:
            print("ECG abnormal threshold detected.")
            # Start a timer to send data to Pi if no user confirmation
            Timer(30, send_to_pi, kwargs={
                "data": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "phone_number": phone
                }
            }).start()

        return render_template(
            "welcome.html",
            email=email,
            name=name,
            phone=phone,
            latitude=latitude,
            longitude=longitude,
            heart_rate=heart_rate,
            num_peaks=num_peaks,
            status=status
        )
    else:
        return redirect(url_for('login'))


# @app.route("/fp_validate", methods=["POST", "GET"])
# def login_with_fingerprint():
#     try:
#         response = requests.post(fp_ip)
#         response_data = response.json()
#         if person["is_logged_in"]:
#             return redirect(url_for('welcome'))

#         if response_data.get('status') == 'success':
#             return jsonify({"status": "success", "message": "Fingerprint validated successfully!"}), 200
#         else:
#             return jsonify({"status": "failure", "message": response_data.get('message', 'Fingerprint validation failed')}), 400

#     except Exception as e:
#         print("Error sending request to Raspberry Pi:", e)
#         return jsonify({"status": "error", "message": "Unable to validate fingerprint. Please try again."}), 500

@app.route('/result', methods=['POST'])
def result():
    print('hereee')
    email = request.form.get('email')
    password = request.form.get('pass')

    # Check if email and password are provided
    if not email or not password:
        return jsonify({
            "status": "error",
            "message": "Email and password are required."
        }), 400

    try:
        # Attempt to authenticate the user with Firebase Auth
        user = auth.get_user_by_email(email)
        print(f"User retrieved: {user}")
        
        user_data = None  # Initialize user_data

        if firestore_initialized:
            print('11111')
            user_ref = db.collection('users').document(user.uid)
            print('2222')
            user_data = user_ref.get()
            print(f"User UID: {user.uid}")
            print(f"User Email: {user.email}")
            print(f"User Display Name: {user.display_name}")
            print(f"User Email Verified: {user.email_verified}")

            # Check if the Firestore document exists
            if user_data.exists:
                print('333')
                user_data_dict = user_data.to_dict()
                print('444')

                # Store necessary user data in session
                session['user_email'] = user.email
                session['user_name'] = user_data_dict.get('name')
                session['user_phone'] = user_data_dict.get('phone')

            else:
                # Handle case where user data does not exist
                return jsonify({
                    "status": "error",
                    "message": "User data not found in Firestore."
                }), 404

        return jsonify({
            "status": "success",
            "redirect_url": "/welcome"  # Redirect to welcome page after success
        }), 200

    except firebase_admin.auth.UserNotFoundError:
        return jsonify({
            "status": "error",
            "message": "User not found. Please check the email."
        }), 400

    except Exception as e:
        print(f"Error: {e}")  
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 400



    except Exception as e:
        # Catch all other exceptions
        print(f"Error during user login: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred. Please try again later."
        }), 500

@app.route('/register', methods=['POST'])
def register_user():
    # Get form data
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('pass')
    phone = request.form.get('phone')

    # Check for missing fields
    if not name or not email or not password:
        return jsonify({"status": "error", "message": "All fields are required."}), 400

    try:
        user = auth.create_user(email=email, password=password, display_name=name)
        if firestore_initialized:
            db.collection('users').document(user.uid).set({
                "name": name,
                "email": email,
                "phone": phone
            })
        return jsonify({
            "status": "success",
            "redirect_url": "/welcome" 
        }), 200

    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return jsonify({"status": "error", "message": f"Registration failed: {str(e)}"}), 400


@app.route("/fp_enroll", methods=["POST", "GET"])
def enroll_fingerprint():
    try:
        response = requests.post(en_ip)  
        response_data = response.json()
        print("Fingerprint enrollment response:", response_data)
        if response_data.get('status') == 'success':
            return jsonify({"status": "success", "message": "Fingerprint enrolled successfully!", "id": response_data.get("id")}), 200
        else:
            return jsonify({"status": "failure", "message": response_data.get('message', 'Fingerprint enrollment failed')}), 400

    except Exception as e:
        print("Error sending request to Raspberry Pi:", e)
        return jsonify({"status": "error", "message": "Unable to enroll fingerprint. Please try again."}), 500
       # return jsonify({"status": "success", "message": "Fingerprint enrolled successfully!", "id": '45'}), 200
@app.route("/capture_face", methods=["POST"])
def capture_face():
    try:
        # Get the image data and email from the request
        data = request.json
        image_data = data.get("image")
        email = data.get("email")

        if not email:
            print("Error: Email is missing in the request.")
            return jsonify({"status": "error", "message": "Email is required to save the face image."}), 400

        if not image_data:
            print("Error: Image data is missing in the request.")
            return jsonify({"status": "error", "message": "Image data is required to capture the face."}), 400

        # Decode the Base64 image
        header, encoded = image_data.split(',', 1)  
        decoded_image = base64.b64decode(encoded)

        # Create the directory if it doesn't exist
        faces_dir = "static/faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
        
        # Sanitize email for filename
        sanitized_email = email.replace("@", "_at_").replace(".", "_dot_")  
        image_path = os.path.join(faces_dir, f"{sanitized_email}_captured_face.jpg")

        # Save the image
        with open(image_path, "wb") as f:
            f.write(decoded_image)

        print(f"Face image saved successfully at: {image_path}")

        return jsonify({"status": "success", "message": "Face captured and saved successfully!", "path": image_path}), 200

    except Exception as e:
        print("Error saving captured face:", e)
        return jsonify({"status": "error", "message": "Failed to save face image."}), 500


@app.route("/validate_face", methods=["POST"])#
def validate_face():
    try:
        faces_dir = "static/faces"

        if not os.path.exists(faces_dir):
            return jsonify({"status": "error", "message": "Face directory not found."}), 500

        # Get all saved face image filenames
        face_files = [f for f in os.listdir(faces_dir) if f.endswith("_captured_face.jpg")]

        if not face_files:
            return jsonify({"status": "error", "message": "No saved face images found."}), 400

        # Initialize the camera
        video_capture = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        recognized_email = None
        max_attempts = 100  # Limit the number of frames to process

        while max_attempts > 0:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y + h, x:x + w]

                # Compare with all saved faces
                for face_file in face_files:
                    # Extract email from filename
                    email_from_filename = face_file.replace("_at_", "@").replace("_dot_", ".").replace("_captured_face.jpg", "")
                    
                    # Print the extracted email for debugging
                    print("Extracted email from filename:", email_from_filename)

                    saved_image_path = os.path.join(faces_dir, face_file)
                    saved_image = cv2.imread(saved_image_path)
                    saved_image_gray = cv2.cvtColor(saved_image, cv2.COLOR_BGR2GRAY)

                    # Resize the captured ROI to match the saved image size
                    face_roi_resized = cv2.resize(face_roi, (saved_image_gray.shape[1], saved_image_gray.shape[0]))

                    # Compare the face with the saved face using SSIM
                    similarity = ssim(saved_image_gray, face_roi_resized)

                    if similarity > 0.6:  # Set a more appropriate threshold for similarity
                        recognized_email = email_from_filename
                        break

                if recognized_email:
                    break

            max_attempts -= 1
            if recognized_email:
                break

        video_capture.release()

        if recognized_email:
            return jsonify({"status": "success", "message": f"Face recognized successfully!", "email": recognized_email}), 200
        else:
            return jsonify({"status": "failure", "message": "Face unrecognized."}), 400

    except Exception as e:
        print("Error in face validation:", e)
        return jsonify({"status": "error", "message": "An error occurred during face validation."}), 500

if __name__ == "__main__":
    app.run()
 