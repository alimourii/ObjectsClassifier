from flask import Flask, render_template, request # type: ignore
from flask_socketio import SocketIO, emit # type: ignore
import cv2 # type: ignore
import base64
import os
from datetime import datetime

import model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

current_frame = None
object_name = None
image_count = 0
OurModel = None
Mode = "Learn"

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use the first webcam on your system
    global current_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            current_frame = frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_feed', {'frame': frame})
            
                
            

                
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")  # Debug print
    socketio.start_background_task(gen_frames)

@socketio.on('take_photo')
def handle_take_photo():
    global current_frame, object_name, image_count
    if current_frame is not None and object_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = os.path.join('photos', object_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_count += 1
        photo_path = os.path.join(dir_path, f'{object_name}_{image_count}.jpg')
        cv2.imwrite(photo_path, current_frame)
        print(f"Photo saved to {photo_path}")

@socketio.on('set_object_name')
def handle_set_object_name(data):
    global object_name, image_count
    object_name = data['object_name']
    image_count = 0
    print(f"Object name set to: {object_name}")
    
@socketio.on('train_model')
def handle_train_model():
    global OurModel
    OurModel = model.Model()
    OurModel.train_model('photos')
    emit('training_complete')
    

@socketio.on('Predict')
def handle_Predict():
    global current_frame, Mode
    emit('prediction_result', {'result': OurModel.predict(current_frame)})
    
@socketio.on('SaveModel')
def handle_SaveModel():
    OurModel.save_model()

if __name__ == '__main__':
    if not os.path.exists('photos'):
        os.makedirs('photos')
    socketio.run(app, debug=True)
