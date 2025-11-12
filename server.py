import eventlet
# --- THIS IS THE FIX ---
# We MUST monkey_patch() *before* any other modules (like flask, socketio, or habitat) are imported.
eventlet.monkey_patch()

import os
import sys

# --- THIS IS THE FIX ---
# Get the absolute path of the directory containing this script (your project root)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Add this path to the beginning of Python's search path
sys.path.insert(0, PROJECT_ROOT)
# -----------------------

import cv2
import base64
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Import our new simulator class
# Now Python can find 'habitat_llm' because we added PROJECT_ROOT to the path
from habitat_llm.examples.robot_keyboard_control_web import RobotSimulator

# NOTE: eventlet.monkey_patch() was moved to the top of the file.

app = Flask(__name__, static_folder='./frontend', static_url_path='/')
CORS(app, resources={r"/socket.io/*": {"origins": "*"}}) # Allow CORS for Socket.IO
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Global Simulator Instance ---
# We create ONE simulator instance when the server starts.
# This is crucial as initialization is very slow.
print("Starting Robot Simulator... This may take a minute.")
try:
    # Use the same settings you used from the command line
    # Set camera preference to head and robot to stretch
    simulator = RobotSimulator(robot_name='stretch', camera_preference='head')
    print("✅ Robot Simulator is ready.")
except Exception as e:
    print(f"❌ FAILED TO INITIALIZE SIMULATOR: {e}")
    print("Please check configs and data paths.")
    simulator = None
# ---------------------------------

def frame_to_base64(frame):
    """Converts a numpy frame (RGB) to a base64 JPEG string."""
    if frame is None:
        return None
    
    # OpenCV expects BGR, but Habitat gives RGB.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Encode as JPEG
    success, buffer = cv2.imencode('.jpg', frame_bgr)
    if not success:
        print("Failed to encode frame")
        return None
    
    # Convert to base64
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"


@app.route('/')
def index():
    """Serve the simple index.html."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve other static files from the frontend folder."""
    # The React Router fallback is removed.
    # We just serve the file directly.
    return send_from_directory(app.static_folder, path)

# --- WebSocket Event Handlers ---

@socketio.on('connect')
def handle_connect(auth=None):
    """A new user connected. Send them the current frame."""
    print("Client connected")
    if simulator:
        frame_data = frame_to_base64(simulator.get_initial_frame())
        if frame_data:
            emit('video_frame', {'frame': frame_data})
    else:
        print("Simulator not ready, cannot send frame.")

@socketio.on('control')
def handle_control(data):
    """Received a control command from a client."""
    if not simulator:
        print("Simulator not ready, ignoring command.")
        return
        
    key = data.get('key', ' ')
    print(f"Received control key: '{key}'")
    
    # Step the simulation
    new_frame = simulator.step(key)
    
    # Broadcast the new frame to ALL connected clients
    frame_data = frame_to_base64(new_frame)
    if frame_data:
        emit('video_frame', {'frame': frame_data}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {eventlet.getcurrent().name}")


if __name__ == '__main__':
    if simulator is None:
        print("Not starting server because simulator failed to initialize.")
    else:
        print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
        socketio.run(app, host='127.0.0.1', port=5000)