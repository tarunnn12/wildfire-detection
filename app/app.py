from flask import Flask, render_template, Response, jsonify
from detector import FireDetector
import cv2
import threading
import time
import datetime
import requests

DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1517114707172003841/vlLIcEgY3fTZfXgDzrd2tnN7ccAh8YBvJs9DL3Kr-b9rKaAVOLiN3GfF1mVGedvCYy3x"
SUSTAINED_FRAMES_REQUIRED = 5
ALERT_COOLDOWN_SECONDS = 30

alert_state = {
    'consecutive_fire': 0,
    'consecutive_smoke': 0,
    'last_alert_time': 0,
}

def send_combined_alert(fire_dets, smoke_dets):
    def _send():
        try:
            lines = ["🚨 **WILDFIRE DETECTION ALERT** 🚨"]

            if fire_dets:
                max_fire = max(d['conf'] for d in fire_dets)
                lines.append(f"🔥 **FIRE DETECTED** — confidence {max_fire:.0%}")

            if smoke_dets:
                max_smoke = max(d['conf'] for d in smoke_dets)
                lines.append(f"💨 **SMOKE DETECTED** — confidence {max_smoke:.0%}")

            lines.append("— Wildfire Detection System")

            payload = {"content": "\n".join(lines)}
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            print(f"Discord alert failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def check_and_alert(detections):
    classes = [d['class'] for d in detections]
    now = time.time()

    if 'fire' in classes:
        alert_state['consecutive_fire'] += 1
    else:
        alert_state['consecutive_fire'] = 0

    if 'smoke' in classes:
        alert_state['consecutive_smoke'] += 1
    else:
        alert_state['consecutive_smoke'] = 0

    cooldown_ok = (now - alert_state['last_alert_time']) > ALERT_COOLDOWN_SECONDS

    if alert_state['consecutive_fire'] == SUSTAINED_FRAMES_REQUIRED and cooldown_ok:
        fire_conf = max([d['conf'] for d in detections if d['class'] == 'fire'], default=0)
        send_discord_alert('fire', fire_conf)
        alert_state['last_alert_time'] = now

    elif alert_state['consecutive_smoke'] == SUSTAINED_FRAMES_REQUIRED and cooldown_ok:
        smoke_conf = max([d['conf'] for d in detections if d['class'] == 'smoke'], default=0)
        send_discord_alert('smoke', smoke_conf)
        alert_state['last_alert_time'] = now

app = Flask(__name__)

detector = FireDetector()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

state = {
    'detections': [],
    'frame_count': 0,
    'fire_count': 0,
    'smoke_count': 0,
    'last_alert': None,
    'status': 'monitoring',
    'fps': 0,
}
lock = threading.Lock()


def generate_frames():
    prev_time = time.time()
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame, dets = detector.predict(frame)

        now = time.time()
        fps = 1 / (now - prev_time + 1e-9)
        prev_time = now

        with lock:
            state['detections'] = dets
            state['frame_count'] += 1
            state['fps'] = round(fps, 1)
            classes = [d['class'] for d in dets]
            if 'fire' in classes:
                state['fire_count'] += 1
                state['status'] = 'fire'
                state['last_alert'] = datetime.datetime.now().strftime('%H:%M:%S')
            elif 'smoke' in classes:
                state['smoke_count'] += 1
                state['status'] = 'smoke'
                state['last_alert'] = datetime.datetime.now().strftime('%H:%M:%S')
            else:
                state['status'] = 'clear'

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def get_state():
    with lock:
        return jsonify(state.copy())


@app.route('/api/send_alert', methods=['POST'])
def send_alert():
    with lock:
        current_dets = state['detections']

    if not current_dets:
        return jsonify({'sent': False, 'message': 'No active detections to alert on'})

    fire_dets = [d for d in current_dets if d['class'] == 'fire']
    smoke_dets = [d for d in current_dets if d['class'] == 'smoke']

    send_combined_alert(fire_dets, smoke_dets)

    with lock:
        state['last_alert'] = datetime.datetime.now().strftime('%H:%M:%S')

    classes_found = ', '.join(set(d['class'] for d in current_dets))
    return jsonify({'sent': True, 'message': f"Alert sent: {classes_found}"})


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
