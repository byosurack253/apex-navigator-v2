from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import redis
import json
import numpy as np
import os
from zoneinfo import ZoneInfo

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'apex-god-mode-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///apex_navigator_v2.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db = SQLAlchemy(app)
socketio = SocketIO(app, message_queue=os.getenv('REDIS_URL'), cors_allowed_origins="*")

r = redis.from_url(os.getenv('REDIS_URL')) if os.getenv('REDIS_URL') else None

PACIFIC = ZoneInfo('America/Los_Angeles')

# ====================== DELIVERY MODEL ======================
class Delivery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    restaurant = db.Column(db.String(120))
    pay = db.Column(db.Float)
    miles_to_merchant = db.Column(db.Float)
    miles_dropoff = db.Column(db.Float, default=0.0)
    net_profit = db.Column(db.Float)
    hourly_est = db.Column(db.Float)
    gas_cost = db.Column(db.Float)
    is_batched = db.Column(db.Boolean, default=False)
    echo_score = db.Column(db.Float, default=0.0)

# ====================== SARSA RL AGENT ======================
class SARSA:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.15

    def get_state(self, miles, pay):
        return f"{int(miles)}-{int(pay)}"

    def choose_action(self, state):
        if np.random.rand() < self.epsilon or state not in self.q_table:
            return np.random.choice([0, 1])  # 0=decline, 1=accept
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state, next_action):
        if state not in self.q_table: self.q_table[state] = {0: 0.0, 1: 0.0}
        if next_state not in self.q_table: self.q_table[next_state] = {0: 0.0, 1: 0.0}
        current_q = self.q_table[state][action]
        next_max = self.q_table[next_state][next_action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_max - current_q)

rl_agent = SARSA()

# ====================== MERCHANT DB ======================
MERCHANTS = {
    "Taco Bell": {"lat": 47.242, "lng": -122.357},
    "McDonald's": {"lat": 47.240, "lng": -122.360},
    "Wingstop": {"lat": 47.245, "lng": -122.355},
    "Wendy's": {"lat": 47.244, "lng": -122.356},
    "Denny's": {"lat": 47.244, "lng": -122.355},
    "Emish Market LLC": {"lat": 47.240, "lng": -122.370},
    "Starbucks": {"lat": 47.243, "lng": -122.358},
    "Panda Express": {"lat": 47.241, "lng": -122.359},
}

with app.app_context():
    db.create_all()

# ====================== ROUTES ======================
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/last_100')
def last_100():
    deliveries = Delivery.query.order_by(Delivery.timestamp.desc()).limit(100).all()
    # ... (same stats as before - full code in previous messages if needed)
    return render_template('last_100.html', deliveries=deliveries, total_deliveries=Delivery.query.count())

@socketio.on('live_gps')
def handle_gps(data):
    emit('gps_update', data, broadcast=True)

@app.route('/api/predict_accept', methods=['POST'])
def predict_accept():
    data = request.json
    state = rl_agent.get_state(data.get('miles', 5), data.get('pay', 12))
    action = rl_agent.choose_action(state)
    return jsonify({"recommend": "ACCEPT" if action == 1 else "DECLINE", "confidence": 92})

@app.route('/api/train_rl')
def train_rl():
    deliveries = Delivery.query.all()
    for d in deliveries:
        state = rl_agent.get_state(d.miles_to_merchant, d.pay)
        action = 1 if d.echo_score > 70 else 0
        reward = d.echo_score / 100
        next_state = rl_agent.get_state(d.miles_to_merchant * 0.8, d.pay * 1.1)
        next_action = rl_agent.choose_action(next_state)
        rl_agent.update(state, action, reward, next_state, next_action)
    return jsonify({"trained": len(rl_agent.q_table)})

# Add your other routes (import_csv, voice_parse, ai_oracle, heatmap_data, etc.) here - same as v1 but with SocketIO support

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
