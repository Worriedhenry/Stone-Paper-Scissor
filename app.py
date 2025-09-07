import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import random
from PIL import Image
from utils.gesture_model import GestureMLP

# -------------------- LSTM Model Setup --------------------
# Move encoding
move_to_int = {'Stone': 0, 'Rock': 0, 'Paper': 1, 'Scissors': 2}
int_to_move = {0: 'Stone', 1: 'Paper', 2: 'Scissors'}

# Custom Dataset
class RPSDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=3, dropout=0.5):
        super(LSTMPredictor, self).__init__()
        self.embedding = nn.Embedding(3, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.long().squeeze(-1)
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# Prediction function with padding
def predict_next_move(model, past_moves, seq_len):
    model.eval()
    with torch.no_grad():
        if len(past_moves) < seq_len:
            padded_moves = [0] * (seq_len - len(past_moves)) + past_moves
        else:
            padded_moves = past_moves[-seq_len:]
        seq = torch.tensor([padded_moves], dtype=torch.float32).unsqueeze(-1)
        pred = model(seq)
        predicted_int = torch.argmax(pred, dim=1).item()
        return predicted_int

# AI chooses counter
def choose_ai_move(predicted_user_int):
    counters = {0: 1, 1: 2, 2: 0}  # Stone -> Paper, Paper -> Scissors, Scissors -> Stone
    return int_to_move[counters[predicted_user_int]]

# Initialize LSTM model
seq_len = 10
lstm_model = LSTMPredictor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
epochs = 50

# -------------------- Original Setup --------------------
model_path='gesture_model_2708.pth'
scaler_mean_path='scaler_mean_2708.npy'
scaler_scale_path='scaler_scale_2708.npy'
CLASSES = ['Stone', 'Paper', 'Scissors', 'Thumbs up', 'Thumbs down', 'other']

model = GestureMLP(input_size=42)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
if 'score' not in st.session_state:
    st.session_state['score'] = {'User': 0, 'AI': 0, 'Draw': 0}
if 'history' not in st.session_state:
    st.session_state['history'] = []  # Each entry: (user_move, ai_move, winner)
score = st.session_state['score']
history = st.session_state['history']

scaler_mean = np.load(scaler_mean_path)
scaler_scale = np.load(scaler_scale_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# -------------------- Assets --------------------
stone_img = Image.open('Assets/Stone.jpg')
paper_img = Image.open('Assets/Paper.jpg')
scissors_img = Image.open('Assets/Scissor.jpg')
thumbs_up_img = Image.open('Assets/Thumbs_up.jpg')
thumbs_down_img = Image.open('Assets/Thumbs_down.jpg')
gesture_imgs = {'Stone': stone_img, 'Paper': paper_img, 'Scissors': scissors_img}
gesture_list = [stone_img, paper_img, scissors_img]

# -------------------- Styling --------------------
st.set_page_config(page_title="Rock Paper Scissors", layout="wide")

st.markdown("""
<style>
body { background-color: #111; color: #fff; }
.css-18e3th9 { padding: 1rem 2rem; background-color: #121212; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
button, .stButton button {
    background-color: #00acc1;
    color: white;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Functions --------------------
def preprocess_landmarks(landmarks):
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y])
    coords = np.array(coords).reshape(1, -1)
    coords = (coords - scaler_mean) / scaler_scale
    return torch.tensor(coords, dtype=torch.float32).to(device)

def get_majority_gesture(buffer):
    filtered = [g for g in buffer if g in ['Stone', 'Paper', 'Scissors', 'Thumbs up', 'Thumbs down']]
    if not filtered:
        return None
    vals, counts = np.unique(filtered, return_counts=True)
    return vals[np.argmax(counts)] if max(counts) >= 1 else None

def determine_winner(user_move, ai_move):
    if user_move == ai_move:
        return 'Draw'
    if (user_move == 'Stone' and ai_move == 'Scissors') or \
       (user_move == 'Scissors' and ai_move == 'Paper') or \
       (user_move == 'Paper' and ai_move == 'Stone'):
        return 'User'
    return 'AI'

def run_countdown(col):
    img_placeholder = col.empty()
    text_placeholder = col.empty()
    start = time.time()
    while time.time() - start < 3:
        img = random.choice(gesture_list)
        img_placeholder.image(img, width=200)
        remaining = int(3 - (time.time() - start))
        text_placeholder.markdown(f"### Countdown: {remaining}...")
        time.sleep(1)
    img_placeholder.empty()
    text_placeholder.empty()

def home_page():
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        cam = st.empty()
    with col2:
        st.subheader("✋ Show thumbs up to START, thumbs down to QUIT")
        st.image(thumbs_up_img, caption="Thumbs Up = Start", width=180)
        st.image(thumbs_down_img, caption="Thumbs Down = Quit", width=180)
    status = st.empty()
    cap = cv2.VideoCapture(0)
    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        gesture = "Show hand..."
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            input_tensor = preprocess_landmarks(landmarks)
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                gesture = CLASSES[idx.item()] if conf.item() > 0.8 else "Uncertain"
                if gesture in ['Thumbs up', 'Thumbs down']:
                    buffer.append(gesture)
                    buffer = buffer[-8:]

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cam.image(frame, channels="BGR")

        if buffer.count("Thumbs up") >= 5:
            st.session_state.app_phase = "game"
            st.rerun()
        elif buffer.count("Thumbs down") >= 5:
            status.warning("👋 Exiting...")
            time.sleep(2)
            st.stop()

        time.sleep(0.05)
    cap.release()

def game_loop():
    # Three columns: Scoreboard/history | Live feed/AI move | Instructions
    col1, col2, col3 = st.columns([1.2, 1.5, 1.1])

    # --- col1: Scoreboard and match history ---
    with col1:
        st.markdown(
            f"""
            <div style='
                background: #222;
                color: #fff;
                padding: 1rem 0.5rem;
                margin-bottom: 1.2rem;
                border-radius: 0.7rem;
                font-size: 1.2rem;
                font-weight: bold;
                letter-spacing: 1px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12);
                text-align: center;
            '>
                🧍 User: {score['User']} &nbsp; | &nbsp; 🤖 AI: {score['AI']} &nbsp; | &nbsp; 🤝 Draw: {score['Draw']}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("#### Match History")
        if history:
            st.table(
                [
                    {"Round": i+1, "User": u, "AI": a, "Winner": w}
                    for i, (u, a, w) in enumerate(history)
                ]
            )
        else:
            st.info("No matches played yet.")

    # --- col2: Live feed (top), AI move (middle) ---
    with col2:
        live_feed = st.empty()
        ai_move_box = st.empty()

    # --- col3: Instructions ---
    with col3:
        st.markdown("### Instructions")
        st.image(thumbs_up_img, caption="Thumbs Up = Play/Pause", width=120)
        st.image(thumbs_down_img, caption="Thumbs Down = Exit", width=120)
        st.markdown("""
        - Show <b> Stone, Paper, Scissors </b> to play at end of countdown<br>            
        - Show <b>Thumbs Up</b> to play/pause<br>
        - Show <b>Thumbs Down</b> to exit
        """, unsafe_allow_html=True)

    cap = cv2.VideoCapture(0)
    buffer = []

    # Countdown and start prediction
    with col2:
        st.markdown("### Countdown starting... Get ready!")
        # Train/predict during countdown
        user_moves = [move_to_int[u] for u, _, _ in history if u in move_to_int]
        ai_move = None
        if len(history) < 4:  # Rounds 1-4
            ai_move = random.choice(['Stone', 'Paper', 'Scissors'])
        else:
            # Train if enough data (need seq_len + 1 for one sequence-target pair)
            if len(user_moves) >= seq_len + 1:
                sequences = []
                targets = []
                for i in range(len(user_moves) - seq_len):
                    sequences.append(user_moves[i:i + seq_len])
                    targets.append(user_moves[i + seq_len])
                sequences = np.array(sequences)
                targets = np.array(targets)
                dataset = RPSDataset(sequences, targets)
                dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
                
                # Training loop
                for epoch in range(epochs):
                    for seq, target in dataloader:
                        optimizer.zero_grad()
                        output = lstm_model(seq)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
            
            # Predict
            predicted_user_int = predict_next_move(lstm_model, user_moves, seq_len)
            ai_move = choose_ai_move(predicted_user_int)
        
        run_countdown(st)

    # Reset buffer for this round
    buffer.clear()

    # Capture user move
    start = time.time()
    frames = []
    while time.time() - start < 0.6:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            input_tensor = preprocess_landmarks(results.multi_hand_landmarks[0])
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                cls = CLASSES[idx.item()]
                if conf.item() > 0.8:
                    buffer.append(cls)
        live_feed.image(frame, channels="BGR")
        time.sleep(0.05)

    user_move = get_majority_gesture(buffer)
    if user_move == 'Thumbs up':
        time.sleep(1)
        post_round_gesture(live_feed, st, cap)
        return
    if user_move == 'Thumbs down':
        st.warning("👋 Exiting...")
        time.sleep(2)
        st.session_state.app_phase = 'home'
        return
    if not user_move or user_move not in ['Stone', 'Paper', 'Scissors']:
        with col2:
            st.error("Could not detect stable gesture. Try again.")
            st.warning("Please show a clear gesture for at least 0.6 seconds.")
            st.rerun()
        return

    # Use the precomputed AI move

    ai_move_box.markdown(f"### 🤖 AI Move: <span style='color:#00acc1;font-size:1.5rem'>{ai_move}</span> YOUR Move: <span style='color:#00acc1;font-size:1.5rem'>{user_move}</span>", unsafe_allow_html=True)

    winner = determine_winner(user_move, ai_move)
    if winner == 'User':
        score['User'] += 1
        with col2:
            st.success("✅ You Win!")
    elif winner == 'AI':
        score['AI'] += 1
        with col2:
            st.error("❌ AI Wins!")
    else:
        score['Draw'] += 1
        with col2:
            st.info("🤝 It's a Draw!")

    time.sleep(2)

    # Update match history (append only if valid RPS move)
    if not history or history[-1] != (user_move, ai_move, winner):
        history.append((user_move, ai_move, winner))
    st.rerun()

def post_round_gesture(cam, status, cap):
    cam.empty()
    post_buffer = []
    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            input_tensor = preprocess_landmarks(results.multi_hand_landmarks[0])
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                cls = CLASSES[idx.item()]
                if conf.item() > 0.8 and cls in ['Thumbs up', 'Thumbs down']:
                    post_buffer.append(cls)
                    post_buffer = post_buffer[-15:]
        cam.image(frame, channels="BGR")
        if post_buffer.count("Thumbs up") >= 10:
            st.rerun()
        elif post_buffer.count("Thumbs down") >= 10:
            status.warning("👋 Exiting...")
            time.sleep(2)
            st.stop()
        time.sleep(0.05)

# -------------------- App Control --------------------
if 'app_phase' not in st.session_state:
    st.session_state.app_phase = 'home'

st.title("🕹️ Rock Paper Scissors — Gesture Edition")

if st.session_state.app_phase == 'home':
    home_page()
elif st.session_state.app_phase == 'game':
    game_loop()