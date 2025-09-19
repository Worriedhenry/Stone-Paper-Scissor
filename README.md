# Rock Paper Scissors Gesture Game

A modern, interactive Rock Paper Scissors game built with Streamlit, utilizing hand gesture recognition via MediaPipe and a pre-trained PyTorch model. The app features a sleek, non-scrollable dark-mode UI with neon accents, ensuring all information fits within a single screen (1280x720 resolution). Players use hand gestures (Thumbs Up, Thumbs Down, Rock, Paper, Scissors) to navigate and play, with a live webcam feed that remains active throughout.

## Features

### General
- **Hand Gesture Recognition**: Uses MediaPipe Hands and a pre-trained `GestureMLP` model to detect six gestures: `Stone`, `Paper`, `Scissors`, `Thumbs up`, `Thumbs down`, `other`.
- **Live Webcam Feed**: Displays a 640x480 feed (scaled to ~350px height) with gesture detection text, active in all phases without freezing.
- **Non-Scrollable UI**: All elements (scoreboard, instructions, gesture bar, feed, status) fit within a 1280x720 window.
- **Dark Mode**: Modern design with a black background (`#1a1a1a`), gradient scoreboard, neon blue (`#00b7eb`) and green (`#00ff88`) accents, and hover effects on gesture images.
- **Performance**: Targets ~30 FPS for smooth operation.

### Home Page
- **Gestures**: Thumbs Up (start game), Thumbs Down (quit).
- **UI Elements**:
  - Scoreboard (40px height, gradient blue).
  - Instructions (80px height, 12px font, neon blue header).
  - Gesture bar (two 60px images: Thumbs Up, Thumbs Down).
  - Live feed with gesture text (e.g., `Thumbs up (0.85)`).
  - Status message (neon green).

### Game Page
- **Game Flow**:
  - Wait for Thumbs Up to start a round (10-second timeout).
  - 3-second countdown with animated text.
  - Capture Rock, Paper, or Scissors for 1 second.
  - Display result, AI move, and updated scoreboard.
  - Post-round: Thumbs Up to replay, Thumbs Down to return to home (10-second timeout).
- **UI Elements**:
  - Scoreboard at top.
  - Instructions listing all gestures.
  - Gesture bar (five 60px images: Thumbs Up, Thumbs Down, Rock, Paper, Scissors; AI move shown post-round).
  - Live feed with gesture text.
  - Status messages (win: green, loss: red, draw: blue).
- **Data Clearing**: Resets AI move and gesture bar on replay.

### Technical Details
- **Model**: `GestureMLP` trained on the [Stone Paper Scissors Hand Landmarks Dataset](https://www.kaggle.com/datasets/aryan7781/stone-paper-scissors-hand-landmarks-dataset) with landmarks in `[0_x, 0_y, ..., 20_x, 20_y]` order.
- **Robustness**: Local MediaPipe `Hands` initialization, `st.rerun()` for transitions (Streamlit ≥1.12.0), error handling for `ValueError: Closing SolutionBase._graph`.
- **Single Feed**: Only the main webcam feed is shown, no secondary images.

## Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: Webcam required for gesture detection
- **Files**:
  - Model: `gesture_model.pth`, `scaler_mean.npy`, `scaler_scale.npy`
  - Assets: `Assets/Stone.jpg`, `Assets/Paper.jpg`, `Assets/Scissor.jpg`, `Assets/Thumbs_up.jpg`, `Assets/Thumbs_down.jpg`

## Installation

1. **Clone the Repository** (if hosted on GitHub):
   ```bash
   git clone <repository-url>
   cd rock-paper-scissors
