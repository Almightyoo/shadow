# ♟️ Shadow - Neural Network Guided Chess Engine

**Shadow** is a neural network-based chess engine inspired by AlphaZero and Leela Chess. It combines the power of **Monte Carlo Tree Search (MCTS)** with deep learning to play strong, creative, and human-like chess.

> 🔥 Estimated strength: ~2500 Elo on Chess.com  
> ✅ Self-play trained | Neural net guided | MCTS driven  
> 🧠 Similar in spirit to AlphaZero / Leela Chess

![Sample Game](https://github.com/user-attachments/assets/64cf80f3-5472-41f9-a88f-0837634d0787)

---

## 🚀 Features

- 🌲 **MCTS** for move selection and long-term planning
- 🧠 **Neural network** evaluation for position scoring and move probabilities
- 🔁 **Self-play training** pipeline
- ♚ **Modern playstyle** that mimics human intuition
- 🛠️ **Custom PGN parsing**, **legal move masking**, and optimized batch processing
- 📈 Designed for future improvements and model scaling

---

## 📷 Sample Game (Shadow vs Shadow)

> Chess, when played perfectly, is a draw — though this match was not perfect, it reflects the engine's strategic depth.

🎮 [View PGN and game details](https://github.com/user-attachments/assets/07bf209c-43db-448d-8692-9f28efcbb79c)

---

## 🧬 Inspiration

Shadow is built upon the principles of:
- [AlphaZero (DeepMind)](https://deepmind.com/research/highlighted-research/alphago)
- [Leela Chess Zero](https://lichess.org/team/lc0-leela-chess-zero)

It simplifies the architecture while keeping core strengths of these systems — learning through **self-play**, using **policy and value networks**, and relying on **search instead of brute force**.

---

## 🛠️ Technical Overview

- **Language:** Python
- **Core libraries:** PyTorch, python-chess, NumPy
- **Architecture:** Policy + Value head neural network
- **Search:** Monte Carlo Tree Search (PUCT)

---

## 📦 Installation

```bash
git clone git@github.com:Almightyoo/shadow.git
cd shadow
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
