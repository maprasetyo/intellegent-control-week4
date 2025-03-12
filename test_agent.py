import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent  # Pastikan file dqn_agent.py berisi agen yang telah dilatih

env = gym.make('CartPole-v1', render_mode="human")  # Pastikan ada render_mode
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen dan load model yang sudah dilatih
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

# Load bobot model jika tersedia
try:
    agent.model.load_weights("dqn_cartpole_weights.h5")
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("No trained model found! Running agent with random weights.")

# Jalankan 5 episode pengujian
for e in range(50):
    state, _ = env.reset(seed=42)
    state = np.array(state, dtype=np.float32).reshape(1, state_size)
    
    for time in range(500):
        env.render()
        action = agent.act(state)  # Ambil aksi berdasarkan model
        next_state, reward, terminated, truncated, _ = env.step(action)  
        done = terminated or truncated  
        
        state = np.array(next_state, dtype=np.float32).reshape(1, state_size)

        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break

env.close()