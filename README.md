# Self_Evolving_RL_PID-AGV
Reinforcement Learning as an adaptive layer over classical industrial control.

Self-evolving Reinforcement Learning agent for adaptive PID tuning in AGV trajectory tracking.
A lightweight, interpretable RL-based approach where the agent dynamically adjusts PID gains online to improve trajectory tracking accuracy and stability under varying conditions.
The project focuses on low-complexity simulation, reproducibility, and industrially compatible control strategies.


[ Ambiente AGV ]
      ↑       ↓
  Estado   Recompensa
      ↑       ↓
 [ Agente RL ]
      ↓
 Ajuste de Ganhos PID

 rl-adaptive-pid/
│
├── env/
│   └── agv_env.py
│
├── agent/
│   └── dqn_agent.py
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md



“Isso não é um brinquedo de RL. É um controlador clássico que aprende.”
