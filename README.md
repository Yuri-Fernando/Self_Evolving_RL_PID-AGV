Self_Evolving_RL_PID-AGV

Reinforcement Learning as an adaptive layer over classical industrial control.

Descrição do Projeto

Este projeto implementa um agente de Aprendizado por Reforço (RL) auto-evolutivo para o ajuste adaptativo de ganhos PID (Proporcional-Integral-Derivativo) em um Veículo Guiado Automaticamente (AGV) simulado. O objetivo é aprimorar a precisão e a estabilidade do rastreamento de trajetória sob condições operacionais variáveis.

A abordagem foca em:

•
Interpretabilidade: Utiliza o RL para modular um controlador clássico (PID), mantendo a base de controle compreensível.

•
Baixa Complexidade: Simulação simplificada para foco no algoritmo de controle.

•
Compatibilidade Industrial: Estratégia de controle que pode ser facilmente integrada em sistemas de controle industrial existentes.

Arquitetura do Sistema

O Agente RL atua como uma camada de meta-controle, ajustando os parâmetros do PID com base no desempenho do AGV no ambiente.

Para uma visualização completa e detalhada, consulte o Diagrama de Blocos Visual.

O ciclo de controle simplificado é:

Plain Text


[ Ambiente AGV ]
      ↑       ↓
  Estado   Recompensa
      ↑       ↓
 [ Agente RL ]
      ↓
 Ajuste de Ganhos PID


Estrutura do Projeto

Plain Text


rl-adaptive-pid/
│
├── env/
│   └── agv_env.py      # Simulação simplificada do ambiente AGV (dinâmica, trajetória, ruído)
│
├── agent/
│   └── dqn_agent.py    # Implementação do Agente RL (DQN ou Q-Learning simples)
│
├── train.py            # Script principal para treinamento do Agente RL
├── evaluate.py         # Script para avaliação do desempenho do sistema treinado
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo


Citação de Destaque


“Isso não é um brinquedo de RL. É um controlador clássico que aprende.”

