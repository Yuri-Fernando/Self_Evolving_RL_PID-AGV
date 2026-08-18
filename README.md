# Self-Evolving RL-PID-AGV

### Reinforcement Learning · Adaptive Control · PID · AGV · Robotics · Industrial AI

## Status

🟢 **Concluído — Projeto de portfólio / Pesquisa Aplicada em Controle e IA**

Projeto desenvolvido para investigar a aplicação de **Reinforcement Learning como camada adaptativa sobre controladores PID clássicos**, com foco em rastreamento de trajetória de **AGVs (Automated Guided Vehicles)**.

A solução utiliza um agente de Reinforcement Learning para ajustar dinamicamente os ganhos do controlador PID durante a operação, buscando melhorar a precisão e a estabilidade do rastreamento sob diferentes condições de trajetória.

O projeto prioriza **baixa complexidade computacional, interpretabilidade, reprodutibilidade e compatibilidade com estratégias de controle industrial**.

---

## Sobre o Projeto

Controladores PID tradicionais normalmente utilizam ganhos definidos previamente. Em cenários nos quais as condições de operação variam, uma sintonia fixa pode não apresentar o mesmo desempenho em todas as situações.

A proposta deste projeto é adicionar uma camada adaptativa baseada em Reinforcement Learning:

```text
Trajetória do AGV
       ↓
Erro de Rastreamento
       ↓
Controlador PID
       ↓
Ação do AGV
       ↓
Estado do Sistema
       ↓
Agente RL
       ↓
Ajuste dos Ganhos PID
       ↓
Novo Ciclo de Controle
```

O agente atua como uma camada de adaptação sobre o controlador clássico, mantendo a estrutura PID como elemento central da estratégia de controle.

---

## Objetivo

Desenvolver uma abordagem de **RL adaptativo para sintonia dinâmica de PID** aplicada ao rastreamento de trajetória de AGVs.

O projeto busca demonstrar:

- Ajuste adaptativo dos ganhos PID;
- Integração entre Reinforcement Learning e controle clássico;
- Adaptação a diferentes condições de trajetória;
- Melhoria da precisão do rastreamento;
- Melhoria da estabilidade do sistema;
- Baixa complexidade computacional;
- Arquitetura reproduzível;
- Aplicabilidade conceitual a ambientes industriais.

---

# Arquitetura

```text
                    ┌─────────────────────┐
                    │ Planned Trajectory  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   AGV Environment   │
                    └──────────┬──────────┘
                               │
                               ▼
                       Tracking Error
                               │
                               ▼
                    ┌─────────────────────┐
                    │    PID Controller   │
                    │     Kp / Ki / Kd     │
                    └──────────┬──────────┘
                               │
                               ▼
                         AGV Dynamics
                               │
                               ▼
                          New State
                               │
                               ▼
                    ┌─────────────────────┐
                    │      RL Agent       │
                    │   Adaptive Layer    │
                    └──────────┬──────────┘
                               │
                               ▼
                      PID Gain Adjustment
                               │
                               └───────────┐
                                           │
                                           ▼
                                   Next Control Cycle
```

---

# Princípio de Funcionamento

O agente observa o estado do sistema de controle e utiliza essa informação para ajustar os parâmetros do controlador PID.

```text
Estado
  ↓
RL Agent
  ↓
Ação
  ↓
ΔKp / ΔKi / ΔKd
  ↓
Novo PID
  ↓
AGV
  ↓
Novo Estado
```

Dessa forma, o PID permanece responsável pelo controle, enquanto o agente de Reinforcement Learning atua como uma camada adaptativa.

---

# Reinforcement Learning

A arquitetura foi desenvolvida para utilizar RL em uma função de ajuste incremental dos ganhos.

O agente pode observar variáveis relacionadas a:

- Erro de trajetória;
- Estado do AGV;
- Evolução do erro;
- Condições do controlador.

A ação está relacionada ao ajuste dos ganhos:

```text
Kp
Ki
Kd
```

O objetivo é encontrar configurações que proporcionem um melhor compromisso entre:

- Precisão;
- Estabilidade;
- Resposta do controlador.

---

# PID Adaptativo

A estrutura tradicional:

```text
PID
 ↓
Kp + Ki + Kd
```

é evoluída para:

```text
RL Agent
    ↓
PID Gain Adaptation
    ↓
Kp / Ki / Kd
    ↓
Trajectory Tracking
```

A principal característica da abordagem é permitir que os ganhos sejam ajustados dinamicamente em vez de depender exclusivamente de uma sintonia fixa.

---

# Foco Técnico

## Baixa Complexidade

A solução busca utilizar uma arquitetura de RL relativamente leve, evitando complexidade desnecessária para o problema.

## Interpretabilidade

O controlador PID permanece explícito, permitindo relacionar diretamente as ações do agente aos ganhos `Kp`, `Ki` e `Kd`.

## Reprodutibilidade

A arquitetura foi construída com foco em experimentação controlada e repetível.

## Compatibilidade Industrial

A abordagem preserva um controlador clássico amplamente conhecido na indústria e adiciona uma camada adaptativa baseada em IA.

---

# Pipeline de Execução

```text
1. Inicialização do ambiente
        ↓
2. Definição da trajetória
        ↓
3. Inicialização do controlador PID
        ↓
4. Inicialização do agente RL
        ↓
5. Observação do estado
        ↓
6. Seleção da ação
        ↓
7. Ajuste dos ganhos PID
        ↓
8. Aplicação do controle
        ↓
9. Atualização do estado do AGV
        ↓
10. Cálculo do erro
        ↓
11. Atualização do agente
        ↓
12. Próximo ciclo
```

---

# Aplicação

O projeto é direcionado ao problema de **trajectory tracking de AGVs**, especialmente em cenários nos quais diferentes condições de trajetória podem exigir diferentes comportamentos do controlador.

A arquitetura pode ser explorada em situações como:

- Trajetórias retas;
- Curvas;
- Mudanças de trajetória;
- Diferentes condições de controle.

---

# O que este projeto demonstra

- Reinforcement Learning;
- Adaptive Control;
- PID Control;
- AGV Control;
- Trajectory Tracking;
- Robotics;
- Industrial AI;
- Controle adaptativo;
- Integração entre IA e controle clássico;
- Ajuste dinâmico de parâmetros;
- Arquitetura modular;
- Simulação de sistemas de controle;
- Desenvolvimento de soluções orientadas à reprodutibilidade.

---

# Tecnologias e Conceitos

| Categoria | Tecnologia / Conceito |
|---|---|
| IA | Reinforcement Learning |
| Controle | PID |
| Aplicação | AGV / Robotics |
| Problema | Trajectory Tracking |
| Abordagem | Adaptive PID Tuning |
| Ambiente | Simulação |
| Arquitetura | RL + Classical Control |
| Desenvolvimento | Python / Simulation Stack |

---

# Limitações

- A abordagem atual é baseada em ambiente de simulação;
- Os resultados dependem da dinâmica e das condições utilizadas no ambiente;
- A aplicação em um AGV físico exige validação adicional;
- Transferência do agente da simulação para hardware real pode exigir ajustes;
- Diferentes trajetórias e dinâmicas podem demandar novas estratégias de treinamento;
- A implementação atual prioriza simplicidade e interpretabilidade em vez de arquiteturas RL mais complexas.

---

# Melhorias Futuras

- Validação em AGV físico;
- Integração com sensores reais;
- Sim-to-Real;
- Transfer Learning;
- Novos espaços de ação;
- Reward shaping avançado;
- Comparação com diferentes algoritmos de RL;
- Testes com trajetórias mais complexas;
- Otimização dos hiperparâmetros;
- Avaliação sistemática de robustez;
- Integração com ROS;
- Implementação embarcada;
- Comparação com métodos clássicos de sintonia PID;
- Aplicação em ambientes industriais mais complexos.

---

# Status Final

🟢 **Concluído**

A implementação atual estabelece a arquitetura principal de **Reinforcement Learning como camada adaptativa sobre controle PID**, contemplando:

- ✅ Agente de Reinforcement Learning;
- ✅ Ajuste adaptativo dos ganhos PID;
- ✅ Rastreamento de trajetória de AGV;
- ✅ Integração RL + PID;
- ✅ Arquitetura orientada à baixa complexidade;
- ✅ Estrutura interpretável;
- ✅ Ambiente de simulação;
- ✅ Foco em reprodutibilidade;
- ✅ Base para evolução para aplicações físicas.

O projeto permanece como uma demonstração de **IA aplicada ao controle industrial e robótica**, explorando a combinação entre métodos clássicos de controle e técnicas modernas de aprendizado por reforço.

---

# Autor

**Yuri Fernando Dubbern**

AI/ML Engineer · Reinforcement Learning · Robotics · Industrial AI · Intelligent Automation

[LinkedIn](https://www.linkedin.com/in/yuridubbern) · [GitHub](https://github.com/Yuri-Fernando) · [Lattes](http://lattes.cnpq.br/7151392692642166) · [Linktree](https://linktr.ee/yuri.f.dubbern)
