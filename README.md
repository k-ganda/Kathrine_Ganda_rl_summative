# Ambulance Simulation with DQN and PPO

This project involves training reinforcement learning (RL) agents to navigate an ambulance simulation environment, where the goal is to pick up and drop off patients as efficiently as possible.

We compare two RL algorithms—Deep Q-Network (DQN) and Proximal Policy Optimization (PPO)—to assess their performance in terms of task completion, exploration strategies, and training stability.

The agent explores a variety of action spaces and learns to optimize its behavior through the following methods:

**1.DQN:** A value-based method that leverages experience replay and target networks.

**2.PPO:** A policy-based method known for its more stable training and higher sample efficiency.

## Key Features

**Custom Environment:** The simulation involves a dynamic environment where the agent must navigate to pick up and drop off patients while avoiding obstacles.

**Action Space:** The agent can perform a range of actions including movement in various directions and accelerating to reach the target more quickly.

**Reward System:** The agent is rewarded based on successful patient pickups and drop-offs, with penalties for collisions and inefficient movement.

## Hyperparameter Tuning

Various hyperparameters were tuned to achieve better agent performance; learning rate, batch size, buffer size etc.

## Set Up and Usage

1. **Clone the repository:**

Start by cloning the repository to your local machine.
On your terminal, run:

```
https://github.com/k-ganda/Kathrine_Ganda_rl_summative.git
```

2. **Set up a virtual environment:**

```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

Install the necessary dependencies using pip:

`pip install -r requirements.txt`

4. **Run Simulation**

On your terminal run:

`python main.py`

This should display the PyOpenGL visualization of the environment, load the model and see the agent's actions.

## Optional: Train Models

To train the DQN or PPO model, run:

`python -m training.pg_training`
`python -m training.dqn_training`

There are however already trained models that you can load and use under the `/models` folder.

## Link To Video Demo

https://www.youtube.com/watch?v=whMm5hPOkbk
