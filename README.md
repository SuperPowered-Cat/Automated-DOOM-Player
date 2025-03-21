# Doom Agent Project

Welcome to the Doom Agent Project! This repository contains code, notebooks, and models for training and evaluating a DOOM agent using the Proximal Policy Optimization (PPO) algorithm with the [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) framework. The project explores training agents on multiple levels:
- **Basic Level**
- **Defend the Center**
- **Deadly Corridor** (with cumulative learning and reward shaping)

The goal is to develop agents that learn to play DOOM through reinforcement learning, and to provide a clean, reproducible project structure for training, evaluation, and further development.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [Notebooks](#notebooks)
  - [Evaluation Scripts](#evaluation-scripts)
- [Training & Logs](#training--logs)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Repository Structure

The repository is organized as follows:

```
doom-agent-project/
├── README.md             # This file
├── LICENSE               # Project license
├── requirements.txt      # Python dependencies list
├── notebooks/            # Jupyter notebooks documenting training experiments
│   ├── 1_basic.ipynb
│   ├── 2_defend_the_center.ipynb
│   └── 3_deadly_corridor.ipynb   
├── src/                  # Python scripts for running trained agents
│   ├── basic_agent.py    # Evaluates the Basic level agent
│   ├── defend_agent.py   # Evaluates the Defend the Center agent
│   └── third_agent.py    # Evaluates the Deadly Corridor agent
├── models/               # Saved PPO models organized by experiment
│   ├── PPO1/             # Models for Basic level
│   ├── PPO2/             # Models for Defend the Center
│   └── PPO3/             # Models for Third level
├── logs/                 # TensorBoard and training logs organized per experiment
│   ├── logs_basic/
│   ├── logs_defend/
│   └── logs_third/
```
---

## Installation and Setup

### Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.7+**
- **pip** (Python package manager)
- **Git**

### Step-by-Step Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SuperPowered-Cat/Automated-DOOM-Player.git
   cd doom-agent-project
   ```
2: Create a Virtual Environment (Recommended)
   - command: "python -m venv venv"
   - activation:
       Unix/macOS: "source venv/bin/activate"
       Windows: "venv\\Scripts\\activate"

3: Install Dependencies
   - command: "pip install -r requirements.txt"
   - note: "This installs packages like vizdoom, gymnasium, stable-baselines3, torch, opencv-python, matplotlib, tensorboard, etc."

4: Download VizDoom Scenarios
   - note: "Ensure the VizDoom configuration files are accessible. The notebooks include steps to clone the ViZDoom repository:"
   - command: "cd github && git clone https://github.com/Farama-Foundation/ViZDoom"

## Usage
### Notebooks
`1_basic.ipynb`
Contains code for training and experimenting with the Basic level environment.

`2_defend_the_center.ipynb`
Contains code for training and experimenting with the Defend the Center level.

`3_deadly_corridor.ipynb`
Contains code for training and experimenting with the Deadly Corridor level using cumulative learning and reward shaping.

You can open these notebooks using Jupyter Notebook or JupyterLab.

Evaluation Scripts
In the src/ folder, you will find Python scripts that load a trained model and run the agent in evaluation mode
   
