# Collaborative-Graph-Coloring-Game

This project applies the $\alpha$-Rank evolutionary methodology to evaluate and rank joint strategies in a stochastic version of the Graph Coloring Game. Unlike traditional game-theoretic concepts like Nash equilibrium, which often fail to capture the complexity of agent dynamics, evolutionary approaches focus on how strategies persist over time. By transforming dynamic games into empirical forms and evaluating strategies based on their stability across repeated interactions, this approach identifies joint strategies that remain resistant to change in the long term.

<sub> This repository contains the code for the paper **_Ranking Joint Policies in Dynamic Games using Evolutionary Dynamics_**, accepted at AAMAS 2025.

<p align="center" style="margin-top: 30px; margin-bottom: 20px;">
  <img src="doc/images/graph.png" style="display: inline-block; width: 36%; margin-right: 5%">
  <img src="doc/images/cw-solution.png" style="display: inline-block; width: 45%">
</p>

## Prerequisites
This project uses **Poetry** for dependency management.

To set up the environment:
1. Install Poetry (if not already installed):
   ```bash
   pip install poetry
   ```
2. Clone the repository and navigate into it:
   ```bash
    git clone https://github.com/nataliakoliou/Collaborative-Graph-Coloring-Game.git
    cd cgcg
    ```
3. Install dependencies:
   ```bash
    poetry install
    ```

## How to Run the Code
The repository provides multiple entry points for running the code:

## Acknowledgments
I would like to express my sincere gratitude to the creators of $\alpha$-Rank for their foundational work in the paper ["α-Rank: Multi-Agent Evaluation by Evolutionary Dynamics"](https://www.nature.com/articles/s41598-019-45619-9). Their novel evolutionary methodology provided the theoretical framework for my own research on ranking joint policies in dynamic games. I would also like to thank DeepMind for including $\alpha$-Rank in their [OpenSpiel](https://github.com/google-deepmind/open_spiel) library. Their implementation was straightforward and easy to apply to my own custom game.

## Author
Natalia Koliou: find me on [LinkedIn](https://www.linkedin.com/in/natalia-koliou-b37b01197/).