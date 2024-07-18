Blackjack Implementation in Reinforcement Learning

This project trains a Reinforcement Learning agent to play Blackjack game. Techniques: Monte Carlo and Q-Learning. Strategies: Basic Strategy (rules without card counting) and Complete Point Count System (Hi-Lo Card Counting). The goal is to develop an agent that can learn and improve its strategy to maximise winnings over time.

## Table of Contents:

1. Project Structure
2. Files Description
3. Reinforcement Learning Algorithms
4. Environment Details
5. How to Run the project?

## Project Structure

blackjack_rl/
├── output-images
├── algorithms.py
├── cc_deck.py
├── environment.py
├── logging-info.log
├── main.py
├── rule_utils.py
└── README.md

## Files Description

- output-images: Graphs used in the research paper, and bar charts used to create tables in the paper.
- main.py: The main script to initiate the game and train the agent. Parameters initialised here.
-  algorithms.py: Contains the implementation of reinforcement learning algorithms used to train the agent: Monte Carlo and Q-Learning.
-  cc_deck.py: Manages the card deck and deals cards for the game for Card Counting
-  environment.py: Defines two Blackjack environments, including the rules and interactions: Basic Strategy Environment (basic rules without card counting) and Card Counting Environment.
-  rule_utils.py: Utility functions related to the rules and logic of Blackjack.
-  logging-info.log: Log file that records the information and progress of the training process.

## Reinforcement Learning Algorithms
The project includes 2 reinforcement learning algorithms to train the agent. These algorithms help the agent learn the optimal strategy through interactions with the game environment.

## Environment Details
The environment simulates a standard Blackjack game with the following features:
-  Dealer and player interactions
-  Card dealing and shuffling
-  Win, lose, and draw conditions
-  Reward system based on the game outcome
- Card Counting System

## How to run the project?
The project can be run directly by running the main.py file. The parameters used to train the models are defined in this file itself. First part of the main function is the code to train the agent to generate reward graphs. Second part is to train the agent and get win counts in the form of bar graphs. The parameters can be changed and results can be reproduced accordingly. When the code is run, the images will be produced in the main folder by default. The graphs used in the research paper, and the bar charts used to create tables - are in the output-images folder. 
