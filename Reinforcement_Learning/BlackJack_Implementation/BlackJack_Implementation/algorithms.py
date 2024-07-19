# ---------------------------------------------------------------------------------------------------------------#
#                                              Algorithms                                                       #
# --------------------------------------------------------------------------------------------------------------#

import random
import numpy as np
from collections import defaultdict
import logging

# ------------------------------------------------------------------------------------------------------------------#
#                                               Q-Learning                                                          #
# ------------------------------------------------------------------------------------------------------------------#

class Q_learning():
    """Q-learning Algorithm"""

    def __init__(self, env, epsilon=1.0, learning_rate=0.5, gamma=0.9):
        """Initialisation
            env: Blackjack Environment
            Q: Q Table
            epsilon: Probability for exploration
            learning_rate: Learning Rate
            gamma: Discount factor
        """
        self.env = env
        self.valid_actions = self.env.action_space
        self.Q = dict()
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

    def update_Q(self, observation, card_counting):
        """Set the initial Q-values to 0.0 if the observation not already in Q table (for all valida actions)
        """
        #Large negative reward for actions not possible
        large_number = -999999999.9
        filtered_actions = self.filter_valid_actions(observation, card_counting)
        if (observation not in self.Q):
            self.Q[observation] = dict((action, 0.0) if action in filtered_actions else (action, large_number) for action in self.valid_actions)


    def get_maxQ(self, observation, card_counting_flag):
        """Get the maximum Q-value of all actions based on the observation; Input: Observation, Output:max_q
        """
        self.update_Q(observation, card_counting_flag)
        max_q = max(self.Q[observation].values())
        return max_q

    def filter_valid_actions(self, observation, card_counting_flag):
        """Filter actions based on current state."""
        if card_counting_flag == False:
            hand_sum, dealer_card, usable_ace, double_down_possible, split_possible, insurance_possible, surrender_possible = observation
        else:
            hand_sum, dealer_card, usable_ace, double_down_possible, split_possible, insurance_possible, surrender_possible, total_points, unseen_cards = observation
        valid_actions = self.valid_actions.copy()

        #Remove double down if not possible
        if not double_down_possible:
            valid_actions.remove(2)

        #Remove split if not possible
        if not split_possible:
            valid_actions.remove(3)

        #Remove insurance if not possible
        if not insurance_possible:
            valid_actions.remove(4)

        return valid_actions

    def choose_action(self, observation, card_counting_flag):
        """Select the action to take based on the observation
           When the observation is seen for the first time, it initialises the Q values to 0.0
           Input: Observation, Output: action
        """
        self.update_Q(observation, card_counting_flag)
        filtered_actions = self.filter_valid_actions(observation, card_counting_flag)
        random_number = random.random()
        #exploit
        if (random_number > self.epsilon):
            maxQ = self.get_maxQ(observation, card_counting_flag)
            #If multiple actions have max Q, pick random from these
            action = random.choice([k for k in self.Q[observation].keys() if self.Q[observation][k] == maxQ and k in filtered_actions])
        #explore
        else:
            action = random.choice(filtered_actions)
        return action

    def learn(self, observation, action, reward, next_observation, card_counting_flag):
        """Input: Observation, action, reward, next_observation """
        self.Q[observation][action] += self.learning_rate * (reward + (self.gamma * self.get_maxQ(next_observation, card_counting_flag)) - self.Q[observation][action])


############################################ Q-Learning episodes and epochs ######################################################
####In Q-Learning, we update the Q value at every step/action in a given episode
def Qlearning_episodes(env, agent, observation, actions_count, episodes, card_counting_flag=False, surrender_enabled=False):
    rewards_episodes = np.zeros(episodes)
    i = 0
    win_count = 0
    loss_count = 0
    tie_count = 0
    while i < episodes:
        episode_total_rewards = 0              #Reset episode_total_rewards for each new episode
        while True:                            #Continue until the episode is done
            action = agent.choose_action(observation, card_counting_flag)
            next_observation, reward, is_done = env.step(action)
            actions_count[action] += 1
            episode_total_rewards += reward
            agent.learn(observation, action, reward, next_observation, card_counting_flag)    #Update Q value for every step in the episode
            observation = next_observation
            if is_done:
                rewards_episodes[i] = episode_total_rewards
                observation = env.reset()
                i += 1
                if reward == 0.5 and surrender_enabled:
                    loss_count += 1
                elif reward > 0:
                    win_count += 1
                elif reward < 0:
                    loss_count += 1
                else:
                    tie_count += 1
                break  #Break the while loop and start a new episode
    return agent, actions_count, rewards_episodes, win_count, loss_count, tie_count


def train_agent_QL(env, epochs, episodes, epsilon, learning_rate, gamma, card_counting_flag, surrender_enabled):
    """This function starts the training and evaluation with the Q-Learning method.
        Input:
        env: Blackjack Environment
        epochs: Number of epochs to be trained
        episodes: Number of players
    """
    episodes = episodes  #Reward calculated over every episode
    rewards = []
    avg_rewards = np.zeros(episodes)
    actions_count_dict = {}
    actions_dict = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT", 4: "INSURANCE", 5: "SURRENDER"}
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5:0}  #Number of actions for every option

    agent = Q_learning(env, epsilon, learning_rate, gamma)

    print("Training Agent with Q-Learning......")
    total_win = 0
    total_loss = 0
    total_tie = 0
    for epoch in range(epochs):
        observation = env.reset()
        agent, actions_count, rewards_episodes, win_count, loss_count, tie_count = Qlearning_episodes(env, agent, observation, actions_count, episodes, card_counting_flag, surrender_enabled)
        avg_rewards += (rewards_episodes - avg_rewards) / (epoch + 1)
        total_win += win_count
        total_loss += loss_count
        total_tie += tie_count
        rewards.append(np.sum(avg_rewards)/episodes)
        logging.info(f'Q-Learning - Rewards at end of epoch: {avg_rewards}')
    # Create an understandable dictionary
    for key, value in actions_dict.items():
        actions_count_dict.update({value: actions_count[key]})
    return agent, actions_count_dict, rewards, total_win, total_loss, total_tie


# ------------------------------------------------------------------------------------------------------------------#
#                                               Monte Carlo                                                         #
# ------------------------------------------------------------------------------------------------------------------#

class MonteCarlo:
    """Monte Carlo On-Policy Algorithm"""

    def __init__(self, env, epsilon=1.0, gamma=0.9):
        """
        Initialization:
        env: Environment
        Q: Q Table
        epsilon: Probability of selecting random action instead of the optimal action
        gamma: Discount factor
        """
        self.env = env
        self.valid_actions = self.env.action_space
        self.Q = dict()
        self.epsilon = epsilon
        self.gamma = gamma
        self.returns = defaultdict(list)

    def update_Q(self, observation, card_counting_flag):
        """This method sets the initial Q-values to 0.0 if the observation is not already included in the Q-table"""
        large_number = -999999999999.0
        filtered_actions = self.filter_valid_actions(observation, card_counting_flag)
        if observation not in self.Q:
            self.Q[observation] = dict((action, 0.0) if action in filtered_actions else (action, large_number) for action in self.valid_actions)

    def filter_valid_actions(self, observation, card_counting_flag):
        """Filter actions based on current state."""
        if not card_counting_flag:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance, surrender_enabled = observation
        else:
            hand_sum, dealer_card, usable_ace, can_double_down, can_split, can_insurance, surrender_enabled, total_points, unseen_cards = observation
        valid_actions = self.valid_actions.copy()

        # Remove double down if not possible
        if not can_double_down:
            valid_actions.remove(2)

        # Remove split if not possible
        if not can_split:
            valid_actions.remove(3)

        # Remove insurance if not possible
        if not can_insurance:
            valid_actions.remove(4)
        return valid_actions

    def get_maxQ(self, observation, card_counting_flag):
        """This method is called when the agent is asked to determine the maximum Q-value
           of all actions based on the observation the environment is in.
            Input: Observation, Output: max_q
        """
        self.update_Q(observation, card_counting_flag)
        max_q = max(self.Q[observation].values())
        return max_q

    def choose_action(self, observation, card_counting_flag=False):
        """This method selects the action to take based on the observation."""
        self.update_Q(observation, card_counting_flag)
        filtered_actions = self.filter_valid_actions(observation, card_counting_flag)
        random_number = random.random()
        if random_number > self.epsilon:
            maxQ = self.get_maxQ(observation, card_counting_flag)
            action = random.choice([k for k in range(len(self.Q[observation])) if self.Q[observation][k] == maxQ and k in filtered_actions])
        else:
            action = random.choice(filtered_actions)
        return action

    def learn(self, episode):
        """
        Update the Q-values using the Monte Carlo update rule.
        """
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            G = sum(rewards[i:] * discounts[:-(1+i)])
            self.returns[(state, actions[i])].append(G)
            self.Q[state][actions[i]] = np.mean(self.returns[(state, actions[i])])

############################################ Monte Carlo episodes and epochs ######################################################
####In Monte Carlo, we update the Q value not at every step/action, but at the end of the given episode
def MC_episodes(env, agent, observation, actions_count, episodes, card_counting_flag, surrender_enabled):
    rewards_episodes = np.zeros(episodes)
    i = 0
    win_count = 0
    loss_count = 0
    tie_count = 0
    while i < episodes:
        episode = []
        episode_total_reward = 0  #Reset episode_total_reward for each new episode
        while True:  # Continue until the episode is done
            action = agent.choose_action(observation, card_counting_flag)
            next_observation, reward, is_done = env.step(action)
            actions_count[action] += 1
            episode_total_reward += reward
            episode.append((observation, action, reward))
            observation = next_observation
            if is_done:
                agent.learn(episode)   #Update Q value after the entire episode
                rewards_episodes[i] = episode_total_reward
                observation = agent.env.reset()
                i += 1
                if reward == 0.5 and surrender_enabled:
                    loss_count += 1
                elif reward > 0:
                    win_count += 1
                elif reward < 0:
                    loss_count += 1
                else:
                    tie_count += 1
                break  #Break the while loop and start a new episode
    return agent, actions_count, rewards_episodes, win_count, loss_count, tie_count

def train_agent_MC(env, epochs, episodes, epsilon, gamma, card_counting_flag, surrender_enabled):
    """
    This function starts the training and evaluation of Monte Carlo on-policy method
    """
    re = []
    total_win = 0
    total_loss = 0
    total_tie = 0
    avg_rewards = np.zeros(episodes)
    actions_count_dict = {}
    actions_dict = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT", 4: "INSURANCE", 5: "SURRENDER"}
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Number of actions performed in each category
    agent = MonteCarlo(env, epsilon, gamma)
    print("Training Agent with Monte Carlo ...")
    for epoch in range(epochs):
        observation = env.reset()
        agent, actions_count, rewards_episodes, win_count, loss_count, tie_count = MC_episodes(env, agent, observation, actions_count, episodes, card_counting_flag, surrender_enabled)
        avg_rewards += (rewards_episodes - avg_rewards) / (epoch + 1)
        re.append(np.sum(avg_rewards) / episodes)
        total_win += win_count
        total_loss += loss_count
        total_tie += tie_count
        logging.info(f'Monte Carlo - Total average rewards: {np.sum(avg_rewards) / episodes}')
    for key, value in actions_dict.items():
        actions_count_dict.update({value: actions_count[key]})
    return agent, actions_count_dict, re, total_win, total_loss, total_tie