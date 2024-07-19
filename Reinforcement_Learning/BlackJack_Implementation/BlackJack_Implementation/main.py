import numpy as np
from environment import BlackjackEnv, BlackjackEnv_CardCounting
from algorithms import Q_learning, MonteCarlo, train_agent_QL, train_agent_MC
import logging
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------#
#                                                  Main Function                                                    #
# ------------------------------------------------------------------------------------------------------------------#

if (__name__ == "__main__"):
    np.random.seed(0)
    logging.basicConfig(filename='log-file.log', level=logging.INFO, format='%(levelname)s - %(message)s', filemode='w')

    #################################### Generating rewards and plotting them #####################################

    #Parameters: Make changes to the parameters here for reproducing reward plots
    #IMP Note: The algorithms are to be run one at a time, by making their respective flag True.
    epochs = 100
    episodes = 5000
    num_decks = 1
    epsilon = [0.3, 0.8]
    learning_rate = [0.001, 0.01, 0.1]
    gamma = 0.5
    technique = 'hi_lo'
    surrender_enabled = True
    card_counting_flag = True
    QL = True
    MC = True
    results = {}
    if card_counting_flag:
        env = BlackjackEnv_CardCounting(number_of_decks=num_decks, technique=technique, surrender_enabled=surrender_enabled)       #technique can be hi_lo or reverse-point
    else:
        env = BlackjackEnv(surrender_enabled=surrender_enabled)

    #######Q Learning---------------------
    if QL:
        for ep in epsilon:
            results[ep] = {}
            act_counts = {}
            for lr in learning_rate:
                agent, dic_actions_count, avg_rewards, _,_,_ = train_agent_QL(env, epochs, episodes, ep, lr, gamma, card_counting_flag, surrender_enabled)
                results[ep][lr] = avg_rewards

        #Plot the results
        num_epsilons = len(epsilon)
        fig, axs = plt.subplots(1, num_epsilons, figsize=(15 * num_epsilons, 7))
        if num_epsilons == 1:
            axs = [axs]
        i = 0
        for ep in epsilon:
            colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rate)))
            k=0
            for lr in learning_rate:
                axs[i].plot(results[ep][lr], label=f'learning_rate: {lr}', color=colors[k])
                k+=1
            axs[i].set_xlabel("Number of epochs")
            axs[i].set_ylabel(f"Average Reward for {episodes} episodes")
            if card_counting_flag:
                axs[i].set_title(f"Q-Learning Reward for Epsilon: {ep} with Card Counting")
            else:
                axs[i].set_title(f"Q-Learning Reward for Epsilon: {ep} with Basic Strategy")
            axs[i].legend()
            i+=1
        plt.tight_layout()
        plt.savefig("Q-Learning.png")
        plt.show()


    ###### Monte Carlo---------------------
    if MC:
        results = {}
        for ep in epsilon:
            agent, dic_actions_count, avg_rewards, _,_,_ = train_agent_MC(env, epochs, episodes, ep, gamma, card_counting_flag, surrender_enabled)
            results[ep] = avg_rewards

        #Plot results
        num_epsilons = len(epsilon)
        fig, axs = plt.subplots(1, num_epsilons, figsize=(15 * num_epsilons, 7))
        if num_epsilons == 1:
            axs = [axs]
        colors = plt.cm.viridis(np.linspace(0, 1, num_epsilons))
        i=0
        for ep in epsilon:
            axs[i].plot(results[ep], label=f'epsilon: {ep}', color=colors[i])
            axs[i].set_xlabel("Number of epochs")
            axs[i].set_ylabel(f"Average Reward for {episodes} episodes")
            if card_counting_flag:
                axs[i].set_title(f"Monte Carlo Reward for Epsilon: {ep} with Card Counting")
            else:
                axs[i].set_title(f"Monte Carlo Reward for Epsilon: {ep} with Basic Strategy")
            # axs[i].grid()
            axs[i].legend()
            i+=1
        plt.tight_layout()
        plt.savefig("Monte-Carlo.png")
        plt.show()

    #################################### Plotting Bar CHarts ####################################

    #Code to plot bar charts: make changes to the parameters here for reproducing bar plots
    epochs = 100
    episodes = 5000
    num_decks = 1
    epsilon = 0.3
    learning_rate = 0.01
    gamma = 0.5
    technique = 'hi_lo'
    surrender_enabled = False
    card_counting_flag = True
    QL = True
    MC = True
    if card_counting_flag:
        env = BlackjackEnv_CardCounting(number_of_decks=num_decks, technique=technique, surrender_enabled=surrender_enabled)
    else:
        env = BlackjackEnv(surrender_enabled=surrender_enabled)
    agentMC, dic_actions_countMC, avg_rewardsMC, total_winMC, total_lossMC, total_tieMC = train_agent_MC(env, epochs, episodes, epsilon, gamma, card_counting_flag, surrender_enabled)
    agentQ, dic_actions_countQ, avg_rewardsQ, total_winQ, total_lossQ, total_tieQ = train_agent_QL(env, epochs, episodes, epsilon, learning_rate, gamma, card_counting_flag, surrender_enabled)

    # Create bar plots
    labels = ['Wins', 'Losses', 'Ties']
    counts_MC = [total_winMC, total_lossMC, total_tieMC]
    counts_Q = [total_winQ, total_lossQ, total_tieQ]
    method_name = ['Monte Carlo', 'Q-Learning']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bars_MC = ax.bar(x - width / 2, counts_MC, width, label=method_name[0])  #Monte Carlo
    bars_QL = ax.bar(x + width / 2, counts_Q, width, label=method_name[1])  #Q-Learning
    ax.set_ylabel('Counts')
    ax.set_title('Game Outcomes Comparison with Surrender enabled')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    all_bars = [bars_MC, bars_QL]
    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("Game_Outcomes_Comparison.png")
    plt.show()




