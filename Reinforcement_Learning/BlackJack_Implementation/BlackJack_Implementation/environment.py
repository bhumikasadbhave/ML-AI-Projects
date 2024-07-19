# ---------------------------------------------------------------------------------------------------------------#
#                                        Blackjack Environments                                                  #
# ---------------------------------------------------------------------------------------------------------------#

import random
from typing import Optional
from rule_utils import UtilityClass
from cc_deck import cc_deck

######## BlackJack Environment with Basic Strategy i.e. without card counting ########
class BlackjackEnv:

    def __init__(self, surrender_enabled):
        self.action_space = [0,1,2,3,4]
        self.actionstaken = 0
        self.insurance_bet = 0
        self.surrender_enabled = surrender_enabled
        if self.surrender_enabled:
            self.action_space = [0,1,2,3,4,5]

    def step(self, action):
        """Learning step"""
        assert action in self.action_space
        if UtilityClass.calculate_hand_sum(self.player) == 21:
            action=0

        if action == 0:  #stand
            done = True
            while UtilityClass.calculate_hand_sum(self.dealer) < 17:
                self.dealer.append(UtilityClass.draw_random_card())
            reward = UtilityClass.compare_scores(UtilityClass.calculate_score(self.player), UtilityClass.calculate_score(self.dealer))
            if UtilityClass.check_natural_blackjack(self.player) and not UtilityClass.check_natural_blackjack(self.dealer):
                reward = 1.0
            if self.insurance_bet > 0:
                if UtilityClass.check_natural_blackjack(self.dealer):
                    reward += 2 * self.insurance_bet     #break even if dealer gets Blackjack
                else:
                    reward -= self.insurance_bet
            self.actionstaken += 1

        elif action == 1:  #hit
            self.player.append(UtilityClass.draw_random_card())
            if UtilityClass.check_bust(self.player):   #if player busts
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
            self.actionstaken += 1

        elif action == 2:  #double down: double the initial bet; stand after taking 1 card
            self.player.append(UtilityClass.draw_random_card())
            if UtilityClass.check_bust(self.player):
                done = True
                reward = -2.0
            else:
                done = True
                while UtilityClass.calculate_hand_sum(self.dealer) < 17:
                    self.dealer.append(UtilityClass.draw_random_card())
                reward = 2.0 * UtilityClass.compare_scores(UtilityClass.calculate_score(self.player), UtilityClass.calculate_score(self.dealer))
            self.actionstaken += 1


        elif action == 3:  #split
            if UtilityClass.split_possible(self.player, self.actionstaken):
                card1 = self.player[0]
                card2 = self.player[1]

                #Splitting the hand
                hand1 = [card1, UtilityClass.draw_random_card()]
                hand2 = [card2, UtilityClass.draw_random_card()]

                if card1 == 1:  #Ace splitting rule: if we split 2 aces, take one card for each hand; and stand
                    hand1 = [card1, UtilityClass.draw_random_card()]
                    hand2 = [card2, UtilityClass.draw_random_card()]
                    done = True
                    reward1 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand1),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward2 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand2),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2    #average reward

                else:
                    #Play hand 1
                    while not UtilityClass.check_bust(hand1) and not UtilityClass.check_natural_blackjack(hand1):
                        hand1.append(UtilityClass.draw_random_card())
                    #Play hand 2
                    while not UtilityClass.check_bust(hand2) and not UtilityClass.check_natural_blackjack(hand2):
                        hand2.append(UtilityClass.draw_random_card())
                    done = True
                    reward1 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand1),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward2 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand2),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2     #average reward

            else:
                done = False
                reward = 0.0
            self.actionstaken += 1


        elif action == 4:  #insurance
            if self.dealer[0] == 1:        #If dealer is showing an Ace
                self.insurance_bet = 0.5   #Take half of initial bet as insurance
                reward = 0.0               #No immediate reward
                done = False
            else:
                reward = 0.0
                done = False

        elif action == 5 and self.surrender_enabled:  #surrender
            done = True
            reward = 0.5

        return self._get_obs(), reward, done

    def _get_obs(self):
        """Set the current observations for the agent to see and choose next steps...to learn..."""
        return (
            UtilityClass.calculate_hand_sum(self.player), self.dealer[0], UtilityClass.has_usable_ace(self.player),
            UtilityClass.double_down_possible(self.player, self.actionstaken),
            UtilityClass.split_possible(self.player, self.actionstaken),
            UtilityClass.insurance_possible(self.dealer[0], self.actionstaken),
            self.surrender_enabled
        )

    def reset(self, seed: Optional[int] = None):
        """Resetting environment"""
        if seed is not None:
            random.seed(seed)
        self.dealer = UtilityClass.draw_initial_hand()    # eg. [10,2]
        self.player = UtilityClass.draw_initial_hand()    # eg. [3,8]
        self.actionstaken = 0
        self.insurance_bet = 0
        return self._get_obs()


######## BlackJack Environment with Card Counting ########
#Here, we calculate the running count of cards when we draw a card, and give it to agent as an observation
class BlackjackEnv_CardCounting:

    def __init__(self, number_of_decks, technique, surrender_enabled):
        self.action_space = [0,1,2,3,4]
        self.actionstaken = 0
        self.insurance_bet = 0
        self.surrender_enabled = surrender_enabled
        self.deck = cc_deck(seed=0, number_of_decks=number_of_decks, technique=technique)  #can have multiple decks with card counting
        if self.surrender_enabled:
            self.action_space = [0,1,2,3,4,5]

    def step(self, action):
        """Learning step"""

        assert action in self.action_space
        if action == 0:  #stand
            done = True
            while UtilityClass.calculate_hand_sum(self.dealer) < 17:
                self.dealer.append(self.deck.draw_card())
            reward = UtilityClass.compare_scores(UtilityClass.calculate_score(self.player), UtilityClass.calculate_score(self.dealer))
            if UtilityClass.check_natural_blackjack(self.player) and not UtilityClass.check_natural_blackjack(self.dealer):
                reward = 1.0
            elif UtilityClass.check_natural_blackjack(self.player) and reward == 1.0:
                reward = 1.5
            if self.insurance_bet > 0:
                if UtilityClass.check_natural_blackjack(self.dealer):
                    reward += 2 * self.insurance_bet
                else:
                    reward -= self.insurance_bet
            self.actionstaken += 1

        elif action == 1:  #hit
            self.player.append(self.deck.draw_card())
            if UtilityClass.check_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
            self.actionstaken += 1

        elif action == 2:  #double down: take one card and stand
            self.player.append(self.deck.draw_card())
            if UtilityClass.check_bust(self.player):
                done = True
                reward = -2.0
            else:
                done = True
                while UtilityClass.calculate_hand_sum(self.dealer) < 17:
                    self.dealer.append(self.deck.draw_card())
                reward = 2.0 * UtilityClass.compare_scores(UtilityClass.calculate_score(self.player), UtilityClass.calculate_score(self.dealer))
            self.actionstaken += 1


        elif action == 3:  #split
            if UtilityClass.split_possible(self.player, self.actionstaken):
                card1 = self.player[0]
                card2 = self.player[1]

                #Splitting the hand
                hand1 = [card1, self.deck.draw_card()]
                hand2 = [card2, self.deck.draw_card()]

                if card1 == 1:  # ce splitting rule
                    hand1 = [card1, self.deck.draw_card()]
                    hand2 = [card2, self.deck.draw_card()]
                    done = True
                    reward1 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand1),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward2 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand2),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2
                else:
                    while not UtilityClass.check_bust(hand1) and not UtilityClass.check_natural_blackjack(hand1):
                        hand1.append(self.deck.draw_card())
                    while not UtilityClass.check_bust(hand2) and not UtilityClass.check_natural_blackjack(hand2):
                        hand2.append(self.deck.draw_card())
                    done = True
                    reward1 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand1),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward2 = UtilityClass.compare_scores(UtilityClass.calculate_score(hand2),
                                                          UtilityClass.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2

            else:
                done = False
                reward = 0.0
            self.actionstaken += 1


        elif action == 4:  #insurance
            if self.dealer[0] == 1:         #If dealer's showing card is an Ace
                self.insurance_bet = 0.5    #Half of the initial bet - insurance
                reward = 0.0                #No immediate reward
                done = False
            else:
                reward = 0.0
                done = False

        elif action == 5 and self.surrender_enabled:  #surrender
            done = True
            reward = 0.5
        return self._get_obs(), reward, done

    def _get_obs(self):
        """Set the current observations for the agent"""
        return (
            UtilityClass.calculate_hand_sum(self.player), self.dealer[0], UtilityClass.has_usable_ace(self.player),
            UtilityClass.double_down_possible(self.player, self.actionstaken),
            UtilityClass.split_possible(self.player, self.actionstaken),
            UtilityClass.insurance_possible(self.dealer[0], self.actionstaken),
            self.surrender_enabled,   #surrender option
            self.deck.total_points,   #running count of total points
            self.deck.unseen_cards    #total unseen cards remaining in the deck
        )

    def reset(self, seed: Optional[int] = None):
        """Resetting the environment"""
        if seed is not None:
            random.seed(seed)

        self.deck.init_deck()
        self.dealer = self.deck.draw_hand()
        self.player = self.deck.draw_hand()
        self.actionstaken = 0
        self.insurance_bet = 0

        return self._get_obs()





