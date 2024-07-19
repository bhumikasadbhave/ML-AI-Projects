# ------------------------------------------------------------------------------------------------------------------#
#                                         Utility Functions                                                         #
# ------------------------------------------------------------------------------------------------------------------#

import random

class UtilityClass:
    card_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    @staticmethod
    def draw_random_card():
        """Draw a random card from deck"""
        return random.choice(UtilityClass.card_deck)

    @staticmethod
    def draw_initial_hand():
        """Draw an initial hand - two random cards"""
        return [UtilityClass.draw_random_card(), UtilityClass.draw_random_card()]

    @staticmethod
    def has_usable_ace(hand):
        """Return true if sum with ace as 11 is less than 21"""
        return 1 in hand and sum(hand) + 10 <= 21

    @staticmethod
    def check_bust(hand):
        """Check if the hand sum exceeds 21"""
        return UtilityClass.calculate_hand_sum(hand) > 21

    @staticmethod
    def calculate_hand_sum(hand):
        """Calculate the sum of the hand considering usable aces"""
        if UtilityClass.has_usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    @staticmethod
    def calculate_score(hand):
        """Calculate the score of the hand - zero is a bust"""
        return 0 if UtilityClass.check_bust(hand) else UtilityClass.calculate_hand_sum(hand)

    @staticmethod
    def compare_scores(player_score, dealer_score):
        """Compare player and dealer score: Return 1 if player wins else -1 """
        return float(player_score > dealer_score) - float(player_score < dealer_score)

    @staticmethod
    def check_natural_blackjack(hand):
        """Check if the hand is a natural blackjack"""
        return sorted(hand) == [1, 10]

    @staticmethod
    def double_down_possible(hand, actions_taken):
        """Check if double down possible: no action taken and only 2 cards """
        return len(hand) == 2 and actions_taken == 0

    @staticmethod
    def split_possible(hand, actions_taken):
        """Check if splitting possible: if no action taken and both cards are same"""
        return UtilityClass.double_down_possible(hand, actions_taken) and hand[0] == hand[1]

    @staticmethod
    def insurance_possible(dealer_up_card, actions_taken):
        """Check if insurance possible: dealer showing card=1"""
        return dealer_up_card == 1 and actions_taken == 0

