# -------------------------------------------------------------------------------------------   -----------#
#                                      Deck Class for Card Counting                                         #
# ----------------------------------------------------------------------------------------------------------#

import random
###### We use Hi-Lo Card Counting method(for drawing cards and giving running count) in the class CC_Deck #####
class cc_deck():
    """Deck of cards for CARD COUNTING"""

    def __init__(self, seed=0, number_of_decks=6, low_limit=6, high_limit=10, technique = 'hi_lo') -> None:
        self.random = random.Random(seed)
        self.number_of_decks = number_of_decks
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.technique = technique
        self.init_deck()

    def init_deck(self):
        """Initialize deck"""

        #1 = Ace, 2-10 = Number cards; Jack / Queen / King = 10
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * self.number_of_decks
        self.random.shuffle(self.deck)
        self.unseen_cards = len(self.deck)
        self.total_points = 0
        self.card_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

    def draw_card(self):
        """Draws a random card and do card counting, update total points ie running count"""

        card = self.deck.pop(0)
        self.unseen_cards -= 1

        if self.technique == 'hi_lo':
            if (card >= self.high_limit or card == 1):
                #cards 10, Jack, Queen, King, Ace
                self.total_points -= 1     #-1 for higher cards

            elif (card <= self.low_limit):
                #cards 2 - 6
                self.total_points += 1     #+1 for lower cards

        if self.technique == 'reverse-point':
            if card in [2, 3, 4, 5, 6]:
                self.total_points += 2     #+2 for 2 to 6
            elif card == 7:
                self.total_points += 1     #+1 for 7
            elif card in [8, 9]:
                self.total_points += 0     #0 for 8 and 9
            elif card in [10, 1]:
                self.total_points -= 2     #-2 for 10, Jack, Queen, King, Ace

        self.card_counter[card] += 1
        return card

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

