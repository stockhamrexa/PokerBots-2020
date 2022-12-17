
"""
A utility to narrow down the probability of a distribution based on a list of direct comparisons between numbers.
"""

import eval7
import hand_type
import numpy as np
import toposort
import scipy.optimize

# Maps each rank to a probability distribution representing where it will be placed in any given permutation
DIST = {
    "2": [0.2561515, 0.191812, 0.1440469, 0.1081664, 0.0812476, 0.0605954, 0.0455561, 0.0341089, 0.0256647, 0.0192342, 0.0144577, 0.0108468, 0.0081118],
    "3": [0.1920533, 0.1731798, 0.1461833, 0.1186864, 0.0940539, 0.0733629, 0.0566437, 0.0434193, 0.0330757, 0.0251478, 0.0190425, 0.0143352, 0.0108162],
    "4": [0.1440349, 0.1462282, 0.1385146, 0.1234872, 0.1047147, 0.0859886, 0.068977, 0.0545089, 0.0424382, 0.0327511, 0.0249122, 0.018999, 0.0144454],
    "5": [0.1079863, 0.1186073, 0.1232588, 0.1205129, 0.1111626, 0.0973258, 0.0818817, 0.0667774, 0.0534508, 0.0419488, 0.0326523, 0.025152, 0.0192833],
    "6": [0.0809578, 0.0941589, 0.1049623, 0.1109027, 0.1108703, 0.1045299, 0.0933219, 0.0798681, 0.0661147, 0.0533661, 0.0422841, 0.03311, 0.0255532],
    "7": [0.0606975, 0.0734786, 0.0861882, 0.0971546, 0.1044553, 0.1061602, 0.1014846, 0.0918223, 0.0798772, 0.0666752, 0.0543378, 0.0434906, 0.0341779],
    "8": [0.0456, 0.0567047, 0.0691337, 0.0816703, 0.0930872, 0.1013681, 0.1044671, 0.1015134, 0.0932094, 0.0817572, 0.0692098, 0.0566397, 0.0456394],
    "9": [0.0341683, 0.0434322, 0.054406, 0.0668995, 0.0797819, 0.0919628, 0.1012582, 0.105938, 0.1042885, 0.0972197, 0.0862177, 0.0735656, 0.0608616],
    "T": [0.0256343, 0.0331604, 0.0423474, 0.0534558, 0.0660798, 0.0797748, 0.0931303, 0.1044503, 0.1109109, 0.1110316, 0.1049076, 0.0941519, 0.0809649],
    "J": [0.0192511, 0.0251067, 0.032546, 0.0420215, 0.053426, 0.0667698, 0.0817706, 0.097284, 0.1111433, 0.1206852, 0.1234588, 0.1185023, 0.1080347],
    "Q": [0.014478, 0.0190749, 0.0249855, 0.0327047, 0.0422912, 0.0544837, 0.0692477, 0.0859324, 0.1048204, 0.1233446, 0.1385445, 0.1459617, 0.1441307],
    "K": [0.0108484, 0.0142928, 0.0190021, 0.0251277, 0.0331424, 0.043452, 0.056686, 0.0735924, 0.0940372, 0.1186199, 0.1460589, 0.1731345, 0.1920057],
    "A": [0.0081386, 0.0107635, 0.0144252, 0.0192103, 0.0256871, 0.034226, 0.0455751, 0.0607846, 0.080969, 0.1082186, 0.1439161, 0.1921107, 0.2559752]
}

# Maps the integer representation of each rank to a string
INDICES = {
    0: "2",
    1: "3",
    2: "4",
    3: "5",
    4: "6",
    5: "7",
    6: "8",
    7: "9",
    8: "T",
    9: "J",
    10: "Q",
    11: "K",
    12: "A"
}

# Maps the string representation of each rank to an integer
RANKS = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12
}

class Permutation:
    """
    An object that contains the current permutation used by the game. Continually updates itself as new rules are
    learned, resampling from a more and more accurate probability distribution.
    """

    def __init__(self):
        """
        The permutation begins as a standard deck of cards. Takes in a mapping of each possible card to its
        corresponding eval7.Card object.
        """
        self.ground_truth = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # The permutation for a standard deck of cards
        self.permutation = self.ground_truth  # The current permutation

        # A dictionary that stores the dependencies for each rank for which we have learned some relative ordering
        self.dependencies = {}

        # The inverse of all dependencies that are defined in self.dependencies
        self.reverse_dependencies = {}

    def add_rule(self, upper, lower):
        """
        Adds a new rule, updating the adjacency matrix. Both lower and upper should be string representations of ranks.
        """
        if upper not in self.dependencies:
            self.dependencies[upper] = set(lower)

        else:
            self.dependencies[upper].add(lower)

        if lower not in self.reverse_dependencies:
            self.reverse_dependencies[lower] = set(upper)

        else:
            self.reverse_dependencies[lower].add(upper)

    def convert_cards(self, cards, perm, to_eval7=False):
        """
        Given a list of cards represented as strings, converts them to their true values given a permutation. If
        to_eval7 is true, they will be converted to eval7.Card objects.
        """
        if cards == []:  # If there are no cards to convert
            return cards

        converted = []

        for card in cards:
            rank = card[0]
            suit = card[1]
            new_rank = perm[RANKS[rank]]

            if to_eval7:
                converted.append(eval7.Card(new_rank + suit))

            else:
                converted.append(new_rank + suit)

        return converted

    def find_flush(self, hand, board):
        """
        If you have a flush, finds what suit the flush consists of. Will only be applied when there are four cards of
        the same suit on the board.
        """
        hand_suits = [card[1] for card in hand]  # The suits of the cards in hand
        board_suits = [card[1] for card in board]  # The suits of the cards in hand

        if hand_suits[0] == hand_suits[1]:  # We only accept flushes with four board cards of the same suit
            return None

        for suit in hand_suits:
            if board_suits.count(suit) == 4:  # If the board has four cards of a suit and you have one
                return suit

        return None

    def find_full_house(self, hand, board):
        """
        If you have a full house, then this function attempts to find which ranks you have three of and which ranks you have
        two of. Returns the rank you have three of first, then the rank you have two of.
        """
        hand_ranks = [card[0] for card in hand]  # The ranks of the cards in hand
        board_ranks = [card[0] for card in board]  # The ranks of the cards in board

        three = None  # The rank we have three of
        two = None  # The rank we have two of

        if hand_ranks[0] == hand_ranks[1]:  # Pocket pairs
            if hand_ranks[0] in board_ranks:
                three = hand_ranks[0]

                for rank in board_ranks:  # Find the rank there is two of
                    if board_ranks.count(rank) == 2:
                        two = rank

                        return (three, two)
            else:
                two = hand_ranks[0]

                for rank in board_ranks:  # Find the rank there is three of
                    if board_ranks.count(rank) == 3:
                        three = rank

                        return (three, two)

        else:  # You did not have pocket pairs
            for rank in hand_ranks:
                if board_ranks.count(rank) == 2:
                    three = rank

                elif board_ranks.count(rank) == 1:
                    two = rank

            if three == None and two != None:  # Three of one rank is on the board and one of the other ranks is in your hand
                for rank in board_ranks:
                    if board_ranks.count(rank) == 3:
                        three = rank

                        return (three, two)

        if three != None and two != None:
            return (three, two)

        else:
            return None

    def find_pair(self, hand, board, type):
        """
        If you have a pair, three of a kind, or four of a kind, as designated by type (1, 3, or 7) then this function
        finds which rank you have multiples of.
        """
        hand_ranks = [card[0] for card in hand]  # The ranks of the cards in hand
        board_ranks = [card[0] for card in board]  # The ranks of the cards in board

        if type == 1:  # A pair has two cards of the same type

            if hand_ranks[0] == hand_ranks[1]:  # If you have a pocket pair
                return hand_ranks[0]

            else:
                for rank in hand_ranks:  # Loop through the ranks in your hand
                    if board_ranks.count(rank) == 1:  # If the pair is made with a card on the board
                        return rank

        elif type == 3:  # A three of a kind has three cards of the same type
            if hand_ranks[0] == hand_ranks[1]:  # If you have a pocket pair
                return hand_ranks[0]

            else:
                for rank in hand_ranks:  # Loop through the ranks in your hand
                    if board_ranks.count(rank) == 2:  # If there are two of this card on the board
                        return rank

        else:  # A four of a kind has four cards of the same type
            if hand_ranks[0] == hand_ranks[1] and board_ranks.count(hand_ranks[0]) == 2:  # If you have a pocket pair
                return hand_ranks[0]

            else:
                for rank in hand_ranks:  # Loop through the ranks in your hand
                    if board_ranks.count(rank) == 3:  # If there are three of this card on the board
                        return rank

    def find_two_pair(self, hand, board):
        """
        If you have two pair, then this function attempts to find which ranks you have pairs of.
        """
        hand_ranks = [card[0] for card in hand]  # The ranks of the cards in hand
        board_ranks = [card[0] for card in board]  # The ranks of the cards in board

        pairs = []  # Ranks you have found pairs of

        if hand_ranks[0] == hand_ranks[1]:  # If we have pocket pairs
            pairs.append(hand_ranks[0])

            for rank in board_ranks:  # Find what board cards gave us our second pair
                if board_ranks.count(rank) == 2 and rank not in pairs:
                    pairs.append(rank)

        else:
            for rank in hand_ranks:
                if rank in board_ranks:
                    pairs.append(rank)

            if len(pairs) != 2:
                for rank in board_ranks:  # Find what board cards gave us our second pair, if it exists
                    if board_ranks.count(rank) == 2 and rank not in pairs:
                        pairs.append(rank)

        if len(pairs) != 2:  # If both pairs on the board
            return None

        else:
            return pairs

    def get_rules(self, hand, opp_hand, board, delta, dependencies):
        """
        Takes the outcome of a showdown and attempts to extrapolate a rule from it. Does nothing if a rule could not be
        found. Otherwise, returns all lower and upper card values.
        """
        type = hand_type.get_type(hand, board)
        opp_type = hand_type.get_type(opp_hand, board)

        if delta == 0:  # If neither player won we cannot determine any rules
            if type == opp_type and type == 5:  # If we tied a flush (all five board cards have the same suit)
                suit = board[0][1]  # The suit the flush is made of
                rules = []  # The rules we were able to learn

                for card in hand:
                    if card[1] == suit:  # If it is the same suit as the flush, it is less than all cards on the board
                        for board_card in board:
                            rules.append((board_card[0], card[0]))

                for card in opp_hand:
                    if card[1] == suit:  # If it is the same suit as the flush, it is less than all cards on the board
                        for board_card in board:
                            rules.append((board_card[0], card[0]))

                if rules != []:  # If we were able to learn a rule
                    return rules

                else:
                    return None

            else:
                return None

        if type == opp_type:  # We can only extrapolate rules when we have the same type of hand as our opponent

            if type == 0:  # High card
                hand_ranks = [card[0] for card in hand]  # The ranks of the cards in hand
                opp_hand_ranks = [card[0] for card in opp_hand]  # The ranks of the cards in hand

                if delta > 0:  # If we won
                    if hand_ranks[1] in dependencies and hand_ranks[0] in dependencies[
                        hand_ranks[1]]:  # If we know our first card is less than our second card
                        upper = hand_ranks[1]
                        lower = []

                        for rank in opp_hand_ranks:  # Your high card is greater than both of your opponents
                            if rank != upper:
                                lower.append(rank)

                        return [(upper, i) for i in lower]

                    elif hand_ranks[0] in dependencies and hand_ranks[1] in dependencies[
                        hand_ranks[0]]:  # If we know our second card is less than our first card
                        upper = hand_ranks[0]
                        lower = []

                        for rank in opp_hand_ranks:  # Your high card is greater than both of your opponents
                            if rank != upper:
                                lower.append(rank)

                        return [(upper, i) for i in lower]

                else:  # If our opponent won
                    if opp_hand_ranks[1] in dependencies and opp_hand_ranks[0] in dependencies[
                        opp_hand_ranks[1]]:  # If we know our opponents first card is less than their second card
                        upper = opp_hand_ranks[1]
                        lower = []

                        for rank in hand_ranks:  # Your high card is greater than both of your opponents
                            if rank != upper:
                                lower.append(rank)

                        return [(upper, i) for i in lower]

                    elif opp_hand_ranks[0] in dependencies and opp_hand_ranks[1] in dependencies[
                        opp_hand_ranks[0]]:  # If we know our opponents second card is less than their first card
                        upper = opp_hand_ranks[0]
                        lower = []

                        for rank in hand_ranks:  # Your high card is greater than both of your opponents
                            if rank != upper:
                                lower.append(rank)

                        return [(upper, i) for i in lower]

            elif type == 1 or type == 3 or type == 7:  # Pair, three of a kind, or four of a kind
                rank = self.find_pair(hand, board, type)  # What rank did you have multiple of
                opp_rank = self.find_pair(opp_hand, board, type)  # What rank did your opponent have multiple of

                if rank == opp_rank or rank == None:  # If there was an error in finding rules (likely caused by a straight)
                    return None

                if delta > 0:  # If we won
                    upper = rank  # The rank of the high card
                    lower = opp_rank  # The rank of the lower card

                    return [(upper, lower)]

                else:  # If our opponent won
                    upper = opp_rank  # The rank of the high card
                    lower = rank  # The rank of the lower card

                    return [(upper, lower)]

            elif type == 2:  # Two pair
                pairs = self.find_two_pair(hand, board)  # What two ranks make up your two pair
                opp_pairs = self.find_two_pair(opp_hand, board)  # What two ranks make up your opponents two pair

                if pairs == None or opp_pairs == None or set(pairs) == set(
                        opp_pairs):  # If both of the pairs were on the board
                    return None

                for rank in pairs:
                    if rank in opp_pairs:  # If there is a common pair
                        pairs.remove(rank)
                        opp_pairs.remove(rank)

                if len(pairs) == 1 and len(opp_pairs) == 1:  # If there was a common pair
                    if delta > 0:  # If we won
                        upper = pairs[0]
                        lower = opp_pairs[0]

                        return [(upper, lower)]

                    else:  # If our opponent won
                        upper = opp_pairs[0]
                        lower = pairs[0]

                        return [(upper, lower)]

                elif len(pairs) == 2 and len(opp_pairs) == 2:  # If they were seperate pairs
                    if delta > 0:  # If we won
                        if pairs[1] in dependencies and pairs[0] in dependencies[
                            pairs[1]]:  # If we know our first card is less than our second card
                            upper = pairs[1]
                            lower = []

                            for rank in opp_pairs:  # Your high card is greater than both of your opponents
                                if rank != upper:
                                    lower.append(rank)

                            return [(upper, i) for i in lower]

                        elif pairs[0] in dependencies and pairs[1] in dependencies[
                            pairs[0]]:  # If we know our second card is less than our first card
                            upper = pairs[0]
                            lower = []

                            for rank in opp_pairs:  # Your high card is greater than both of your opponents
                                if rank != upper:
                                    lower.append(rank)

                            return [(upper, i) for i in lower]

                    else:  # If our opponent won
                        if opp_pairs[1] in dependencies and opp_pairs[0] in dependencies[
                            opp_pairs[1]]:  # If we know our opponents first card is less than their second card
                            upper = opp_pairs[1]
                            lower = []

                            for rank in pairs:  # Your high card is greater than both of your opponents
                                if rank != upper:
                                    lower.append(rank)

                            return [(upper, i) for i in lower]

                        elif opp_pairs[0] in dependencies and opp_pairs[1] in dependencies[
                            opp_pairs[0]]:  # If we know our opponents second card is less than their first card
                            upper = opp_pairs[0]
                            lower = []

                            for rank in pairs:  # Your high card is greater than both of your opponents
                                if rank != upper:
                                    lower.append(rank)

                            return [(upper, i) for i in lower]

            elif type == 5:  # A flush
                suit = self.find_flush(hand, board)  # What rank is our flush made of
                opp_suit = self.find_flush(opp_hand, board)  # What rank is our opponents flush made of

                if suit == None or opp_suit == None or suit != opp_suit:  # If the flush is not made up of four board cards for both players
                    return None

                else:
                    if delta > 0:  # If we won
                        for card in hand:
                            if suit in card:
                                upper = card[0]

                        for card in opp_hand:
                            if opp_suit in card:
                                lower = card[0]

                        return [(upper, lower)]

                    else:  # If our opponent won
                        for card in opp_hand:
                            if opp_suit in card:
                                upper = card[0]

                        for card in hand:
                            if suit in card:
                                lower = card[0]

                        return [(upper, lower)]

            elif type == 6:  # A full house
                ranks = self.find_full_house(hand, board)
                opp_ranks = self.find_full_house(opp_hand, board)

                if ranks == None or opp_ranks == None:
                    return None

                if ranks[0] == opp_ranks[0]:  # If both players have three of the same rank
                    if delta > 0:  # If we won
                        upper = ranks[1]
                        lower = opp_ranks[1]

                        return [(upper, lower)]

                    else:  # If our opponent won
                        upper = opp_ranks[1]
                        lower = ranks[1]

                        return [(upper, lower)]

                else:  # If both players have three of a different rank
                    if delta > 0:  # If we won
                        upper = ranks[0]
                        lower = opp_ranks[0]

                        return [(upper, lower)]

                    else:  # If our opponent won
                        upper = opp_ranks[0]
                        lower = ranks[0]

                        return [(upper, lower)]

            else:
                return None
                
    def showdown(self, hand, opp_hand, board, delta):
        """
        Takes in the results from a showdown and uses them to update the permutation our bot is using. Returns the new
        average permutation we are using.
        """
        rules = self.get_rules(hand, opp_hand, board, delta, self.dependencies) # Attempt to extrapolate rules

        if rules != None: # If we learned a rule, we are going to try to resample

            try:
                for upper, lower in rules: # Add all of the rules in
                    self.add_rule(upper, lower)

                self.ordering = self.topo() # Recompute the topological sort
                return rules
                

            except: # These rules generated a cyclic dependency, reset the adjacency matrices
                for upper, lower in rules:
                    self.dependencies[upper].remove(lower)
                    self.reverse_dependencies[lower].remove(upper)
                return None

        return None

    def topo(self):
        """
        Performs a topological sort on all ranks which we have learned a relative ranking for.
        """
        return list(toposort.toposort(self.dependencies))
