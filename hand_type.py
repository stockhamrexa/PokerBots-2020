"""
A utility to determine what type of hand you have at any stage of the game (i.e. high card, full house, etc). It assumes
a standard deck permutation.
"""

import eval7 as eval

# All cards in a standard 52 card deck
cards = ["2c", "2d", "2h", "2s", "3c", "3d", "3h", "3s", "4c", "4d", "4h", "4s", "5c", "5d", "5h", "5s", "6c", "6d",
         "6h", "6s", "7c", "7d", "7h", "7s", "8c", "8d", "8h", "8s", "9c", "9d", "9h", "9s", "Tc", "Td", "Th", "Ts",
         "Jc", "Jd", "Jh", "Js", "Qc", "Qd", "Qh", "Qs", "Kc", "Kd", "Kh", "Ks", "Ac", "Ad", "Ah", "As"]

# Maps all possible hand types to an integer
hand_map = {
    "High Card": 0,
    "Pair": 1,
    "Two Pair": 2,
    "Trips": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Quads": 7,
    "Straight Flush": 8,
    "Royal Flush": 9
}

# The ordering of ranks
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

def get_street(board):
    """
    Takes in all board cards and returns what street you are on. Streets are represented as 0, 3, 4, 5 for preflop, flop,
    river, and turn respectively.
    """
    return len(board)

def get_type(hand, board):
    """
    Given your hand and board cards, both represented as a list of strings, return what type of hand it is. All possible
    hand types, in order of increasing strength, are: High card, pair, two pair, three of a kind, straight, flush, full
    house, four of a kind, straight flush, royal flush.
    """
    formatted_hand = [eval.Card(card) for card in hand]
    formatted_board = [eval.Card(card) for card in board]

    board_score = eval.evaluate(formatted_board) # The hand strength of just the board cards
    full_score = eval.evaluate(formatted_hand + formatted_board) # The hand strength of the hand and board cards together

    board_eval = eval.hand_type(board_score) # The hand type of just the board cards
    full_eval = eval.hand_type(full_score) # The hand type of the hand and board cards together

    if board_eval == full_eval: # If the best five card hand is potentially being made without any cards from your hand
        if full_score > board_score: # The best hand includes your hole cards
            return hand_map[full_eval]

        else: # Return the hand type of just your hole cards
            hand_ranks = [card[0] for card in hand]

            if len(set(hand_ranks)) == 1:  # Both cards have the same rank
                return hand_map["Pair"]

            return hand_map["High Card"]

    return hand_map[full_eval]