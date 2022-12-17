"""
A utility to generate monte-carlo simulations used to calculate the equity of a poker hand during any street. This will
not be required for the pre-flop betting stage as the file preflop_odds.pickle stores pre-computed preflop hand strengths.
This utilizes the deuces library, which was compiled locally in C.
"""

import equity_calc

def get_deck():
    """
    Creates an instance of a deck.
    """
    return equity_calc.Deck()

def get_evaluator():
    """
    Creates an instance of the EvaluatorNumpy class which can evaluate hands.
    """
    return equity_calc.EvaluatorNumpy()

def card_to_str(card):
    """
    Given a card represented as a binary integer, converts it to a string representation.
    """
    return equity_calc.Card.int_to_str(card)

def str_to_card(card):
    """
    Given a string representation of a card, converts it to a binary integer.
    """
    return equity_calc.Card.new(card)

def get_strength(hand, board, evaluator, iters=100):
    """
    Runs a monte carlo simulation for iters iterations to approximate the hand strength at any point in the game.
    """
    cards = hand + board # Convert all cards to binary integers and put them in a single list
    win_tie_prob = evaluator.analyze_hand(cards, n_players=2, n_sims=iters)
    return win_tie_prob # The first index is the probability of winning with that hand, the second is tying