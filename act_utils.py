"""
A set of utilities used to make decisions by bots implementing counterfactual regret minimization (CFR).
"""

import numpy as np
import pickle

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction

continue_cost_cutoffs = np.array([2.5, 10, 22.5, 40, 62.5, 90, 117.5, 145, 172.5])
grand_total_cutoffs = np.array([5, 20, 45, 80, 125, 180, 235, 290, 345])
strength_cutoffs = np.array([.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]) # Only used for post-flop hands

with open("preflop_odds.pickle", "rb") as file: # A dictionary of pre-computed preflop odds and hand types
    PREFLOP_ODDS = pickle.load(file)

preflop_strengths = [PREFLOP_ODDS[strength][0][0] for strength in PREFLOP_ODDS] # Make a list of all preflop hand strengths
percentiles = np.percentile(np.array(preflop_strengths), np.arange(1, 100)) # Convert hand strengths into percentiles

def get_bucket(button, continue_cost, pot_after_continue, street, strength):
    """
    Places a game state into a single bucket. Takes in button (0 if small blind, 1 if big blind), the cost to continue
    the round of betting, the grand total in the pot after you continue, the street, and card strength.
    """
    continue_bucket = np.searchsorted(continue_cost_cutoffs, continue_cost)
    grand_total_bucket = np.searchsorted(grand_total_cutoffs, pot_after_continue)

    if street == 0: # If we are preflop, find what percentile our hand strength is in (100 options)
        strength_bucket = np.searchsorted(percentiles, strength)

    else:
        street = street - 2 # Ensures streets have the value 0, 1, 2, or 3
        strength_bucket = np.searchsorted(strength_cutoffs, strength)
        # strength_bucket = int(min(strength * 100, 99))

    return (40000 * button) + (10000 * street) + (1000 * continue_bucket) + (100 * grand_total_bucket) + strength_bucket

def calculate_strategy(regrets):
    """
    Given the regrets of taking each of the four possible actions for a given bucket, computes a probability
    distribution over each of the actions, where the action with the highest regret of not choosing it is the action
    that is most likely to be selected. All negative regrets are ignored. The outputs list must be a numpy array.
    """
    regrets = np.maximum(regrets, 0)
    regrets = regrets.clip(min=0) # Ignore all negagtive regrets

    total = np.sum(regrets)

    if total == 0: # If there are no positive regrets, return uniformly even odds of choosing any action.
        return np.ones(4) / 4

    else: # Otherwise, compute a probability distribution over actions with positive regrets
        return regrets / total

def act(action, legal_actions, round_state, pip, continue_cost, pot_after_continue):
    """
    Takes in one of four possible action types as a number in [0, 1, 2, 3] and executes it for the player. Requires that
    it is provided a list of legal actions, the current round state, the pip, cost to continue the round of betting, and
    the pot size if you choose to continue.
    """
    if action == 0:  # Check or fold
        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()

    elif action == 1:  # Check or call
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

    elif action == 2:  # Bet roughly .75 of the pot
        bet = min(np.random.normal(loc=.75, scale=.5), 2)

    else:  # Bet roughly 1.25 of the pot
        bet = min(np.random.normal(loc=1.25, scale=.5), 2)

    if RaiseAction in legal_actions:
        min_raise, max_raise = round_state.raise_bounds()  # The smallest and largest number of chips for a legal bet/raise
        commit_amount = int(pip + continue_cost + (bet * pot_after_continue))
        commit_amount = max(commit_amount, min_raise)
        commit_amount = min(commit_amount, max_raise)

        return RaiseAction(commit_amount)

    elif CallAction in legal_actions:
        return CallAction()

    else:  # Only legal action that keeps us in the game
        return CheckAction()