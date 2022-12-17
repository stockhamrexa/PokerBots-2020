import act_utils
import hand_type
import pickle
import numpy as np
import permutation_solver

from hand_type import get_type
from hand_sim import get_evaluator, get_strength, str_to_card
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Node:
    def __init__(self, name, index):
        self.name = name
        self.edges = []
        self.index = index


    def getName(self):
        return self.name

    def addEdge(self, node):
        if node not in self.edges:
            self.edges.append(node)
            # sorted(self.edges, key=valueOfOrderDict.get)

    def getIndex(self):
        return self.index

class Player(Bot):
    '''
    A bot that plays using a Nash equilibrium computed using counterfactual regret minimization. Plays under the
    assumption that there is no permutation of ranks.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.cache = {} # A cache of hand strengths that we have seen so that they do not have to be computed again

        with open('card_lookups.pickle', 'rb') as file:
            self.card_map = pickle.load(file) # A mapping of all cards in a deck to their corresponding Card object

        self.checkfold = False  # If we should check/ fold the rest of the game to guarantee a win
        self.evaluator = get_evaluator() # Evaluator object used to calculate hand strength
        self.parentDictionary = {'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],'T':[],'J':[],'Q':[],'K':[],'A':[]}
        self.predictedOrder = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
        self.values = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.P = permutation_solver.Permutation()
        self.confidenceInterval = 6e9

        with open('preflop_odds.pickle', 'rb') as file:
            self.preflop_odds = pickle.load(file) # A dictionary of pre-computed preflop odds and hand types

        self.shove_counts = np.array([0, 0, 0, 0])  # The number of times we have shoved preflop/postflop

        self.regretsum = np.genfromtxt('regretsum_small_2.csv', delimiter=',') # A numpy array of regrets for each bucket
        self.stratsum = np.genfromtxt('stratsum_small_2.csv', delimiter=',')  # A numpy array of strategies for each bucket

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        print(self.predictedOrder)
        self.hand = round_state.hands[active] # Your hand
        self.board = [] # The board cards

        self.formatted_hand = [self.card_map[i] for i in round_state.hands[active]] # Your hand, converted to Card objects
        self.formatted_board = [] # The board cards, converted to Card objects

        self.street = 0 # What street you are on
        self.shoved = [False, False] # Did you shove preflop or postflop
        self.strength, self.type = self.calc() # Hand strength and hand type

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        if not self.checkfold: # If we are not using a checkfold strategy, see if we can
            self.play_checkfold(game_state)

        opp_hand = terminal_state.previous_state.hands[1-active]

        if opp_hand != [] and self.street == 5:
            my_delta = terminal_state.deltas[active]

            rules = self.P.showdown(self.hand, opp_hand, self.board, my_delta)
            #rules = self.get_rules(self.hand, opp_hand, self.board, my_delta)
            if rules != None:
                self.confidenceInterval = self.confidenceInterval/2
                for rule in rules:
                    print("RULE", rule)
                    self.predictedOrder, self.parentDictionary = self.checkSwapCondition(self.predictedOrder,
                                        self.parentDictionary, rule[1], rule[0], True)

        went_to_postflop = 1 if self.street > 0 else 0  # Did we make it past preflop
        self.shove_counts += np.array([int(self.shoved[0]), 1, int(self.shoved[1]), went_to_postflop])  # Update shove counts

        self.update_board(terminal_state.previous_state)  # Updates the board before going to the showdown (self.get_action is not called when both players go all in before the river)

    def get_action(self, game_state, round_state, active):
        '''
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # The actions you are allowed to take

        if self.checkfold or len(legal_actions) == 1: # Plays a strategy of only checking and folding.
            if CheckAction in legal_actions:
                return CheckAction()

            return FoldAction()

        self.update_board(round_state) # Update the street and board cards if the street has changed, along with hand strength and type

        active = round_state.button % 2 # Are we the small blind (0) or big blind (1)
        pip = round_state.pips[active]  # The number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1 - active]  # The number of chips your opponent has contributed to the pot this round of betting
        continue_cost = opp_pip - pip  # The number of chips needed to stay in the pot

        stack = round_state.stacks[active]  # The number of chips you have remaining
        opp_stack = round_state.stacks[1 - active]  # The number of chips your opponent has remaining
        contribution = STARTING_STACK - stack  # The number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # The number of chips your opponent has contributed to the pot
        pot_after_continue = contribution + opp_contribution + continue_cost  # Pot size if you continue

        pot_odds = float(continue_cost) / pot_after_continue  # Compute the equity needed for it to be worth it to continue

        if pot_odds >= .4:  # Shove behavior
            action = self.play_shove(continue_cost, pot_odds, self.strength)

        else:
            bucket = act_utils.get_bucket(active, continue_cost, pot_after_continue, self.street, self.strength) # Bucket the game state
            outputs = self.stratsum[bucket] # See how many times we have visited this bucket

            if np.sum(outputs) < 1000:
                action = self.play_default(continue_cost, pot_odds, self.strength)

            else:
                strategy = act_utils.calculate_strategy(self.stratsum[bucket])
                action = np.searchsorted(np.cumsum(strategy), np.random.random())

        return act_utils.act(action, legal_actions, round_state, pip, continue_cost, pot_after_continue)

    def calc(self):
        """
        Calculates the probability of winning this hand given the cards in your hand and on the board.
        """
        hand = []
        board = []
        for card in self.hand:
            number = card[:-1]
            suit = card[-1]
            currentCard = self.predictedOrder.index(number)
            hand.append(self.values[currentCard] + suit)

        for card in self.board:
            number = card[:-1]
            suit = card[-1]
            currentCard = self.predictedOrder.index(number)
            board.append(self.values[currentCard] + suit)

        if self.street == 0: # If we are preflop, perform a lookup
            cards = "".join(hand)
            return self.preflop_odds[cards][0] # The first index is the probability of winning

        else:
            key1 = frozenset("".join(hand))
            key2 = frozenset("".join(board))

            if key1 in self.cache and key2 in self.cache[key1]: # If we have seen this hand and board before
                return self.cache[key1][key2]

            else:
                strength = get_strength(self.formatted_hand, self.formatted_board, self.evaluator, iters=100)
                type = hand_type.get_type(hand, board)
                if get_type(hand, board) == 4:
                    strength[0] = strength[0]/2
                self.cache[key1] = {key2: [strength[0], type]}
                return strength[0], type

    def play_checkfold(self, game_state):
        """
        Calculates whether or not it is possible to check/fold the rest of the game, sacrificing 3 dollars every two
        hands to come out on top.
        """
        bankroll = game_state.bankroll # How much you have earned over the course of the game
        rounds_left = NUM_ROUNDS - game_state.round_num + 1

        if bankroll > 0: # If you are losing, never convert to a check/fold strategy
            chips_lost = 1.5 * rounds_left

            if bankroll - chips_lost > 0: # If you can check and fold and still be the winner
                self.checkfold = True

    def play_default(self, continue_cost, pot_odds, strength):
        """
        The default strategy to play with. Takes into account the continue cost, pot odds, and hand strength.
        """
        if strength > .85:
            return 3

        if continue_cost > 0:  # Our opponent has raised the stakes
            if continue_cost > 1:
                strength -= .25  # Intimidation factor

            # Is it worth it to stay in the game?
            if strength >= pot_odds:  # Staying in the game has positive EV
                if strength > .5 and np.random.random() < strength:  # Commit more sometimes
                    return 2

                return 1

            else:  # Staying in the game has negative EV
                return 0

        else:
            if np.random.random() < strength:  # Balance bluffs with value bets
                return 2
            return 1

    def play_shove(self, continue_cost, pot_odds, strength):
        """
        The strategy to take if we decide that shoving is a viable option. Takes into account the continue cost, pot
        odds, and hand strength.
        """
        if self.street == 0:
            self.shoved[0] = True
            shoves = self.shove_counts[0] + 1 # The number of times your opponent has shoved preflop
            hands = self.shove_counts[1] + 1 # The number of times you have been to the preflop

        else:
            self.shoved[1] = True
            shoves = self.shove_counts[2] + 1 # The number of times your opponent has shoved postflop
            hands = self.shove_counts[3] + 1 # The number of times you have been to the postflop

        if hands < 50:  # Inexperienced shoving
            return self.play_default(continue_cost, pot_odds, strength)

        # Undercut opponent shove frequency
        if strength >= min(1 - float(shoves) / hands / 2, .99):
            return 3

        else:
            return 0

    def update_board(self, round_state):
        """
        Called by get_action. Checks to see if the street has changed, updating the street and the board as well as hand
        strength and hand type if it has to minimize the number of calls to the time intensive str_to_card and calc
        functions.
        """
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively

        if street != self.street: # If we changed streets, update the board cards
            self.street = street
            self.board = round_state.deck[:street]
            self.formatted_board = [self.card_map[i] for i in round_state.deck[:street]]

            self.strength, self.type = self.calc()  # Calculate hand strength and hand type

    def createTree(self, parentDictionary, cardObjectDictionary, topNode, index):
        for child in parentDictionary[topNode]:
            if cardObjectDictionary[child].getIndex() > index:
                cardObjectDictionary[topNode].addEdge(cardObjectDictionary[child])
                self.createTree(parentDictionary, cardObjectDictionary, child, index)

    def dep_resolve(self, node, resolved, unresolved):
       unresolved.append(node)
       for edge in node.edges:
          if edge not in resolved:
             if edge in unresolved:
                print("Exception, straight probably caused this")
             self.dep_resolve(edge, resolved, unresolved)
       resolved.append(node)
       unresolved.remove(node)
       return resolved


    def swap(self, predictedOrder, parentDictionary, lesserCard, index):
        two = Node('2', predictedOrder.index('2'))
        three = Node('3', predictedOrder.index('3'))
        four = Node('4', predictedOrder.index('4'))
        five = Node('5', predictedOrder.index('5'))
        six = Node('6', predictedOrder.index('6'))
        seven = Node('7', predictedOrder.index('7'))
        eight = Node('8', predictedOrder.index('8'))
        nine = Node('9', predictedOrder.index('9'))
        ten = Node('T', predictedOrder.index('T'))
        jack = Node('J', predictedOrder.index('J'))
        queen = Node('Q', predictedOrder.index('Q'))
        king = Node('K', predictedOrder.index('K'))
        ace = Node('A', predictedOrder.index('A'))
        cardObjectDictionary = {'2':two,'3':three,'4':four, '5':five,'6':six,'7':seven, '8':eight,'9':nine,'T':ten,'J':jack,'Q':queen,'K':king,'A':ace}
        objectCardDictionary = {cardObjectDictionary[key]: key for key in cardObjectDictionary.keys()}
        # valueOfOrderDict = {}
        # for x in range(13):
        #     valueOfOrderDict[cardObjectDictionary[predictedOrder[x]]] = x
        self.createTree(parentDictionary, cardObjectDictionary, lesserCard,index)
        addList = self.dep_resolve(cardObjectDictionary[lesserCard], [], [])
        for item in addList[::-1]:
            temp = predictedOrder.pop(predictedOrder.index(objectCardDictionary[item]))
            predictedOrder.insert(index, temp)
        return predictedOrder

        # predictedOrder.insert(index, predictedOrder.pop(predictedOrder.index(lesserCard)))
        # for child in parentDictionary[lesserCard]:
        #     if predictedOrder.index(child) < index:
        #         predictedOrder = swap(predictedOrder,parentDictionary,child,index)

    def alreadyLesser(self, parentDictionary, greaterCard, lesserCard):
        if lesserCard in parentDictionary[greaterCard]:
            return True
        else:
            for card in parentDictionary[greaterCard]:
                if self.alreadyLesser(parentDictionary, card, lesserCard):
                    return True
        return False

    #predictedOrder is the current permutation that we believe
    #parentDictionary keeps track of all the comparisons that don't result in a swap
    #Card A is our card, Card B is oppent card
    #whichIsGreater is a boolean, False if we won, True if they won
    def checkSwapCondition(self, predictedOrder, parentDictionary, cardA, cardB, whichIsGreater):
        #If they won and they shouldn't have, our card is valued as less, but we value it as more
        try:
            if whichIsGreater and predictedOrder.index(cardA) > predictedOrder.index(cardB):
                #the index of the card we thought was lesser
                lowestIndex = predictedOrder.index(cardB)
                predictedOrder = self.swap(predictedOrder, parentDictionary, cardA, lowestIndex)
                if not self.alreadyLesser(parentDictionary, cardB, cardA):
                    parentDictionary[cardB].append(cardA)
            #If we won and we shouldn't have, our card is valued as more, but we value it as less
            elif not whichIsGreater and predictedOrder.index(cardA) < predictedOrder.index(cardB):
                #get index where
                lowestIndex = predictedOrder.index(cardA)
                predictedOrder = self.swap(predictedOrder, parentDictionary, cardB, lowestIndex)
                if not self.alreadyLesser(parentDictionary, cardA, cardB):
                    parentDictionary[cardA].append(cardB)
            #If they won and they should have, we have cards in correect location
            elif whichIsGreater and predictedOrder.index(cardA) < predictedOrder.index(cardB):
                if not self.alreadyLesser(parentDictionary, cardB, cardA):
                    parentDictionary[cardB].append(cardA)
            #If we won and we should have, we have the cards in the correct location
            elif not whichIsGreater and predictedOrder.index(cardA) > predictedOrder.index(cardB):
                if not self.alreadyLesser(parentDictionary, cardA, cardB):
                    parentDictionary[cardA].append(cardB)
            return predictedOrder, parentDictionary
        except:
            return predictedOrder, parentDictionary

if __name__ == '__main__':
    run_bot(Player(), parse_args())