"""
http://outlace.com/rlpart2.html
"""
from __future__ import division
from pdb import set_trace as debug
import math
import random
from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


class GameState(object):
    status_dict = {1: "In progress", 2: "Player wins", 3: "Draw", 4: "Player loses"}

    def __init__(self, player_hand, dealer_hand, status):
        self.player_hand = player_hand
        self.dealer_hand = dealer_hand
        self.status = status

    def __str__(self):
        return "(%s, %s, %s)" % (self.player_hand, self.dealer_hand, self.status)

    def read_status_string(self):
        """Read the game status as string

        :param status: Game status
        :return:
        """
        return self.status_dict[self.status]


def random_card():
    card = random.randint(1, 13)
    if card > 10:
        card = 10
    return card


def usable_ace(hand):
    """Check whether a usable ace exists

    :param hand: A hand is a tuple e.g. (14, False), i.e. a total card value of 14 without a usable ace
    :return:
    """
    val, ace = hand
    # accept a hand, if the Ace can be an 11 without busting the hand, it's usable
    return (ace) and ((val + 10) <= 21)


def total_value(hand):
    val, ace = hand
    if usable_ace(hand):
        return val + 10
    return val


def add_card(hand, card):
    val, ace = hand
    if card == 1:  # If card is ace
        ace = True
    return (val + card, ace)


def eval_dealer(dealer_hand):
    while total_value(dealer_hand) < 17:
        dealer_hand = add_card(dealer_hand, random_card())
    return dealer_hand


def play(state, action):
    """

    :param state: Game state, (player total, usable_ace), (dealer total, usable ace), game_status;
                  Example ((15, True), (9, False), 1)
    :param action: stay or hit action => = 0 or 1
    :return:
    """
    if action == 0:  # action = stay
        state.dealer_hand = eval_dealer(state.dealer_hand)

        player_tot = total_value(state.player_hand)
        dealer_tot = total_value(state.dealer_hand)
        if dealer_tot > 21:
            state.status = 2  # player wins
        elif dealer_tot == player_tot:
            state.status = 3  # draw
        elif dealer_tot < player_tot:
            state.status = 2
        elif dealer_tot > player_tot:
            state.status = 4  # player loses

    elif action == 1:  # action = hit
        state.player_hand = add_card(state.player_hand, random_card())
        state.dealer_hand = eval_dealer(state.dealer_hand)
        player_tot = total_value(state.player_hand)
        if player_tot == 21:
            if total_value(state.dealer_hand) == 21:
                state.status = 3  # draw
            else:
                state.status = 2  # player wins!
        elif player_tot > 21:
            state.status = 4  # player loses
        elif player_tot < 21:
            state.status = 1
    return state


def init_game():
    """Start a game of blackjack

    :return:
    """
    # status: 1 = in progress; 2 = player won; 3 = draw;
    #         4 = dealer won/player loses
    status = 1
    player_hand = add_card((0, False), random_card())
    player_hand = add_card(player_hand, random_card())
    dealer_hand = add_card((0, False), random_card())

    # evaluate if player wins from first hand
    if total_value(player_hand) == 21:
        if total_value(dealer_hand) != 21:
            status = 2  # player wins after first deal!
        else:
            status = 3  # draw

    return GameState(player_hand, dealer_hand, status)


def init_state_space():
    """Create a list of all the possible states

    :return:
    """
    states = []
    for card in range(1, 11):
        for val in range(11, 22):
            states.append((val, False, card))
            states.append((val, True, card))
    return states


def init_state_actions(states):
    """Create a dictionary (key-value pairs) of all possible state-actions and their values

    This creates our Q-value look up table
    """
    av = {}
    for state in states:
        av[(state, 0)] = 0.0
        av[(state, 1)] = 0.0
    return av


#
state = init_game()
print(state)

while state.status == 1:
    state = play(state, 1)
    print(state)

logger.info("ALL DONE\n")
