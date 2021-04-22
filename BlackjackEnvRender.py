
import enum
import random
import numpy as np
import gym
from gym import error, spaces, utils

ranks = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "jack": 10,
    "queen": 10,
    "king": 10,
    "ace": (1, 11)
}


class Suit(enum.Enum):
    spades = "spades"
    clubs = "clubs"
    diamonds = "diamonds"
    hearts = "hearts"


class Card:
    def __init__(self, suit, rank, value):
        self.suit = suit
        self.rank = rank
        self.value = value

    def __str__(self):
        return self.rank + " of " + self.suit.value


class Deck:
    def __init__(self, num=1):
        self.cards = []
        for i in range(num):
            for suit in Suit:
                for rank, value in ranks.items():
                    self.cards.append(Card(suit, rank, value))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop(0)

    def peek(self):
        if len(self.cards) > 0:
            return self.cards[0]

    def add_to_bottom(self, card):
        self.cards.append(card)

    def __str__(self):
        result = ""
        for card in self.cards:
            result += str(card) + "\n"
        return result

    def __len__(self):
        return len(self.cards)


# This follows the same, official rules every time.
# Still need to figure out what happens if there are multiple Aces.
def dealer_eval(player_hand):
    num_ace = 0
    use_one = 0
    for card in player_hand:
        if card.rank == "ace":
            num_ace += 1
            use_one += card.value[0]  # use 1 for Ace
        else:
            use_one += card.value

    if num_ace > 0:
        # See if using 11 instead of 1 for the Aces gets the
        # dealer's hand value closer to the [17, 21] range

        # The dealer will follow Hard 17 rules.
        # This means the dealer will not hit again if
        # the Ace yields a 17.

        # This also means that Aces initially declared as 11's can
        # be changed to 1's as new cards come.

        ace_counter = 0
        while ace_counter < num_ace:
            # Only add by 10 b/c 1 is already added before
            use_eleven = use_one + 10

            if use_eleven > 21:
                return use_one
            elif use_eleven >= 17 and use_eleven <= 21:
                return use_eleven
            else:
                # The case where even using Ace as eleven is less than 17.
                use_one = use_eleven

            ace_counter += 1

        return use_one
    else:
        return use_one


def player_eval(player_hand):
    num_ace = 0
    # use_one means that every ace that in the hand is counted as one.
    use_one = 0
    for card in player_hand:
        if card.rank == "ace":
            num_ace += 1
            use_one += card.value[0]  # use 1 for Ace
        else:
            use_one += card.value

    if num_ace > 0:
        # Define player policy for Aces:
        # Make Aces 11 if they get you to the range [18,21]
        # Otherwise, use one.

        ace_counter = 0
        while ace_counter < num_ace:
            # Only add by 10 b/c 1 is already added before
            use_eleven = use_one + 10

            if use_eleven > 21:
                return use_one
            elif use_eleven >= 18 and use_eleven <= 21:
                return use_eleven
            else:
                # This allows for some Aces to be 11s, and others to be 1.
                use_one = use_eleven

            ace_counter += 1

        return use_one
    else:
        return use_one

def dealer_turn(dealer_hand, deck):
    # Calculate dealer hand's value.
    dealer_value = dealer_eval(dealer_hand)

    # Define dealer policy (is fixed to official rules)

    # The dealer keeps hitting until their total is 17 or more
    while dealer_value < 17:
        # hit
        dealer_hand.append(deck.deal())
        dealer_value = dealer_eval(dealer_hand)

    return dealer_value, dealer_hand, deck




INITIAL_BALANCE = 1000
NUM_DECKS = 6


class BlackjackEnvRender(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BlackjackEnvRender, self).__init__()

        # Initialize the blackjack deck.
        self.bj_deck = Deck(NUM_DECKS)

        self.player_hand = []
        self.dealer_hand = []

        self.reward_options = {"lose": -100, "tie": 0, "win": 100}

        # hit = 0, stand = 1
        self.action_space = spaces.Discrete(2)

        '''
        First element of tuple is the range of possible hand values for the player. (3 through 20)
        This is the possible range of values that the player will actually have to make a decision for.
        Any player hand value 21 or above already has automatic valuations, and needs no input from an
        AI Agent. 

        However, we also need to add all the hand values that the agent could possibly end up in when
        they bust. Maybe the agent can glean some correlations based on what hand value they bust at,
        so this should be in the observation space. Also, the layout of OpenAI Gym environment class
        makes us have to include the bust-value in the step() function because we need to return that
        done is true alongside the final obs, which is the bust-value.
        '''

        # Second element of the tuple is the range of possible values for the dealer's upcard. (2 through 11)
        self.observation_space = spaces.Tuple((spaces.Discrete(18), spaces.Discrete(10)))

        self.done = False

    def _take_action(self, action):
        if action == 0:  # hit
            self.player_hand.append(self.bj_deck.deal())

        # re-calculate the value of the player's hand after any changes to the hand.
        self.player_value = player_eval(self.player_hand)

    def step(self, action):
        self._take_action(action)

        # End the episode/game is the player stands or has a hand value >= 21.
        self.done = action == 1 or self.player_value >= 21

        # rewards are 0 when the player hits and is still below 21, and they
        # keep playing.
        rewards = 0

        if self.done:
            # CALCULATE REWARDS
            if self.player_value > 21:  # above 21, player loses automatically.
                rewards = self.reward_options["lose"]
            elif self.player_value == 21:  # blackjack! Player wins automatically.
                rewards = self.reward_options["win"]
            else:
                ## Begin dealer turn phase.

                dealer_value, self.dealer_hand, self.bj_deck = dealer_turn(self.dealer_hand, self.bj_deck)

                ## End of dealer turn phase

                # ------------------------------------------------------------#

                ## Final Compare

                if dealer_value > 21:  # dealer above 21, player wins automatically
                    rewards = self.reward_options["win"]
                elif dealer_value == 21:  # dealer has blackjack, player loses automatically
                    rewards = self.reward_options["lose"]
                else:  # dealer and player have values less than 21.
                    if self.player_value > dealer_value:  # player closer to 21, player wins.
                        rewards = self.reward_options["win"]
                    elif self.player_value < dealer_value:  # dealer closer to 21, dealer wins.
                        rewards = self.reward_options["lose"]
                    else:
                        rewards = self.reward_options["tie"]

        self.balance += rewards

        # Subtract by 1 to fit into the possible observation range.
        # This makes the possible range of 3 through 20 into 1 through 18
        player_value_obs = self.player_value - 2

        # get the value of the dealer's upcard, this value is what the agent sees.
        # Subtract by 1 to fit the possible observation range of 1 to 10.
        upcard_value_obs = dealer_eval([self.dealer_upcard]) - 1

        # the state is represented as a player hand-value + dealer upcard pair.
        obs = np.array([player_value_obs, upcard_value_obs])

        return obs, rewards, self.done, {}

    def reset(self):  # resets game to an initial state
        # Add the player and dealer cards back into the deck.
        self.bj_deck.cards += self.player_hand + self.dealer_hand

        # Shuffle before beginning. Only shuffle once before the start of each game.
        self.bj_deck.shuffle()

        self.balance = INITIAL_BALANCE

        self.done = False

        # returns the start state for the agent
        # deal 2 cards to the agent and the dealer
        self.player_hand = [self.bj_deck.deal(), self.bj_deck.deal()]
        self.dealer_hand = [self.bj_deck.deal(), self.bj_deck.deal()]
        self.dealer_upcard = self.dealer_hand[0]

        # calculate the value of the agent's hand
        self.player_value = player_eval(self.player_hand)

        # Subtract by 1 to fit into the possible observation range.
        # This makes the possible range of 2 through 20 into 1 through 18
        player_value_obs = self.player_value - 2

        # get the value of the dealer's upcard, this value is what the agent sees.
        # Subtract by 1 to fit the possible observation range of 1 to 10.
        upcard_value_obs = dealer_eval([self.dealer_upcard]) - 1

        # the state is represented as a player hand-value + dealer upcard pair.
        obs = np.array([player_value_obs, upcard_value_obs])

        return obs

    def render(self, mode='human', close=False):
        # convert the player hand into a format that is
        # easy to read and understand.
        hand_list = []
        for card in self.player_hand:
            hand_list.append(card.rank)

        # re-calculate the value of the dealer upcard.
        upcard_value = dealer_eval([self.dealer_upcard])

        print(f'Balance: {self.balance}')
        print(f'Player Hand: {hand_list}')
        print(f'Player Value: {self.player_value}')
        print(f'Dealer Upcard: {upcard_value}')
        print(f'Done: {self.done}')

        print()