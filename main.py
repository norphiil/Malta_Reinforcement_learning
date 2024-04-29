import random

CARD = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
}


class Card:
    def __init__(self, card, suit: set, value: str):
        self.card = card
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.card} of {self.suit} | Value {self.value}"


class Deck:
    def __init__(self):
        self.create()

    def create(self):
        self.cards = []
        for suit in ["♥", "♦", "♣", "♠"]:
            for card in CARD:
                self.cards.append(Card(card, suit, CARD[card]))

    def reset(self):
        self.create()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        return self.cards.pop()


class User:
    def __init__(self):
        self.cards: list[Card] = []

    def reset(self):
        self.cards = []

    def add_card(self, card: Card):
        self.cards.append(card)

    def get_score(self):
        score = 0
        num_aces = 0
        card: Card
        for card in self.cards:
            score += card.value
            if card.card == "A":
                num_aces += 1

        while score > 21 and num_aces:
            score -= 10
            num_aces -= 1

        return score

    def __str__(self):
        return ", ".join(str(card) for card in self.cards) + f"\n Score: {self.get_score()}"


class Player(User):
    def __init__(self):
        super().__init__()


class Dealer(User):
    def __init__(self):
        super().__init__()


class PlayerRLAgent(Player):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__()
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {}  # Q(s, a) table

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["h", "s"])  # explore
        else:
            return self.get_best_action(state)  # exploit

    def get_best_action(self, state):
        if state not in self.q_table:
            print(f"State {state} not in Q-table")
            self.q_table[state] = {"h": 0.0, "s": 0.0}
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        if next_state is None:
            next_state_value = 0
        else:
            next_state_value = max(self.q_table.get(next_state, {"h": 0.0, "s": 0.0}).values())
        current_value = self.q_table.get(state, {"h": 0.0, "s": 0.0}).get(action, 0.0)
        new_value = (1 - self.alpha) * current_value + self.alpha * (reward + self.gamma * next_state_value)
        self.q_table.setdefault(state, {"h": 0.0, "s": 0.0})[action] = new_value


class BlackJackWithRL:
    def __init__(self, deck: Deck, player, dealer):
        self.deck: Deck = deck
        self.player: Player = player
        self.dealer: Dealer = dealer

    def play(self):
        self.deck.shuffle()
        self.player.add_card(self.deck.draw_card())
        self.player.add_card(self.deck.draw_card())
        self.dealer.add_card(self.deck.draw_card())
        state = (self.player.get_score(), self.dealer.cards[0].value)

        print("Player: ", self.player)
        print("Dealer: ", self.dealer)
        # Player's turn
        while self.player.get_score() < 21:
            action = None
            if isinstance(self.player, PlayerRLAgent):
                state = (self.player.get_score(), self.dealer.cards[0].value)
                action = self.player.choose_action(state)
            else:
                action = input("Do you want to hit or stand? (h/s): ")

            if action == "h":
                self.player.add_card(self.deck.draw_card())
                print("Player: ", self.player)
            elif action == "s":
                break

        if self.player.get_score() > 21:
            print("Player busts! Dealer wins.")
            if isinstance(self.player, PlayerRLAgent):
                self.player.update_q_table(state, action, -1, None)
            return

        # Dealer's turn
        while self.dealer.get_score() < 17:
            self.dealer.add_card(self.deck.draw_card())
            print("Dealer: ", self.dealer)

        if self.dealer.get_score() > 21:
            print("Dealer busts! Player wins.")
            if isinstance(self.player, PlayerRLAgent):
                self.player.update_q_table(state, action, 1, None)
            return

        if self.player.get_score() > self.dealer.get_score():
            print("Player wins!")
            if isinstance(self.player, PlayerRLAgent):
                self.player.update_q_table(state, action, 1, None)
        elif self.player.get_score() < self.dealer.get_score():
            print("Dealer wins!")
            if isinstance(self.player, PlayerRLAgent):
                self.player.update_q_table(state, action, -1, None)
        else:
            print("It's a tie!")

    def game_reset(self):
        self.player.reset()
        self.dealer.reset()
        self.deck.reset()

    def start(self, num_episodes=1000):
        for i in range(num_episodes):
            print(f"Episode {i}")
            self.game_reset()
            self.play()


# Create RL agents for player and  create the dealer
player = PlayerRLAgent()
dealer = Dealer()

deck = Deck()
print(f"Deck of {len(deck.cards)} cards created")

# Create Blackjack game with RL agents
game_with_rl = BlackJackWithRL(deck, player, dealer)
game_with_rl.start(100)
