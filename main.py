import random
import math

CARD = {
    "A": 11,
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
        self.usable_ace = False
        
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
            
        if num_aces >= 1:
            self.usable_ace = True
        else:
            self.usable_ace = False

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
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.9):
        super().__init__()
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {}  # Q(s, a) table
        self.q_table_counts = {}  # Q(s, a) table for counts

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["h", "s"])  # explore
        else:
            return self.get_best_action(state)  # exploit

    def get_best_action(self, state):
        #print("Choosing best action")
        if state not in self.q_table:
            #print(f"State {state} not in Q-table")
            self.q_table[state] = {"h": 0.0, "s": 0.0}
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        self.q_table_counts.setdefault(state, {}).setdefault(action, 0)
        self.q_table_counts[state][action] += 1
        
        if next_state is None:
            next_state_value = 0
        else:
            next_state_value = max(self.q_table.get(next_state, {"h": 0.0, "s": 0.0}).values())
            #print("\n*****Next state value = %5.2f*****\n" % (next_state_value))
            
        current_value = self.q_table.get(state, {"h": 0.0, "s": 0.0}).get(action, 0.0)
        new_value = (1 - self.alpha) * current_value + self.alpha * (reward + self.gamma * next_state_value)
        self.q_table.setdefault(state, {"h": 0.0, "s": 0.0})[action] = new_value
        
    def print_q_table(self):
        sorted_q_table = sorted(self.q_table.items(), key=lambda x: (x[0][0], x[0][1]))

        for key, value in sorted_q_table:
            print(key, value)
        
        print("__________________________________________")
            
        sorted_q_table_c = sorted(self.q_table_counts.items(), key=lambda x: (x[0][0], x[0][1]))

        for key, value in sorted_q_table_c:
            print(key, value)


class BlackJackWithRL:
    def __init__(self, deck: Deck, player: Player, dealer: Dealer):
        self.deck = deck
        self.player = player
        self.dealer = dealer

    def play(self, choose_action_fn=None, episode_count=None):
        self.deck.shuffle()
        self.player.add_card(self.deck.draw_card())
        self.player.add_card(self.deck.draw_card())
        self.dealer.add_card(self.deck.draw_card())

        print("Player: ", self.player)
        print("Dealer: ", self.dealer)
        
        # Player's turn
        while self.player.get_score() < 12:
            self.player.add_card(self.deck.draw_card())
            print("Player [Automatic Draw]: ", self.player)

        if self.player.get_score() == 21:
            state = None

        episode = []

        while self.player.get_score() < 21:
            state = (self.player.get_score(), self.dealer.cards[0].value, self.player.usable_ace)
            if choose_action_fn:
                action = choose_action_fn(state, episode_count, len(episode))
            elif isinstance(self.player, PlayerRLAgent):
                state = (self.player.get_score(), self.dealer.cards[0].value, self.player.usable_ace)
                action = self.player.choose_action(state)
            else:
                action = input("Do you want to hit or stand? (h/s): ")

            episode.append((state, action))

            if action == "h":
                self.player.add_card(self.deck.draw_card())
                print("Player: ", self.player)
            elif action == "s":
                #print("Player has chosen to stand")
                break

        if self.player.get_score() > 21:
            #print("Player busts! Dealer wins.\n")
            if choose_action_fn:
                return episode, -1
            elif isinstance(self.player, PlayerRLAgent):
                self.player.update_q_table(state, action, -1, None)
            return

        # Dealer's turn
        while self.dealer.get_score() < 17:
            self.dealer.add_card(self.deck.draw_card())
            #print("Dealer: ", self.dealer)

        if self.dealer.get_score() > 21:
            #print("Dealer busts! Player wins.\n")
            if choose_action_fn:
                return episode, 1
            elif isinstance(self.player, PlayerRLAgent) and state is not None:
                self.player.update_q_table(state, action, 1, None)
            return

        if self.player.get_score() > self.dealer.get_score():
            #print("Player wins!\n")
            if choose_action_fn:
                return episode, 1
            elif isinstance(self.player, PlayerRLAgent) and state is not None:
                self.player.update_q_table(state, action, 1, None)
        elif self.player.get_score() < self.dealer.get_score():
            #print("Dealer wins!\n")
            if choose_action_fn:
                return episode, -1
            elif isinstance(self.player, PlayerRLAgent) and state is not None:
                self.player.update_q_table(state, action, -1, None)
            return
        else:
            #print("It's a tie!\n")
            return episode, 0

    def run_episode_mc(self, episode_count):
        self.game_reset()
        self.deck.shuffle()

        choose_action_fn = lambda state, ep_count, ep_len: self.player.choose_action(state, ep_count, ep_len)

        episode, reward = self.play(choose_action_fn, episode_count)
        self.player.update_q_table_mc(episode, reward)
        
    def run_episode_sarsa(self, episode_count):
        self.game_reset()
        self.deck.shuffle()
        self.player.add_card(self.deck.draw_card())
        self.player.add_card(self.deck.draw_card())
        self.dealer.add_card(self.deck.draw_card())

        print("Player: ", self.player)
        print("Dealer: ", self.dealer)
        
        player_score = self.player.get_score()

        # Player's turn
        while player_score < 12:
            self.player.add_card(self.deck.draw_card())
            print("Player [Automatic Draw]: ", self.player)
            player_score = self.player.get_score()

        state = (player_score, self.dealer.cards[0].value, self.player.usable_ace)
        
        if player_score == 21:  # Player starts with 21
            state = None
            action = None
        else: 
            action = self.player.choose_action(state, episode_count)

        while True:
            if state is None:  # Player starts with 21, no action needed
                break

            next_state, reward = self.step(action)
            
            if next_state is None:
                break

            next_action = self.player.choose_action(next_state, episode_count)
            self.player.update_q_table_sarsa(state, action, reward, next_state)

            state = next_state
            action = next_action

            if action == "s" or state is None:
                break
            
        # Dealer's turn
        while self.dealer.get_score() < 17:
            self.dealer.add_card(self.deck.draw_card())
            print("Dealer: ", self.dealer)

        dealer_score = self.dealer.get_score()
        player_score = self.player.get_score()
        if player_score > 21:
            #print("Player busts! Dealer wins.\n")
            reward = -1
        elif player_score == 21:
            if dealer_score == 21:
                #print("It's a draw!\n")
                reward = 0
            else:
                #print("Player wins!\n")
                reward = 1
        elif dealer_score > 21:
            #print("Dealer busts! Player wins.\n")
            reward = 1
        else:
            if player_score > dealer_score:
                #print("Player wins!\n")
                reward = 1
            elif player_score < dealer_score:
                #print("Dealer wins!\n")
                reward = -1
            else:
                #print("It's a tie!\n")
                reward = 0

        # Update Q-table for final game state
        if state is not None:
            self.player.update_q_table_sarsa(state, action, reward, None)

    def step(self, action):
        if action == "h":
            self.player.add_card(self.deck.draw_card())
            player_score = self.player.get_score()
            print("Player: ", self.player)
            if player_score > 21:
                next_state = None
                reward = -1
            else:
                next_state = (player_score, self.dealer.cards[0].value, self.player.usable_ace)
                reward = 0
        else:
            next_state = None
            reward = 0

        return next_state, reward

    def game_reset(self):
        self.player.reset()
        self.dealer.reset()
        self.deck.reset()

    def start(self, num_episodes=1000):
        for episode_count in range(num_episodes):
            print(f"Episode {episode_count}")
            if isinstance(self.player, MonteCarloOnPolicyControl):
                self.run_episode_mc(episode_count)
            elif isinstance(self.player, SARSAOnPolicyControl):
                self.run_episode_sarsa(episode_count)
            else:
                self.game_reset()
                self.play()

                
class MonteCarloOnPolicyControl(PlayerRLAgent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, exploring_starts=False, epsilon_config='1/k'):
        super().__init__(alpha, gamma, epsilon)
        self.exploring_starts = exploring_starts
        self.epsilon_config = epsilon_config
        self.returns_sum = {}
        self.returns_count = {}

    def choose_action(self, state, episode_count, episode):
        if self.exploring_starts and state[0] in range(12, 21) and len(episode) == 1:
            #print("Explore!")
            return random.choice(["h", "s"])
        else:
            #print("Exploit!!")
            epsilon = self.calculate_epsilon(episode_count)
            if random.random() < epsilon:
                return random.choice(["h", "s"])
            else:
                return self.get_best_action(state)

    def calculate_epsilon(self, episode_count):
        if self.epsilon_config == '1/k':
            return 1 / (episode_count + 1)
        elif self.epsilon_config == 'e^(-k/1000)':
            return math.exp(-episode_count / 1000)
        elif self.epsilon_config == 'e^(-k/10000)':
            return math.exp(-episode_count / 10000)
        else:
            raise ValueError("Invalid epsilon configuration")

    def update_q_table_mc(self, episode, reward):
        visited_states_actions = set()
        for state, action in episode:
            self.q_table_counts.setdefault(state, {}).setdefault(action, 0)
            self.q_table_counts[state][action] += 1
        
            if (state, action) not in visited_states_actions:
                visited_states_actions.add((state, action))
                if (state, action) not in self.returns_sum:
                    self.returns_sum[(state, action)] = 0.0
                    self.returns_count[(state, action)] = 0
                self.returns_sum[(state, action)] += reward
                self.returns_count[(state, action)] += 1
                self.q_table.setdefault(state, {"h": 0.0, "s": 0.0})
                self.q_table[state][action] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

class SARSAOnPolicyControl(PlayerRLAgent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_config='1/k'):
        super().__init__(alpha, gamma, epsilon)
        self.epsilon_config = epsilon_config

    def choose_action(self, state, episode_count):
        epsilon = self.calculate_epsilon(episode_count)
        if random.random() < epsilon:
            return random.choice(["h", "s"])  # Explore
        else:
            return self.get_best_action(state)  # Exploit

    def calculate_epsilon(self, episode_count):
        if self.epsilon_config == '1/k':
            return 1 / (episode_count + 1)
        elif self.epsilon_config == 'e^(-k/1000)':
            return math.exp(-episode_count / 1000)
        elif self.epsilon_config == 'e^(-k/10000)':
            return math.exp(-episode_count / 10000)
        else:
            raise ValueError("Invalid epsilon configuration")

    def update_q_table_sarsa(self, state, action, reward, next_state):
        if state is None:
            return  # Ignore None states
    
        self.q_table_counts.setdefault(state, {}).setdefault(action, 0)
        self.q_table_counts[state][action] += 1
        
        if next_state is None:  # Terminal state
            q_value = self.q_table.get(state, {"h": 0.0, "s": 0.0}).get(action, 0.0)
            new_value = q_value + self.alpha * (reward - q_value)
        else:
            current_value = self.q_table.get(state, {"h": 0.0, "s": 0.0}).get(action, 0.0)
            next_state_value = max(self.q_table.get(next_state, {"h": 0.0, "s": 0.0}).values())
            new_value = current_value + self.alpha * (reward + self.gamma * next_state_value - current_value)

        self.q_table.setdefault(state, {"h": 0.0, "s": 0.0})[action] = new_value

class QLearningOffPolicyControl(SARSAOnPolicyControl):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_config='1/k'):
        super().__init__(alpha, gamma, epsilon, epsilon_config)

    def update_q_table_sarsa(self, state, action, reward, next_state):
        if state is None:
            return  # Ignore None states

        # Q-Learning update rule
        next_state_best_action = self.get_best_action(next_state) if next_state is not None else None
        next_state_value = 0 if next_state is None else self.q_table.get(next_state, {"h": 0.0, "s": 0.0}).get(next_state_best_action, 0.0)
        current_value = self.q_table.get(state, {"h": 0.0, "s": 0.0}).get(action, 0.0)
        new_value = current_value + self.alpha * (reward + self.gamma * next_state_value - current_value)

        self.q_table.setdefault(state, {"h": 0.0, "s": 0.0})[action] = new_value
        
        self.q_table_counts.setdefault(state, {}).setdefault(action, 0)
        self.q_table_counts[state][action] += 1

# Create RL agents for player and create the dealer
player = QLearningOffPolicyControl()
dealer = Dealer()

deck = Deck()
print(f"Deck of {len(deck.cards)} cards created")

# Create Blackjack game with RL agents
game_with_rl = BlackJackWithRL(deck, player, dealer)
game_with_rl.start(1000)

#player.print_q_table()