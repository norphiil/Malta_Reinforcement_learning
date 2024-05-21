import random
import math
import matplotlib.pyplot as plt
import os
import numpy as np

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
        self.cards: list[Card] = []
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
        self.usable_ace: bool = False

    def reset(self):
        self.cards = []

    def add_card(self, card: Card):
        self.cards.append(card)

    def get_score(self):
        score: int = 0
        num_aces: int = 0
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
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.9):
        super().__init__()
        self.alpha: float = alpha  # learning rate
        self.gamma: float = gamma  # discount factor
        self.epsilon: float = epsilon  # exploration rate
        self.q_table: dict = {}  # Q(s, a) table
        self.q_table_counts: dict = {}  # Q(s, a) table for counts

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["h", "s"])  # explore
        else:
            return self.get_best_action(state)  # exploit

    def get_best_action(self, state):
        # print("Choosing best action")
        if state not in self.q_table:
            # print(f"State {state} not in Q-table")
            self.q_table[state] = {"h": 0.0, "s": 0.0}
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        self.q_table_counts.setdefault(state, {}).setdefault(action, 0)
        self.q_table_counts[state][action] += 1

        if next_state is None:
            next_state_value = 0
        else:
            next_state_value = max(self.q_table.get(next_state, {"h": 0.0, "s": 0.0}).values())
            # print("\n*****Next state value = %5.2f*****\n" % (next_state_value))

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
        self.deck: Deck = deck
        self.player: Player = player
        self.dealer: Dealer = dealer
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.win_loss_table = []

    def play(self, choose_action_fn=None, episode_count=None):
        self.deck.shuffle()
        self.player.add_card(self.deck.draw_card())
        self.player.add_card(self.deck.draw_card())
        self.dealer.add_card(self.deck.draw_card())

        # print("Player: ", self.player)
        # print("Dealer: ", self.dealer)

        # Player's turn
        while self.player.get_score() < 12:
            self.player.add_card(self.deck.draw_card())
            # print("Player [Automatic Draw]: ", self.player)

        if self.player.get_score() == 21:
            state = None

        episode = []

        while self.player.get_score() < 21:
            state = (self.player.get_score(), self.dealer.cards[0].value, self.player.usable_ace)
            if choose_action_fn:
                action = choose_action_fn(state, episode_count, episode)
            elif isinstance(self.player, PlayerRLAgent):
                state = (self.player.get_score(), self.dealer.cards[0].value, self.player.usable_ace)
                action = self.player.choose_action(state)
            else:
                action = input("Do you want to hit or stand? (h/s): ")

            episode.append((state, action))

            if action == "h":
                self.player.add_card(self.deck.draw_card())
                # print("Player: ", self.player)
            elif action == "s":
                # print("Player has chosen to stand")
                break

        if self.player.get_score() > 21:
            # print("Player busts! Dealer wins.\n")
            if choose_action_fn:
                self.loss_count += 1
                return episode, -1
            elif isinstance(self.player, PlayerRLAgent):
                self.player.update_q_table(state, action, -1, None)
            return

        # Dealer's turn
        while self.dealer.get_score() < 17:
            self.dealer.add_card(self.deck.draw_card())
            # print("Dealer: ", self.dealer)

        if self.dealer.get_score() > 21:
            # print("Dealer busts! Player wins.\n")
            if choose_action_fn:
                self.win_count += 1
                return episode, 1
            elif isinstance(self.player, PlayerRLAgent) and state is not None:
                self.player.update_q_table(state, action, 1, None)
            return

        if self.player.get_score() > self.dealer.get_score():
            # print("Player wins!\n")
            if choose_action_fn:
                self.win_count += 1
                return episode, 1
            elif isinstance(self.player, PlayerRLAgent) and state is not None:
                self.player.update_q_table(state, action, 1, None)
        elif self.player.get_score() < self.dealer.get_score():
            # print("Dealer wins!\n")
            if choose_action_fn:
                self.loss_count += 1
                return episode, -1
            elif isinstance(self.player, PlayerRLAgent) and state is not None:
                self.player.update_q_table(state, action, -1, None)
            return
        else:
            # print("It's a tie!\n")
            self.draw_count += 1
            return episode, 0

    def run_episode_mc(self, episode_count):
        self.game_reset()
        self.deck.shuffle()

        self.player: MonteCarloOnPolicyControl
        choose_action_fn = lambda state, ep_count, ep_len: self.player.choose_action(state, ep_count, ep_len)

        episode, reward = self.play(choose_action_fn, episode_count)
        self.player.update_q_table_mc(episode, reward)

    def run_episode_sarsa(self, episode_count):
        self.game_reset()
        self.deck.shuffle()
        self.player.add_card(self.deck.draw_card())
        self.player.add_card(self.deck.draw_card())
        self.dealer.add_card(self.deck.draw_card())

        # print("Player: ", self.player)
        # print("Dealer: ", self.dealer)

        player_score = self.player.get_score()

        # Player's turn
        while player_score < 12:
            self.player.add_card(self.deck.draw_card())
            # print("Player [Automatic Draw]: ", self.player)
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
            self.player: SARSAOnPolicyControl
            self.player.update_q_table_sarsa(state, action, reward, next_state)

            state = next_state
            action = next_action

            if action == "s" or state is None:
                break

        # Dealer's turn
        while self.dealer.get_score() < 17:
            self.dealer.add_card(self.deck.draw_card())
            # print("Dealer: ", self.dealer)

        dealer_score = self.dealer.get_score()
        player_score = self.player.get_score()
        if player_score > 21:
            # print("Player busts! Dealer wins.\n")
            reward = -1
            self.loss_count += 1
        elif player_score == 21:
            if dealer_score == 21:
                # print("It's a draw!\n")
                reward = 0
                self.draw_count += 1
            else:
                # print("Player wins!\n")
                reward = 1
                self.win_count += 1
        elif dealer_score > 21:
            # print("Dealer busts! Player wins.\n")
            reward = 1
            self.win_count += 1
        else:
            if player_score > dealer_score:
                # print("Player wins!\n")
                reward = 1
                self.win_count += 1
            elif player_score < dealer_score:
                # print("Dealer wins!\n")
                reward = -1
                self.loss_count += 1
            else:
                # print("It's a tie!\n")
                reward = 0
                self.draw_count += 1

        # Update Q-table for final game state
        if state is not None:
            self.player: SARSAOnPolicyControl
            self.player.update_q_table_sarsa(state, action, reward, None)

    def step(self, action):
        if action == "h":
            self.player.add_card(self.deck.draw_card())
            player_score = self.player.get_score()
            # print("Player: ", self.player)
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
        self.win_loss_table = []
        for episode_count in range(num_episodes):
            # print(f"Episode {episode_count}")

            if ((episode_count+1) % 1000) == 0:
                self.win_loss_table.append((self.win_count, self.draw_count, self.loss_count))
                self.win_count, self.draw_count, self.loss_count = 0, 0, 0

            if isinstance(self.player, MonteCarloOnPolicyControl):
                self.run_episode_mc(episode_count)
            elif isinstance(self.player, SARSAOnPolicyControl):
                self.run_episode_sarsa(episode_count)
            else:
                self.game_reset()
                self.play()


class MonteCarloOnPolicyControl(PlayerRLAgent):
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, exploring_starts: bool = False, epsilon_config: str = '1/k'):
        super().__init__(alpha, gamma, epsilon)
        self.exploring_starts = exploring_starts
        self.epsilon_config = epsilon_config
        self.returns_sum = {}
        self.returns_count = {}

    def choose_action(self, state, episode_count, episode):
        if self.exploring_starts and state[0] in range(12, 21) and len(episode) == 1:
            # print("Explore!")
            return random.choice(["h", "s"])
        else:
            # print("Exploit!!")
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
        if self.epsilon_config == '0.1':
            return 0.1
        elif self.epsilon_config == '1/k':
            return 1 / (episode_count + 1)
        elif self.epsilon_config == 'e^(-k/1000)':
            return math.exp(-episode_count / 1000)
        elif self.epsilon_config == 'e^(-k/10000)':
            return math.exp(-episode_count / 10000)
        else:
            raise ValueError("Invalid epsilon configuration")

    def update_q_table_sarsa(self, state, action, reward, next_state):
        if state is None or state[0] == 21:
            return

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
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, epsilon_config: str = '1/k'):
        super().__init__(alpha, gamma, epsilon, epsilon_config)

    def update_q_table_sarsa(self, state, action, reward, next_state):
        if state is None or state[0] == 21:
            return

        # Q-Learning update rule
        next_state_best_action = self.get_best_action(next_state) if next_state is not None else None
        next_state_value = 0 if next_state is None else self.q_table.get(next_state, {"h": 0.0, "s": 0.0}).get(next_state_best_action, 0.0)
        current_value = self.q_table.get(state, {"h": 0.0, "s": 0.0}).get(action, 0.0)
        new_value = current_value + self.alpha * (reward + self.gamma * next_state_value - current_value)

        self.q_table.setdefault(state, {"h": 0.0, "s": 0.0})[action] = new_value

        self.q_table_counts.setdefault(state, {}).setdefault(action, 0)
        self.q_table_counts[state][action] += 1


def plot_results(records, algorithm_name, config_name, suffix):
    wins = [record[0] for record in records]
    draws = [record[1] for record in records]
    losses = [record[2] for record in records]

    # Create x-axis values representing each 1000 episodes
    episodes = list(range(1000, (len(records) + 1) * 1000, 1000))

    plt.figure(figsize=(26, 15))

    plt.plot(episodes, wins, label='Wins')
    plt.plot(episodes, draws, label='Draws')
    plt.plot(episodes, losses, label='Losses')

    plt.xlabel('Episodes')
    plt.ylabel('Count')
    plt.title(f'{algorithm_name} ({config_name}) - Results per 1000 Episodes')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    cwd = os.getcwd()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    file_name = os.path.join(cwd, "plots/")
    file_name = file_name + "line_" + suffix + ".png"
    plt.savefig(file_name)
    plt.close()


def plot_state_action_counts(q_table_counts, algorithm_name, config_name, suffix):
    flattened_counts = [(state, action, count) for state, actions in q_table_counts.items() for action, count in actions.items()]

    sorted_counts = sorted(flattened_counts, key=lambda x: x[2], reverse=True)

    states_actions_true = [f"{state[:-1]} ({action})" for state, action, _ in sorted_counts if True in state]
    counts_true = [count for state, _, count in sorted_counts if True in state]

    states_actions_false = [f"{state[:-1]} ({action})" for state, action, _ in sorted_counts if True not in state]
    counts_false = [count for state, _, count in sorted_counts if True not in state]

    # Plot chart for states with "True"
    plt.figure(figsize=(26, 15))
    plt.bar(states_actions_true, counts_true, color='skyblue')
    plt.xlabel('State-Action Pair')
    plt.ylabel('Count')
    plt.title(f'{algorithm_name} ({config_name}) - State-Action Pair Counts Holding an Ace')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.margins(x=0.001)

    # Save the plot to a file
    cwd = os.getcwd()
    file_name_true = os.path.join(cwd, "plots/") + "count_true_" + suffix + ".png"
    plt.savefig(file_name_true)
    plt.close()

    # Plot chart for states without "True"
    plt.figure(figsize=(26, 15))
    plt.bar(states_actions_false, counts_false, color='skyblue')
    plt.xlabel('State-Action Pair')
    plt.ylabel('Count')
    plt.title(f'{algorithm_name} ({config_name}) - State-Action Pair Counts Without an Ace')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.margins(x=0.001)

    # Save the plot to a file
    file_name_false = os.path.join(cwd, "plots/") + "count_false_" + suffix + ".png"
    plt.savefig(file_name_false)
    plt.close()


def build_strategy_table(q_table, suffix, title):
    action_map = {"h": "H", "s": "S"}

    strategy_table_with_ace = {}
    strategy_table_without_ace = {}

    for player_sum in range(20, 11, -1):
        strategy_table_with_ace[player_sum] = {}
        strategy_table_without_ace[player_sum] = {}

        for dealer_card in range(2, 11):
            state_with_ace = (player_sum, dealer_card, True)
            best_action_with_ace = q_table.get(state_with_ace, {"h": 0.0, "s": 0.0})
            # Assign H or S based on the best action
            strategy_table_with_ace[player_sum][dealer_card] = action_map[max(best_action_with_ace, key=best_action_with_ace.get)]

            state_without_ace = (player_sum, dealer_card, False)
            best_action_without_ace = q_table.get(state_without_ace, {"h": 0.0, "s": 0.0})
            # Assign H or S based on the best action
            strategy_table_without_ace[player_sum][dealer_card] = action_map[max(best_action_without_ace, key=best_action_without_ace.get)]

    # Convert strategy tables to matrices
    matrix_with_ace = np.array([[strategy_table_with_ace[player_sum][dealer_card] for dealer_card in range(2, 11)] for player_sum in range(20, 11, -1)])
    matrix_without_ace = np.array([[strategy_table_without_ace[player_sum][dealer_card] for dealer_card in range(2, 11)] for player_sum in range(20, 11, -1)])

    plot_and_save_strategy_table(matrix_with_ace, f"strategy_table_with_ace_{suffix}", f"{title} - Available Ace")
    plot_and_save_strategy_table(matrix_without_ace, f"strategy_table_no_ace_{suffix}", f"{title} - No available Ace")


def plot_and_save_strategy_table(matrix, filename, title):
    import matplotlib.colors as mcolors
    cmap_light = mcolors.ListedColormap(['lightblue', 'lightcoral'])

    # Create heatmap
    plt.figure(figsize=(10, 6))

    # Convert strings to numerical data
    matrix_numeric = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 'H':
                matrix_numeric[i, j] = 0.5
                plt.text(j, i, 'H', ha='center', va='center', color='black')  # Annotation for 'H'
            elif matrix[i, j] == 'S':
                matrix_numeric[i, j] = 0.0
                plt.text(j, i, 'S', ha='center', va='center', color='black')  # Annotation for 'S'

    plt.imshow(matrix_numeric, cmap=cmap_light, interpolation='nearest')

    plt.xlabel("Dealer's Card")
    plt.ylabel("Player's Sum")
    plt.title(title)
    plt.xticks(np.arange(9), np.arange(2, 11))
    plt.yticks(np.arange(9), np.arange(20, 11, -1))

    # Save plot to file
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "plots/") + filename + ".png"
    plt.savefig(file_path)
    plt.close()


def calculate_mean_win_loss(win_loss_table):
    last_records = win_loss_table[-10:]
    wins = sum(record[0] for record in last_records) / len(last_records)
    draws = sum(record[1] for record in last_records) / len(last_records)
    losses = sum(record[2] for record in last_records) / len(last_records)
    return wins, draws, losses


def calculate_dealer_advantage(win_loss_table):
    last_records = win_loss_table[-10:]
    wins = sum(record[0] for record in last_records)
    losses = sum(record[2] for record in last_records)
    return (losses - wins) / (losses + wins)


dealer_advantages = []


def run_algorithm_configurations(num_episodes=100000):
    configurations = [
        {"algorithm": "Monte Carlo", "exploring_starts": True, "epsilon_config": "1/k", "file_suffix": "mc_t_1k"},
        {"algorithm": "Monte Carlo", "exploring_starts": False, "epsilon_config": "1/k", "file_suffix": "mc_f_1k"},
        {"algorithm": "Monte Carlo", "exploring_starts": False, "epsilon_config": "e^(-k/1000)", "file_suffix": "mc_f_-k1000"},
        {"algorithm": "Monte Carlo", "exploring_starts": False, "epsilon_config": "e^(-k/10000)", "file_suffix": "mc_f_-k10000"},
        {"algorithm": "SARSA", "epsilon_config": "0.1", "file_suffix": "sarsa_e"},
        {"algorithm": "SARSA", "epsilon_config": "1/k", "file_suffix": "sarsa_1k"},
        {"algorithm": "SARSA", "epsilon_config": "e^(-k/1000)", "file_suffix": "sarsa_-k1000"},
        {"algorithm": "SARSA", "epsilon_config": "e^(-k/10000)", "file_suffix": "sarsa_-k10000"},
        {"algorithm": "QLearning", "epsilon_config": "0.1" ,"file_suffix": "qlearn_e"},
        {"algorithm": "QLearning", "epsilon_config": "1/k", "file_suffix": "qlearn_1k"},
        {"algorithm": "QLearning", "epsilon_config": "e^(-k/1000)", "file_suffix": "qlearn_-k1000"},
        {"algorithm": "QLearning", "epsilon_config": "e^(-k/10000)", "file_suffix": "qlearn_-k10000"}
    ]

    total_unique_sa_pairs = {}

    for config in configurations:
        print(f"Running {config['algorithm']} with epsilon_config={config['epsilon_config']}")
        player = None
        if config['algorithm'] == "Monte Carlo":
            player = MonteCarloOnPolicyControl(exploring_starts=config['exploring_starts'], epsilon_config=config['epsilon_config'])
        elif config['algorithm'] == "SARSA":
            player = SARSAOnPolicyControl(epsilon_config=config['epsilon_config'])
        elif config['algorithm'] == "QLearning":
            player = QLearningOffPolicyControl(epsilon_config=config['epsilon_config'])

        dealer = Dealer()
        deck = Deck()
        game_with_rl = BlackJackWithRL(deck, player, dealer)
        game_with_rl.start(num_episodes)

        total_explored = sum(len(v) for v in player.q_table_counts.values())
        total_unique_sa_pairs.setdefault(config["algorithm"], []).append(total_explored)

        plot_results(game_with_rl.win_loss_table, config['algorithm'], config['epsilon_config'], config["file_suffix"])
        plot_state_action_counts(player.q_table_counts, config['algorithm'], config['epsilon_config'], config["file_suffix"])
        build_strategy_table(player.q_table, config["file_suffix"], f"{config['algorithm']} {config['epsilon_config']}")

        dealer_advantage = calculate_dealer_advantage(game_with_rl.win_loss_table)
        dealer_advantages.append(dealer_advantage)
    # Plot histograms for each algorithm
    for algorithm, total_explored_list in total_unique_sa_pairs.items():
        plt.figure()  # Create a new figure for each algorithm
        plt.bar(range(len(total_explored_list)), total_explored_list)
        plt.xlabel('Configuration')
        plt.ylabel('Total Unique State-Action Pairs Explored')
        plt.title(f'Total Unique State-Action Pairs Explored for {algorithm}')
        plt.xticks(range(len(total_explored_list)), [f'Config {i+1}' for i in range(len(total_explored_list))])

        # Save the plot to a file
        cwd = os.getcwd()
        file_name = os.path.join(cwd, "plots/") + "exploration" + algorithm + ".png"
        plt.savefig(file_name)
        plt.close()

    # Plot dealer advantages as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(dealer_advantages)), dealer_advantages)
    plt.xlabel('Algorithm Configuration')
    plt.ylabel('Dealer Advantage')
    plt.title('Dealer Advantage of Different Algorithm Configurations')
    plt.xticks(range(len(configurations)), [f"{config['algorithm']}-{config['epsilon_config']}" for config in configurations], rotation=90)
    plt.grid(axis='y')  # Add gridlines only on the y-axis
    plt.tight_layout()
    file_name = os.path.join(cwd, "plots/") + "dealer_advantage.png"
    plt.savefig(file_name)
    plt.close()


run_algorithm_configurations(100000)
