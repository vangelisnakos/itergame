import numpy as np
import nashpy as nash

from src import utils

# Game Setup
cooperate_payoff = 3
cooperate_alone_payoff = 0
defect_payoff = 1
defect_alone_payoff = 5

A = np.array([
    [cooperate_payoff, cooperate_alone_payoff],
    [defect_alone_payoff, defect_payoff]
])

B = np.array([
    [cooperate_payoff, defect_alone_payoff],
    [cooperate_alone_payoff, defect_payoff]
])

game = nash.Game(A, B)

# Game Print
strategies = ["Cooperate", "Defect"]
cell_width = 18

utils.print_normal_form_game(
    A,
    B,
    choice_names=strategies,
    title="Prisoner's Dilemma"
)

utils.print_nash_equilibria(
    game,
    choice_names=strategies
)

cooperative_solution = utils.print_cooperative_outcome(
    A,
    B,
    choice_names=strategies,
)

# Iteration Start
n_actions = len(strategies)
alpha = np.ones(n_actions)

for t in range(10):
    opponent_action = np.random.randint(n_actions)

    belief = alpha / alpha.sum()

    expected_welfare = np.zeros(n_actions)

    for i in range(n_actions):
        for j in range(n_actions):
            expected_welfare[i] += belief[j] * (A[i, j] + B[i, j])

    my_action = int(np.argmax(belief))
    alpha[opponent_action] += alpha.sum()

    print(f"Round {t + 1}")
    print("  Belief:", dict(zip(strategies, belief.round(2))))
    print("  My action:", strategies[my_action])
    print("  Opponent:", strategies[opponent_action])
    print()

d = 1
