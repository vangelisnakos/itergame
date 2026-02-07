import numpy as np


def print_normal_form_game(*players, choice_names=None, cell_width=20, title=None):
    """
    Pretty-print an n-player normal-form game with identical action sets.

    Parameters
    ----------
    *players : np.ndarray
        One payoff matrix per player. All must have the same shape.
    choice_names : list[str], optional
        Action names (shared by all players).
    cell_width : int, optional
        Width of each payoff cell.
    title : str, optional
        Optional title printed above the table.
    """
    if len(players) < 2:
        raise ValueError("At least two players are required")

    shapes = [p.shape for p in players]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All payoff matrices must have the same shape")

    if players[0].ndim != 2:
        raise ValueError("Only 2-action-dimension games are supported")

    n_players = len(players)
    n_actions_row, n_actions_col = players[0].shape

    if n_actions_row != n_actions_col:
        raise ValueError("Only square action spaces are supported")

    n_actions = n_actions_row

    if choice_names is None:
        choice_names = [f"Action {i+1}" for i in range(n_actions)]
    else:
        if len(choice_names) != n_actions:
            raise ValueError("Length of choice_names must match number of actions")

    row_label_width = max(len(name) for name in choice_names) + 2

    # ---- Title ----
    if title is not None:
        print(f"\n{title}")

    # ---- Header ----
    header = f"{'':{row_label_width}}"
    header += "".join(f"{name:^{cell_width}}" for name in choice_names)
    print(header)
    print("-" * len(header))

    # ---- Rows ----
    for i, row_name in enumerate(choice_names):
        row = f"{row_name:{row_label_width}}"
        for j in range(n_actions):
            payoff = tuple(int(p[i, j]) for p in players)
            row += f"{str(payoff):^{cell_width}}"
        print(row)

def print_nash_equilibria(game, choice_names=None, precision=2, title="Nash Equilibria"):
    """
    Pretty-print Nash equilibria for a 2-player NashPy game.

    Parameters
    ----------
    game : nashpy.Game
        A NashPy game instance.
    choice_names : list[str], optional
        Action names shared by both players.
    precision : int, optional
        Decimal precision for strategy probabilities.
    title : str, optional
        Title printed above equilibria.
    """
    equilibria = list(game.support_enumeration())

    if not equilibria:
        print("\nNo Nash equilibria found.")
        return

    n_actions = len(equilibria[0][0])

    if choice_names is None:
        choice_names = [f"Action {i+1}" for i in range(n_actions)]
    else:
        if len(choice_names) != n_actions:
            raise ValueError("Length of choice_names must match number of actions")

    print(f"\n{title}:\n")

    for k, eq in enumerate(equilibria, start=1):
        print(f"Equilibrium {k}")

        for player_idx, strategy in enumerate(eq, start=1):
            probs = {
                name: round(float(p), precision)
                for name, p in zip(choice_names, strategy)
            }
            print(f"  Player {player_idx}: {probs}")

        print()


def print_cooperative_outcome(*players, choice_names=None, title="Cooperative Optimum") -> np.ndarray:
    """
    Print the joint action(s) that maximize total payoff
    (i.e. cooperative / social-welfare optimum).

    Parameters
    ----------
    *players : np.ndarray
        One payoff matrix per player (same shape).
    choice_names : list[str], optional
        Action names shared by all players.
    title : str, optional
        Title printed above the result.
    """
    if len(players) < 2:
        raise ValueError("At least two players are required")

    shapes = [p.shape for p in players]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All payoff matrices must have the same shape")

    if players[0].ndim != 2:
        raise ValueError("Only 2D normal-form games are supported")

    n_actions = players[0].shape[0]

    if choice_names is None:
        choice_names = [f"Action {i+1}" for i in range(n_actions)]
    else:
        if len(choice_names) != n_actions:
            raise ValueError("Length of choice_names must match number of actions")

    # ---- Compute social welfare ----
    total_payoff = sum(players)
    max_value = np.max(total_payoff)

    optimal_indices = np.argwhere(total_payoff == max_value)

    print(f"\n{title}")
    print(f"(Max total payoff = {int(max_value)})\n")

    for idx in optimal_indices:
        i, j = idx
        actions = (choice_names[i], choice_names[j])
        payoffs = tuple(int(p[i, j]) for p in players)

        print(f"Actions: {actions}")
        print(f"Payoffs: {payoffs}\n")

    return optimal_indices
