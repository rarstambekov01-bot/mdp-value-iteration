"""
Практикалық жұмыс № 5
Марков шешім қабылдау процестері (MDP) және мәндер итерациясы (Value Iteration)

Жеткізу роботы:
    A  B  C  G(+10)
    D  #  E  X(-10)
    S  F  H  I
S - старт, G - мақсат, X - қауіпті ұяшық, # - қабырға.
"""

# ---------- 1. Әлемнің сипаттамасы ----------
STATES = ["A", "B", "C", "D", "E", "S", "F", "H", "I"]   # G және X - терминалдық
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# ұяшық координаттары: (қатар, баған)
POS = {"A": (0, 0), "B": (0, 1), "C": (0, 2), "G": (0, 3),
       "D": (1, 0),              "E": (1, 2), "X": (1, 3),
       "S": (2, 0), "F": (2, 1), "H": (2, 2), "I": (2, 3)}
NAME = {v: k for k, v in POS.items()}                    # координата -> ұяшық аты

MOVE = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
# таңдалған бағытқа қатысты (сол жақ, оң жақ) ауытқу
SIDES = {"UP": ("LEFT", "RIGHT"), "DOWN": ("RIGHT", "LEFT"),
         "LEFT": ("DOWN", "UP"), "RIGHT": ("UP", "DOWN")}

ARROWS = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}


def go(s, d):
    """Бір қадам. Қабырғаға немесе шетке тірелсе - орнында қалады."""
    r, c = POS[s]
    dr, dc = MOVE[d]
    return NAME.get((r + dr, c + dc), s)


def make_reward(goal=10, danger=-10, step=-0.1):
    def reward(s2):
        return goal if s2 == "G" else danger if s2 == "X" else step
    return reward


def make_transitions(p_main=0.8, reward=None):
    """transitions(s, a) -> [(ықтималдық, келесі күй, марапат), ...]"""
    p_side = (1 - p_main) / 2
    reward = reward or make_reward()

    def transitions(s, a):
        left, right = SIDES[a]
        result = []
        for p, d in [(p_main, a), (p_side, left), (p_side, right)]:
            s2 = go(s, d)
            result.append((p, s2, reward(s2)))
        return result
    return transitions


# ---------- 2. Мәндер итерациясы ----------
def value_iteration(gamma=0.9, theta=0.001, p_main=0.8,
                    goal=10, danger=-10, step=-0.1, verbose=False):
    transitions = make_transitions(p_main, make_reward(goal, danger, step))
    V = {s: 0.0 for s in STATES}                          # V0 = 0

    def q_value(s, a):
        # Q(s,a) = sum P * (R + gamma * V(s'))   (G, X үшін V = 0)
        return sum(p * (r + gamma * V.get(s2, 0.0))
                   for p, s2, r in transitions(s, a))

    iterations = 0
    while True:
        iterations += 1
        delta = 0.0
        new_V = V.copy()                                  # тек ЕСКІ V қолданамыз
        for s in STATES:
            new_V[s] = max(q_value(s, a) for a in ACTIONS)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if verbose and iterations <= 2:
            print(f"V{iterations}:", {s: round(v, 4) for s, v in V.items()})
        if delta < theta:
            break

    policy = {s: max(ACTIONS, key=lambda a: q_value(s, a)) for s in STATES}
    return V, policy, iterations


def show(V, policy, iterations):
    print(f"Итерация саны: {iterations}")
    for s in STATES:
        print(f"{s}  V = {V[s]:6.2f}   {policy[s]:5s} {ARROWS[policy[s]]}")


def route(policy, start="S", max_steps=20):
    """Ең ықтимал маршрут (80% бағытпен жүргенде)."""
    path, s = [start], start
    while s not in ("G", "X") and len(path) < max_steps:
        s = go(s, policy[s])
        path.append(s)
    return " -> ".join(path)


if __name__ == "__main__":
    print("=== Негізгі есеп (gamma=0.9) ===")
    V, pi, it = value_iteration(verbose=True)
    show(V, pi, it)
    print("Маршрут:", route(pi))

    experiments = [
        ("gamma: 0.9 -> 0.2",            dict(gamma=0.2)),
        ("Негізгі қозғалыс: 0.8 -> 0.6", dict(p_main=0.6)),
        ("Қауіпті күй: -10 -> -30",      dict(danger=-30)),
        ("Қадам марапаты: -0.1 -> -1",   dict(step=-1)),
    ]
    base_policy = pi
    for title, params in experiments:
        print(f"\n=== Эксперимент: {title} ===")
        V2, pi2, it2 = value_iteration(**params)
        show(V2, pi2, it2)
        changed = [f"{s}: {ARROWS[base_policy[s]]} -> {ARROWS[pi2[s]]}"
                   for s in STATES if base_policy[s] != pi2[s]]
        print("Саясат өзгерісі:", ", ".join(changed) if changed else "өзгермеді")
