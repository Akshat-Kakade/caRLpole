"""Model predictive control (MPC) for CartPole, no learning either.

Every single step:
  1. take the current state
  2. use the physics equations (copied from gymnasium's cartpole.py) to simulate
     every possible sequence of the next HORIZON actions. only 2 actions so that's
     2^10 = 1024 sequences, numpy does them all at once
  3. score each sequence (pole angle off vertical = bad, falling over = very bad)
  4. do the FIRST action of the best sequence, throw the rest away, replan next step

The big difference from RL: MPC is handed the exact model of the world.
RL has to figure everything out from just rewards.

usage (from the repo root):
       python -m baselines.mpc                 (100 episodes headless, prints scores)
       python -m baselines.mpc --render -n 3   (watch it)
"""

import argparse
import itertools

import gymnasium as gym
import numpy as np

HORIZON = 10    # how many steps ahead to look (10 * 0.02s = 0.2 seconds)
W_THETA = 1.0   # cost on pole angle
W_X = 0.001     # cost on cart position. tiny on purpose, see below
W_VEL = 0.01    # small cost on both velocities so it doesn't thrash around
FAIL_COST = 1e6

# physics constants, same as gymnasium's CartPoleEnv
GRAVITY, MASS_CART, MASS_POLE, HALF_LEN, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
TOTAL_MASS = MASS_CART + MASS_POLE
POLEMASS_LEN = MASS_POLE * HALF_LEN
X_LIMIT, THETA_LIMIT = 2.4, 12 * 2 * np.pi / 360


def model_step(state, action):
    """CartPole physics, works on a whole batch of states at once.
    state is shape (4, n), action is shape (n,) of 0s and 1s."""
    return physics_step(state, np.where(action == 1, FORCE, -FORCE))


def physics_step(state, force):
    """same thing but takes the force directly (interactive.py uses this so force can be 0)."""
    x, x_dot, theta, theta_dot = state
    cos, sin = np.cos(theta), np.sin(theta)

    temp = (force + POLEMASS_LEN * theta_dot**2 * sin) / TOTAL_MASS
    theta_acc = (GRAVITY * sin - cos * temp) / (
        HALF_LEN * (4.0 / 3.0 - MASS_POLE * cos**2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LEN * theta_acc * cos / TOTAL_MASS

    return np.stack([
        x + TAU * x_dot,
        x_dot + TAU * x_acc,
        theta + TAU * theta_dot,
        theta_dot + TAU * theta_acc,
    ])


class MPC:
    def __init__(self, horizon=HORIZON):
        # every possible action sequence, shape (2^horizon, horizon)
        self.sequences = np.array(list(itertools.product([0, 1], repeat=horizon)))

    def step(self, obs):
        n, horizon = self.sequences.shape
        # copy the current state once per candidate sequence
        state = np.repeat(np.asarray(obs, dtype=np.float64)[:, None], n, axis=1)
        cost = np.zeros(n)
        failed = np.zeros(n, dtype=bool)

        for t in range(horizon):
            state = model_step(state, self.sequences[:, t])
            x, x_dot, theta, theta_dot = state
            failed |= (np.abs(x) > X_LIMIT) | (np.abs(theta) > THETA_LIMIT)
            cost += W_THETA * theta**2 + W_X * x**2 + W_VEL * (x_dot**2 + theta_dot**2)

        cost += FAIL_COST * failed
        best = np.argmin(cost)
        return int(self.sequences[best, 0])


# note on W_X: I tried bumping it to 0.1 to keep the cart centred and it got WORSE (~340 avg),
# the cart ran off the edge. 0.2s of lookahead is too short to see that you need to push
# away from the centre first to get back to it (same thing as the comment in pid.py).
# it just greedily shoves towards the middle and loses the pole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    controller = MPC()
    scores, max_drift = [], []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total, done, xs = 0.0, False, []
        while not done:
            obs, reward, terminated, truncated, _ = env.step(controller.step(obs))
            total += reward
            xs.append(abs(obs[0]))
            done = terminated or truncated
        scores.append(total)
        max_drift.append(max(xs))
        if args.render or args.episodes <= 10:
            print(f"ep {ep}: reward {total:.0f}, furthest from centre {max(xs):.2f}")

    env.close()
    print(f"MPC over {args.episodes} eps | mean {np.mean(scores):.1f} | min {np.min(scores):.0f}"
          f" | avg max drift {np.mean(max_drift):.2f}")


if __name__ == "__main__":
    main()
