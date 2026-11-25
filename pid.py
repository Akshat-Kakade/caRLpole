"""PID controller for CartPole, no learning at all, just hand-tuned gains.

Two loops added together:
  - angle loop (the actual PID): keep the pole upright
  - position loop (PD): gently pull the cart back to the middle so it doesn't drift off

CartPole only lets me push full left or full right, so I just take the sign
of the control signal (bang-bang control).

usage: python pid.py                 (100 episodes headless, prints scores)
       python pid.py --render -n 3   (watch it)
"""

import argparse

import gymnasium as gym
import numpy as np

# angle loop
KP = 1.0    # pole leaning right -> push right
KI = 0.1    # barely matters here tbh, nothing pushes the pole steadily one way
KD = 0.5    # react to how fast it's falling, not just where it is. without this it fails in ~40 steps

# position loop
KX = 0.1    # cart off to the right -> nudge right?? see the note in step()
KXD = 0.3

DT = 0.02   # CartPole's timestep, need it for the integral


class PID:
    def __init__(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, obs):
        x, x_dot, theta, theta_dot = obs
        self.integral += theta * DT

        u = KP * theta + KI * self.integral + KD * theta_dot

        # weird bit: to move the cart left you first have to push it right a bit, so the
        # pole tips left, and then the angle loop chases it left. so the position terms
        # have the SAME sign as the angle ones. flipping them makes it drift off the edge
        u += KX * x + KXD * x_dot

        return 1 if u > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    controller = PID()
    scores, max_drift = [], []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        controller.reset()
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
    print(f"PID over {args.episodes} eps | mean {np.mean(scores):.1f} | min {np.min(scores):.0f}"
          f" | avg max drift {np.mean(max_drift):.2f}")


if __name__ == "__main__":
    main()
