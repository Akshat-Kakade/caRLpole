"""Mess with the cartpole while a controller tries to keep it up.

mouse:
  grab the pole tip and drag  -> shove the pole
  grab the cart and drag      -> pull the cart around
keys:
  1 PID   2 MPC   3 REINFORCE   4 me (arrow keys)   0 nothing
  space  random kick
  r      reset
  esc    quit

I don't end the episode at 12 degrees like real CartPole does, only when the pole
actually falls over or the cart hits a wall. so you can see who recovers from what.

usage (from the repo root):
       python -m interactive.app                    (starts with PID)
       python -m interactive.app --start reinforce
"""

import argparse

import numpy as np
import pygame

from baselines.mpc import FORCE, HALF_LEN, TAU, THETA_LIMIT, X_LIMIT, MPC, physics_step
from baselines.pid import PID

W, H = 900, 480
SCALE = 160                      # pixels per metre
TRACK_Y = 330
CART_W, CART_H = 70, 36
POLE_PX = int(2 * HALF_LEN * SCALE)

TIP_STRENGTH = 40.0              # how hard dragging the tip shoves it, tuned by feel
CART_STRENGTH = 30.0
DRAG_DAMPING = 4.0               # stops the cart drag from turning into a wild spring
FALLEN = np.pi / 2               # pole lying flat = actually fell over


def load_reinforce():
    """returns a fn obs -> action, or None if I haven't trained one yet."""
    import torch
    from agents.reinforce import MODEL_PATH, Policy

    if not MODEL_PATH.exists():
        return None

    policy = Policy()
    policy.load_state_dict(torch.load(MODEL_PATH))
    policy.eval()

    def act(obs):
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.distributions.Categorical(logits=logits).sample())

    return act


def to_px(x):
    return int(W / 2 + x * SCALE)


def tip_px(state):
    x, _, theta, _ = state
    return (to_px(x) + POLE_PX * np.sin(theta), TRACK_Y - CART_H / 2 - POLE_PX * np.cos(theta))


def apply_drag(state, grabbed, mouse_x):
    """my mouse messing with it. not proper physics, just a spring pulling
    the tip (or cart) towards the cursor by nudging its velocity."""
    state = state.copy()
    if grabbed == "tip":
        pull = (mouse_x - tip_px(state)[0]) / SCALE     # metres between tip and cursor
        state[3] += TIP_STRENGTH * pull * TAU
    elif grabbed == "cart":
        pull = (mouse_x - to_px(state[0])) / SCALE
        state[1] += (CART_STRENGTH * pull - DRAG_DAMPING * state[1]) * TAU
    return state


def random_start(rng):
    return rng.uniform(-0.05, 0.05, size=4)  # same as gymnasium's reset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="pid", choices=["pid", "mpc", "reinforce", "me", "none"])
    parser.add_argument("--max-frames", type=int, default=0, help="for testing, 0 = run forever")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("cartpole: grab the tip or the cart")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    rng = np.random.default_rng()

    pid, mpc, reinforce = PID(), MPC(), load_reinforce()
    mode = args.start
    state = random_start(rng)
    grabbed = None            # "tip", "cart" or None
    alive_time, best_time = 0.0, 0.0
    message, message_timer = "", 0
    frame = 0

    def reset(msg=""):
        nonlocal state, alive_time, message, message_timer
        state = random_start(rng)
        pid.reset()
        alive_time = 0.0
        message, message_timer = msg, 90

    running = True
    while running:
        # input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys = {pygame.K_1: "pid", pygame.K_2: "mpc", pygame.K_3: "reinforce",
                        pygame.K_4: "me", pygame.K_0: "none"}
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in keys:
                    mode = keys[event.key]
                    if mode == "reinforce" and reinforce is None:
                        mode = "none"
                        message, message_timer = "no trained model in models/, run agents.reinforce first", 120
                    pid.reset()
                elif event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_SPACE:
                    state[3] += rng.choice([-1, 1]) * rng.uniform(1.0, 2.0)  # whack the pole
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                tx, ty = tip_px(state)
                d_tip = np.hypot(mx - tx, my - ty)
                d_cart = np.hypot(mx - to_px(state[0]), my - TRACK_Y)
                if min(d_tip, d_cart) < 70:
                    grabbed = "tip" if d_tip < d_cart else "cart"
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                grabbed = None

        # controller picks a force
        obs = state.astype(np.float32)
        if mode == "pid":
            force = FORCE if pid.step(obs) == 1 else -FORCE
        elif mode == "mpc":
            force = FORCE if mpc.step(obs) == 1 else -FORCE
        elif mode == "reinforce":
            force = FORCE if reinforce(obs) == 1 else -FORCE
        elif mode == "me":
            pressed = pygame.key.get_pressed()
            force = FORCE * (pressed[pygame.K_RIGHT] - pressed[pygame.K_LEFT])
        else:
            force = 0.0

        # physics
        state = physics_step(state[:, None], np.array([force]))[:, 0]

        if grabbed:
            state = apply_drag(state, grabbed, pygame.mouse.get_pos()[0])

        # fell over or hit the wall?
        alive_time += TAU
        if abs(state[2]) > FALLEN:
            best_time = max(best_time, alive_time)
            reset("fell over")
        elif abs(state[0]) > X_LIMIT:
            best_time = max(best_time, alive_time)
            reset("hit the wall")

        # draw
        screen.fill((245, 245, 240))
        pygame.draw.line(screen, (60, 60, 60), (to_px(-X_LIMIT), TRACK_Y + CART_H // 2),
                         (to_px(X_LIMIT), TRACK_Y + CART_H // 2), 3)
        for wall in (-X_LIMIT, X_LIMIT):
            pygame.draw.line(screen, (200, 60, 60), (to_px(wall), TRACK_Y - 30),
                             (to_px(wall), TRACK_Y + CART_H // 2), 4)

        cx = to_px(state[0])
        cart_rect = pygame.Rect(cx - CART_W // 2, TRACK_Y - CART_H // 2, CART_W, CART_H)
        pygame.draw.rect(screen, (70, 90, 140) if grabbed != "cart" else (110, 140, 210), cart_rect, border_radius=4)

        # pole goes red once it's past where real CartPole would've ended the episode
        pole_color = (200, 140, 80) if abs(state[2]) < THETA_LIMIT else (210, 60, 60)
        base = (cx, TRACK_Y - CART_H // 2)
        tip = tip_px(state)
        pygame.draw.line(screen, pole_color, base, tip, 10)
        pygame.draw.circle(screen, (40, 40, 40), base, 5)
        pygame.draw.circle(screen, (110, 140, 210) if grabbed == "tip" else (90, 90, 90),
                           (int(tip[0]), int(tip[1])), 9)

        # little arrow for which way the controller is pushing
        if force != 0:
            d = 1 if force > 0 else -1
            ax = cx + d * (CART_W // 2 + 8)
            pygame.draw.polygon(screen, (90, 170, 90), [(ax, TRACK_Y - 8), (ax, TRACK_Y + 8), (ax + d * 16, TRACK_Y)])

        if grabbed:
            pygame.draw.line(screen, (150, 150, 150), tip if grabbed == "tip" else base,
                             pygame.mouse.get_pos(), 1)

        lines = [
            f"controller: {mode.upper()}   [1 PID  2 MPC  3 REINFORCE  4 me  0 none]",
            f"angle {np.degrees(state[2]):6.1f} deg   cart x {state[0]:5.2f}",
            f"up for {alive_time:5.1f}s   best {best_time:5.1f}s",
            "drag tip/cart with mouse | space kick | r reset | esc quit",
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, (30, 30, 30)), (12, 10 + 20 * i))
        if message_timer > 0:
            screen.blit(font.render(message, True, (200, 60, 60)), (W // 2 - 50, 110))
            message_timer -= 1

        pygame.display.flip()
        clock.tick(int(1 / TAU))   # 50 fps = real time, same timestep as CartPole

        frame += 1
        if args.max_frames and frame >= args.max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
