"""Load the trained REINFORCE policy and just watch it play, no training.

usage: python play.py            (3 episodes)
       python play.py -n 10      (10 episodes)
"""

import argparse

import torch

from reinforce import Policy, watch

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--episodes", type=int, default=3)
parser.add_argument("--weights", default="reinforce_policy.pt")
args = parser.parse_args()

# same network shape as in training, then drop the saved weights into it
policy = Policy()
policy.load_state_dict(torch.load(args.weights))
policy.eval()

watch(policy, episodes=args.episodes)
