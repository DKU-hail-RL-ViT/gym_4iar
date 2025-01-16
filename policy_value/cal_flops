import argparse

import numpy as np
import torch

from collections import defaultdict
from fiar_env import Fiar, turn
from policy_value_network_copy import PolicyValueNet
from policy_value.mcts_copy import MCTSPlayer
from prev_codes.efficient_mcts_copy import EMCTSPlayer
from policy_value.file_utils import *

from thop import profile

parser = argparse.ArgumentParser()

""" tuning parameter """
parser.add_argument("--n_playout", type=int, default=400)  # compare with 2, 10, 50, 100, 400
parser.add_argument("--quantiles", type=int, default=3)  # compare with 3, 9, 27, 81
parser.add_argument('--epsilon', type=float, default=0.7)  # compare with 0.1, 0.4, 0.7

"""Efficient Search Hyperparameter"""
# EQRDQN (2, 5832), (10, 29160), (50, 145800), (100, 291600),(400, 1166400)
# EQRQAC (2, 5832), (10, 29160), (50, 145800), (100, 291600),(400, 1166400)

parser.add_argument('--effi_n_playout', type=int, default=400)
parser.add_argument('--search_resource', type=int, default=1166400)

""" RL model """
# parser.add_argument("--rl_model", type=str, default="DQN")  # action value ver
# parser.add_argument("--rl_model", type=str, default="QRDQN")  # action value ver
# parser.add_argument("--rl_model", type=str, default="AC")  # Actor critic state value ver
# parser.add_argument("--rl_model", type=str, default="QAC")  # Actor critic action value ver
# parser.add_argument("--rl_model", type=str, default="QRAC")   # Actor critic state value ver
parser.add_argument("--rl_model", type=str, default="QRQAC")  # Actor critic action value ver
# parser.add_argument("--rl_model", type=str, default="EQRDQN") # Efficient search + action value ver
# parser.add_argument("--rl_model", type=str, default="EQRQAC")  # Efficient search + Actor critic action value ver

""" MCTS parameter """
parser.add_argument("--buffer_size", type=int, default=10000)
parser.add_argument("--c_puct", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--self_play_sizes", type=int, default=100)
parser.add_argument("--training_iterations", type=int, default=100)
parser.add_argument("--temp", type=float, default=1.0)

""" Policy update parameter """
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learn_rate", type=float, default=1e-3)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--kl_targ", type=float, default=0.02)

""" Policy evaluate parameter """
parser.add_argument("--win_ratio", type=float, default=0.0)
parser.add_argument("--init_model", type=str, default=None)

args = parser.parse_args()

# make all args to variables
n_playout = args.n_playout
buffer_size = args.buffer_size
c_puct = args.c_puct
epochs = args.epochs
self_play_sizes = args.self_play_sizes
training_iterations = args.training_iterations
temp = args.temp
batch_size = args.batch_size
learn_rate = args.learn_rate
lr_mul = args.lr_mul
lr_multiplier = args.lr_multiplier
kl_targ = args.kl_targ
win_ratio = args.win_ratio
init_model = args.init_model
rl_model = args.rl_model
quantiles = args.quantiles
epsilon = args.epsilon
search_resource = args.search_resource
effi_n_playout = args.effi_n_playout


def policy_evaluate(env, current_mcts_player, old_mcts_player, single_forward_flops, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    training_mcts_player = current_mcts_player
    opponent_mcts_player = old_mcts_player
    win_cnt = defaultdict(int)

    for j in range(n_games):
        winner = start_play(env, training_mcts_player, opponent_mcts_player, single_forward_flops)
        win_cnt[winner] += 1
        print("{} / 30 ".format(j + 1))

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("---------- win: {}, tie:{}, lose: {} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio, training_mcts_player


def start_play(env, player1, player2, single_forward_flops):
    """start a game between two players"""
    obs, _ = env.reset()
    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0
    player_in_turn = players[current_player]
    net_wrapper.reset_count()

    while True:
        # synchronize the MCTS tree with the current state of the game
        move, flops = player_in_turn.get_action(env, game_iter=-1, temp=1e-3, return_prob=0)

        print(f"[RESULT] MCTS Used FLOPs: {flops}")
        obs, reward, terminated, info = env.step(move)

        raise ValueError("End")


def measure_mcts_flops(env, net_wrapper, single_forward_flops, player_in_turn):

    net_wrapper.reset_count()

    move, move_probs = player_in_turn.get_action(env, game_iter=-1, temp=1e-3, return_prob=0)

    total_call_count = net_wrapper.count
    total_flops = total_call_count * single_forward_flops

    return move, move_probs, total_flops


def get_single_forward_flops(model, input_shape=(1, 4, 6, 7)):
    dummy_input = torch.randn(input_shape)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params


class PolicyValueNetWithCounter:
    def __init__(self, core_model: PolicyValueNet):
        self.core_model = core_model
        self.count = 0

    def reset_count(self):
        self.count = 0

    def policy_value_fn(self, state):
        self.count += 1
        return self.core_model.policy_value_fn(state)


if __name__ == '__main__':

    wandb.init(entity="hails",
               project="gym_4iar_elo",
               name="calculate_flops",
               config=args.__dict__
               )

    env = Fiar()
    obs, _ = env.reset()

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    # init_model = f"Eval/{rl_model}_nmcts{n_playout}/train_100.pth"
    # init_model = f"Eval/{rl_model}_nmcts{effi_n_playout}/train_100.pth"
    init_model = f"Eval/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_100.pth"
    # init_model = f"Eval/{rl_model}_nmcts{n_playout}_quantiles{quantiles}_eps{epsilon}/train_100.pth"
    policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                      quantiles, model_file=init_model, rl_model=rl_model)

    input_size = (1, 5, 9, 4)
    single_forward_flops, params = get_single_forward_flops(policy_value_net.policy_value_net, input_size)
    print(f"[INFO] Single forward FLOPs: {single_forward_flops}, Params: {params}")

    net_wrapper = PolicyValueNetWithCounter(policy_value_net.policy_value_net)

    if rl_model in ["EQRDQN", "EQRQAC"]:
        curr_mcts_player = EMCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout,
                                       epsilon, search_resource, is_selfplay=0, rl_model=rl_model)
    elif rl_model in ["DQN", "QRDQN", "QRAC", "QRQAC"]:
        curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, quantiles,
                                      epsilon, is_selfplay=0, rl_model=rl_model)
    else:
        curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout,
                                      epsilon, is_selfplay=0, rl_model=rl_model)

    win_ratio, curr_mcts_player = policy_evaluate(env, curr_mcts_player, curr_mcts_player, single_forward_flops)
