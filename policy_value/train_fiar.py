import numpy as np
import random
import os
import wandb
import datetime

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize, carculate_area
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer

# from policy_value.policy_value_mcts_pure import RandomAction
import argparse


# make argparser
parser = argparse.ArgumentParser()
""" tuning parameter """
parser.add_argument("--n_playout", type=int, default=100)
parser.add_argument("--buffer_size", type=int, default=10000)
""" MCTS parameter """
parser.add_argument("--c_puct", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--self_play_sizes", type=int, default=10)      # temporary 8 , default 100
parser.add_argument("--training_iterations", type=int, default=100)
parser.add_argument("--temp", type=float, default=0.1)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
""" Policy update parameter """
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learn_rate", type=float, default=2e-4)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--kl_targ", type=float, default=0.02)
""" Policy evaluate parameter """
parser.add_argument("--win_ratio", type=float, default=0.0)
parser.add_argument("--init_model", type=str, default=None)

""" RL name """
parser.add_argument("--rl_model", type=str, default="AC")
# parser.add_argument("--rl_model", type=str, default="DQN")
# parser.add_argument("--rl_model", type=str, default="QRDQN")

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


def policy_value_fn(board):  # board.shape = (9,4)
    # return uniform probabilities and 0 score for pure MCTS
    availables = [i for i in range(36) if not np.any(board[3][i // 4][i % 4] == 1)]
    action_probs = np.ones(len(availables)) / len(availables)
    return zip(availables, action_probs), 0


def get_equi_data(env, play_data):
    """augment the data set by flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_prob, winner in play_data:
        # flip horizontally
        equi_state = np.array([np.fliplr(s) for s in state])
        equi_mcts_prob = np.fliplr(mcts_prob.reshape(env.state_.shape[1], env.state_.shape[2]))
        extend_data.append((equi_state,
                            np.flipud(equi_mcts_prob).flatten(),
                            winner))
    return extend_data


def collect_selfplay_data(mcts_player, n_games=100):
    # self-play 100 games and save in data_buffer(queue)
    # in data_buffer store all steps of self-play so it should be large enough
    data_buffer = deque(maxlen=36*n_games*20) # temporary 36*n_games*20 , default 36*n_games
    for self_play_i in range(n_games):
        rewards, play_data = self_play(env, mcts_player, temp, self_play_i)
        play_data = list(play_data)[:]
        # augment the data
        play_data = get_equi_data(env, play_data)
        data_buffer.extend(play_data)

    return data_buffer


def self_play(env, mcts_player, temp=1e-3, self_play_i=0):
    states, mcts_probs, current_player = [], [], []
    obs, _ = env.reset()

    player_0 = turn(obs)
    player_1 = 1 - player_0

    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[player_0] + obs[player_1]

    while True:
        while True:
            action = None
            move_probs = None
            if obs[3].sum() == 36:
                print('self_play_draw')
            else:
                move, move_probs = mcts_player.get_action(env, temp, return_prob=1)
                action = move
            action2d = action2d_ize(action)

            if obs[3, action2d[0], action2d[1]] == 0.0:
                break

        # store the data
        states.append(obs)
        mcts_probs.append(move_probs)
        current_player.append(turn(obs))

        obs, reward, terminated, info = env.step(action)

        player_0 = turn(obs)
        player_1 = 1 - player_0

        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])
        obs_post[3] = obs[player_0] + obs[player_1]

        end, winners = env.self_play_winner()

        if len(current_player) % 2 == 1:
            winners = 1
        else:
            winners = -0.5

        if end:

            # print(winners, "\t 끝났을 때 이긴사람")
            if obs[3].sum() == 36:
                print('self_play_draw')
            obs, _ = env.reset()

            # reset MCTS root node
            mcts_player.reset_player()

            print("batch i:{}, episode_len:{}".format(
                self_play_i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))


            if winners != -1:  # non draw
                if winners == -0.5:  # when win white player adjust to 0
                    winners = 0

                # print(winners, "winner") # if 0 백이 이김, if 1 흑이 이김
                print(current_player, "current_players")

                # if winner is current player, winner_z = 1
                winners_z[np.array(current_player) == 1 - winners] = 1.0
                winners_z[np.array(current_player) != 1 - winners] = -1.0


            # print(winners_z, "\n final update array \n")

            return winners, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul, policy_value_net,  data_buffers=None):
    k, kl, loss, entropy = 0, 0, 0, 0
    lr_multiplier = lr_mul
    update_data_buffer = [data for buffer in data_buffers for data in buffer]

    """update the policy-value net"""
    mini_batch = random.sample(update_data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)

    for k in range(epochs):
        loss, entropy = policy_value_net.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch,
            learn_rate * lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1)
                     )
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break

    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    explained_var_old = (1 -
                         np.var(np.array(winner_batch) - old_v.flatten()) /
                         (np.var(np.array(winner_batch)) + 1e-10))
    explained_var_new = (1 -
                         np.var(np.array(winner_batch) - new_v.flatten()) /
                         (np.var(np.array(winner_batch)) + 1e-10))

    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{},"
           "explained_var_old:{:.3f},"
           "explained_var_new:{:.3f}"
           ).format(kl,
                    lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new))
    return loss, entropy, lr_multiplier, policy_value_net


def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    training_mcts_player = current_mcts_player  # training Agent
    opponent_mcts_player = old_mcts_player
    # leaf_mcts_player = MCTS_leaf(policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    # random_action_player = RandomAction() # random actions Agent
    win_cnt = defaultdict(int)

    for j in range(n_games):

        # reset for each game
        winner = start_play(env, training_mcts_player, opponent_mcts_player)
        if winner == -0.5:
            winner = 0
        win_cnt[winner] += 1
        print("{} / 30 ".format(j + 1))

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio, training_mcts_player


def start_play(env, player1, player2):
    """start a game between two players"""
    obs, _ = env.reset()

    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0
    move = None
    player_in_turn = players[current_player]

    while True:
        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env, temp=0.1, return_prob=0)
        obs, reward, terminated, info = env.step(move)
        assert env.state_[3][action2d_ize(move)] == 1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

        else:
            # print(env)
            wandb.log({"Reward": reward[1]})
            obs, _ = env.reset()
            return winner


if __name__ == '__main__':

    # wandb intialize
    wandb.init(mode="online",
               entity="hails",
               project="gym_4iar",
               name="FIAR-" + rl_model + "-MCTS" + str(n_playout) +
                    "-Date" + str(datetime.datetime.now()),
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

    if init_model:
        # start training from an initial policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1],
                                          env.state().shape[2],
                                          model_file=init_model, rl_model=rl_model)
    else:
        # start training from a new policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1],
                                          env.state().shape[2], rl_model=rl_model)

    # policy_value_net_old = copy.deepcopy(policy_value_net)
    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=1)

    data_buffer_training_iters = deque(maxlen=20)

    try:
        for i in range(training_iterations):
            data_buffer_each = collect_selfplay_data(curr_mcts_player, self_play_sizes)     # collect self-play data
            data_buffer_training_iters.append(data_buffer_each)

            loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=lr_multiplier,
                                                                           policy_value_net=policy_value_net,
                                                                           data_buffers=data_buffer_training_iters)
            wandb.log({"loss": loss, "entropy": entropy})

            if i==0:
                policy_evaluate(env, curr_mcts_player, curr_mcts_player)
                model_file = f"nmcts{n_playout}/train_{i + 1:03d}.pth"
                policy_value_net.save_model(model_file)
                print("model saved")
            else:
                existing_files = [int(file.split('_')[-1].split('.')[0])
                                  for file in os.listdir(f"nmcts{n_playout}")
                                  if file.startswith('train_')]

                old_i = max(existing_files)
                best_old_model = f"nmcts{n_playout}/train_{old_i:03d}.pth"
                policy_value_net_old = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
                                                      best_old_model, rl_model=rl_model)
                # when evaluating, non use dirichlet noise
                curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=0)
                old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout, is_selfplay=0)
                win_ratio, eval_mcts_player = policy_evaluate(env, curr_mcts_player, old_mcts_player)

                print("\t win rate : ", round(win_ratio * 100, 3), "%")
                wandb.log({"Win Rate Evaluation": round(win_ratio * 100, 3)})

                if win_ratio > 0.6:  # if better, update the policy
                    old_mcts_player = eval_mcts_player

                    model_file = f"nmcts{n_playout}/train_{i + 1:03d}.pth"
                    policy_value_net.save_model(model_file)
                    print("\t New best policy!!!")

                else:
                    print("\t low win-rate") # if worse it just reject and does not go back


    except KeyboardInterrupt:
        print('\n\rquit')
