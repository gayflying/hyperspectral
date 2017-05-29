import argparse
import os

cwd = os.getcwd()
parser = argparse.ArgumentParser()

# env parameters
parser.add_argument('--game', type=str, default='Breakout-v0',
                    help='Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
parser.add_argument('--use_gpu', type=bool, default=False)

# model parameters
parser.add_argument('--use_lstm', type=bool, default=False,
                    help='wether use lstm network')
parser.add_argument('--frames', type=int, default=4,
                    help='the num of frames that a state contains')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--entropy_beta', type=float, default=0.01)

# train parameters
parser.add_argument('--local_t_max', type=int, default=32,
                    help='async interval of a single thread. In fact it is the same as batch size')
parser.add_argument('--max_time_step', type=int, default=10*10**7)
parser.add_argument('--initial_alpha_low', type=float, default=1e-4)
parser.add_argument('--initial_alpha_high', type=float, default=1e-2)
parser.add_argument('--initial_alpha_log_rate', type=float, default=0.4226)
parser.add_argument('--thread_num', type=int, default=8)

# log parameters
parser.add_argument('--log_interval', type=int, default=100,
                    help='log interval')
parser.add_argument('--performance_log_interval', type=int, default=1000,
                    help='performance log interval')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--log_file', type=str, default='tmp/a3c_log')
parser.add_argument('--use_chechpoint', type=bool, default=False)

# gradient applier parameters
parser.add_argument('--rmsp_alpha', type=float, default=0.99)
parser.add_argument('--rmsp_epsilon', type=float, default=0.1)
parser.add_argument('--grad_norm_clip', type=float, default=40.0)

args = parser.parse_args()

# game conf
keymap = {'Breakout-v0': [1, 4, 5]}

# additional parameters
args.action_size = len(keymap[args.game])
args.action_map = keymap[args.game]