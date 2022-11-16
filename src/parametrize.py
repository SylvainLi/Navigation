import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import cProfile

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--threads", default=10, type=int, help="Number of CPU threads to use.")
parser.add_argument("--seller_hidden_actor", default=32, type=int, help="Size of the hidden layer of the seller network.")
parser.add_argument("--seller_hidden_critic", default=256, type=int, help="Size of the hidden layer of the seller network.")
parser.add_argument("--buyer_hidden_actor", default=32, type=int, help="Size of the hidden layer of the buyer network.")
parser.add_argument("--buyer_hidden_critic", default=256, type=int, help="Size of the hidden layer of the buyer network.")
parser.add_argument("--actor_learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=1e-3, type=float, help="Learning rate.")

# MIP params
parser.add_argument("--scale", default=1, type=float, help="The scale of the bundles.") # The good has to be bought scale by scale.
parser.add_argument("--utility_type", default="sqrt", type=str, help="The utility function for good. Supported: linear, sqrt")
parser.add_argument("--price_G", default=1, type=float, help="the price of the items of good when fixed.")

# Simulation params
parser.add_argument("--num_sellers", default=4, type=int, help="Number of sellers on the market.")
parser.add_argument("--num_buyers", default=4, type=int, help="Number of buyers on the market.")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size of the training.")
parser.add_argument("--l2", default=0.01, type=float, help="L2 regularization constant.")
parser.add_argument("--gamma", default=0.99, type=float, help="Decay factor.")
parser.add_argument("--c_mean", default=0.0, type=float, help="Decay factor.")
parser.add_argument("--clip_norm", default=0.1, type=float, help="Gradient clip norm.")
parser.add_argument("--target_tau", default=0.002, type=float, help="Target network update weight.")
parser.add_argument("--train_noise_var", default=0.05, type=float, help="Critic update.")
parser.add_argument("--train_noise_clip", default=0.0, type=float, help="Critic update.")
parser.add_argument("--episodes", default=30, type=int, help="Training episodes.")
parser.add_argument("--eval_every", default=50, type=int, help="Evaluate every _ steps.")
parser.add_argument("--eval_for", default=100, type=int, help="Evaluation steps.")
parser.add_argument("--actor_update_freq", default=3, type=int, help="Train actor every _ critic trainings.")
parser.add_argument("--show_window", default=1_000_000, type=int, help="Train actor every _ critic trainings.")
parser.add_argument("--steps_in_episode", default=10, type=int, help="Number of days in each episode.")
parser.add_argument("--entropy_regularization_buyer", default=0.003, type=float, help="Variance of the folded gauss distribution to use initially.")
parser.add_argument("--entropy_regularization_seller", default=0.003, type=float, help="Variance of the folded gauss distribution to use initially.")

# Resources
parser.add_argument("--buyer_starting_money", default=0, type=float, help="The amount of money at the start of the simulation.")
parser.add_argument("--buyer_starting_supply", default=0, type=float, help="The amount of supply at the start of the simulation.")
parser.add_argument("--seller_starting_money", default=0, type=float, help="The amount of money at the start of the simulation.")
parser.add_argument("--seller_starting_supply", default=0, type=float, help="The amount of supply at the start of the simulation.")

parser.add_argument("--buyer_earning_per_day", default=1/8, type=float, help="The amount of money that the buyer recieves at the start of each day in the simulation.")
parser.add_argument("--seller_earning_per_day", default=1/4, type=float, help="The amount of supply that the seller recieves at the start of each day in the simulation.")
parser.add_argument("--demands", default=[1, 1, 1, 5], type=list, help="Good required per Market for agents.")
parser.add_argument("--earnings", default=[4, 5, 6, 1], type=list, help="Money gained by agents per step.")

parser.add_argument("--seller_resupply_model", default='constant', type=str, help="What amount of supply does a seller receive in a given day. Supported are: constant, rand_constant, cos, rand_cos.")
parser.add_argument("--sellers_share_buffer", default=False, type=bool, help="If the sellers randomly exchange samples used for training.")
parser.add_argument("--reward_shaping", default=False, type=bool, help="Use reward shaping.")
parser.add_argument("--shaping_const", default=0.1, type=float, help="Reward shaping scaling constant.")
parser.add_argument("--upgoing_policy", default=True, type=bool, help="Use upgoing policy update.")
parser.add_argument("--reward_clipping", default=True, type=bool, help="Clip rewards to [-clip_const,clip_const].")
parser.add_argument("--clip_const", default=1., type=float, help="Clipping constant.")
parser.add_argument("--fairness", default='talmud', type=str, help="Fairness models: free, talmud.")
parser.add_argument("--perishing_good", default=False, type=bool, help="Supplies are only stored by sellers for one step.")
parser.add_argument("--market_model", default='greedy', type=str, help="Market model to use: random, greedy, average, absolute.")

parser.add_argument("--consumption_on_step", default=1., type=float, help="Amount of supply consumed by buyer per step.")
parser.add_argument("--min_trade_volume", default=0., type=float, help="Minimum amount of supplies/rights to trade.")
parser.add_argument("--max_trade_volume", default=1., type=float, help="Maximum amount of money the rights/supplies can cost.")
parser.add_argument("--max_trade_price", default=1., type=float, help="Maximum amount of money the rights/supplies can cost.")
parser.add_argument("--renew_rights_every", default=1, type=int, help="Every _ steps to renew rights according to fairness.")
parser.add_argument("--in_stock_supply_reward", default=1., type=float, help="The reward gained by buyer if whole consumption is met.")
parser.add_argument("--missing_supply_reward", default=0., type=float, help="The reward gained by buyer if whole consumption is missing.")
parser.add_argument("--final_money_reward", default=1., type=float, help="The reward gained by buyer per unit of money at the end of an episode.")
parser.add_argument("--end_supply_reward", default=-1/8, type=float, help="The reward gained by seller for each unit of supply at the end of a step.")
parser.add_argument("--final_good_reward", default=1/2, type=float, help="The reward gained by seller for each unit of supply at the end of an episode.")

parser.add_argument("--path", default="../data/test.json", type=str, help="Path to the JSON file.")
parser.add_argument("--mode", default="advice", type=str, help="Either \"advice\", which will run the equilibrium algorithm and return the adviced price for the commodity, or \"predict\", which will run the machine learning program which will predict the prices according to the json file.")

args = parser.parse_args([] if "__file__" not in globals() else None)
