import argparse
import numpy as np
parser = argparse.ArgumentParser()

# consumption, earning, in stock supply reward
Buyers = np.array([[10, 2, 20],
                   [2, 10, 5],
                   [4, 9, 7],
                   [5, 2, 6 ]])
                   
parser.add_argument("--path", default="../data/test.json", type=str,help="Path to input file in JSON.")
parser.add_argument("--mode", default="advice", type=str,help="launching mode. Can be either \"advice\" or \"predict\".")
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--threads", default=10, type=int, help="Number of CPU threads to use.")
parser.add_argument("--seller_hidden_layer_size", default=128, type=int, help="Size of the hidden layer of the seller network.")
parser.add_argument("--buyer_hidden_layer_size", default=128, type=int, help="Size of the hidden layer of the buyer network.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")

# Simulation params
parser.add_argument("--num_sellers", default=1, type=int, help="Number of sellers on the market.")
parser.add_argument("--num_buyers", default=len(Buyers), type=int, help="Number of buyers on the market.")
parser.add_argument("--steps_in_episode", default=20, type=int, help="Number of days in each episode.")
parser.add_argument("--batch_size", default=2048, type=int, help="Batch size of the training.")
parser.add_argument("--l2", default=0.05, type=float, help="L2 regularization constant.")
parser.add_argument("--c_mean", default=0.05, type=float, help="Mean close to 1/2 scaling.")
parser.add_argument("--gamma", default=0.99, type=float, help="Decay factor.")
parser.add_argument("--clip_norm", default=0.1, type=float, help="Gradient clip norm.")
parser.add_argument("--target_tau", default=0.002, type=float, help="Target network update weight.")
parser.add_argument("--max_memory_len", default=10_000, type=int, help="Maximum number of transitions to keep in buffer.")
parser.add_argument("--episodes", default=100, type=int, help="Training episodes.")
parser.add_argument("--eval_every", default=20, type=int, help="Evaluate every _ steps.")
parser.add_argument("--train_every", default=16, type=int, help="Train every _ steps.")
parser.add_argument("--variance", default=0.4, type=float, help="Variance of the folded gauss distribution to use initially.")
parser.add_argument("--min_variance", default=0.1, type=float, help="Decay of variance per epoch.")
parser.add_argument("--min_variance_at", default=2000, type=int, help="Decay of variance per epoch.")
parser.add_argument("--mask_prob", default=0.0, type=float, help="Probability of zeroing some prediction.")
parser.add_argument("--min_mask_prob", default=0.0, type=float, help="Probability of zeroing some prediction.")
parser.add_argument("--min_mask_prob_at", default=1000, type=int, help="Decay of variance per epoch.")
parser.add_argument("--mask_const", default=0.5, type=float, help="How much will the masked output be reduced.")
parser.add_argument("--mask_const_rights", default=0.5, type=float, help="How much will the masked volume of rights be reduced.")

# MIP params
parser.add_argument("--scale", default=1, type=float, help="The scale of the bundles.") # The good has to be bought scale by scale.
parser.add_argument("--utility_type", default="sqrt", type=str, help="The utility function for good. Supported: linear, sqrt")
parser.add_argument("--price_G", default=1, type=float, help="the price of the items of good when fixed.")

# Resources
parser.add_argument("--buyer_starting_money", default=0, type=float, help="The amount of money at the start of the simulation.")
parser.add_argument("--buyer_starting_supply", default=0, type=float, help="The amount of supply at the start of the simulation.")
parser.add_argument("--seller_starting_money", default=0, type=float, help="The amount of money at the start of the simulation.")
parser.add_argument("--seller_starting_supply", default=0, type=float, help="The amount of supply at the start of the simulation.")

parser.add_argument("--buyer_earning_per_day", default=1, type=float, help="The amount of money that the buyer recieves at the start of each day in the simulation.")
parser.add_argument("--seller_earning_per_day", default=15, type=float, help="The amount of supply that the seller recieves at the start of each day in the simulation.")
parser.add_argument("--consumptions", default=Buyers[:,0], type=list, help="Supplies required per step for agents.")
parser.add_argument("--earnings", default=Buyers[:,1], type=list, help="Money gained by agents per step.")

parser.add_argument("--seller_resupply_model", default='constant', type=str, help="What amount of supply does a seller receive in a given day. Supported are: constant, rand_constant, cos, rand_cos.")
parser.add_argument("--resupply_array", default=[], type=list, help="")
parser.add_argument("--fairness_model", default='eternal_talmud', type=str, help="Fairness models: eternal_talmud, free_market, free ordered market, talmud, eternal_talmud(eternal rights).")
parser.add_argument("--perishing_supplies", default=True, type=bool, help="Supplies are only stored by sellers for one step.")

parser.add_argument("--consumption_on_step", default=1., type=float, help="Amount of supply consumed by buyer per step.")
parser.add_argument("--min_trade_volume", default=0., type=float, help="Minimum amount of supplies/rights to trade.")
parser.add_argument("--max_trade_volume", default=100, type=float, help="Maximum amount of money the rights/supplies can cost.")
parser.add_argument("--max_trade_price", default=10, type=float, help="Maximum amount of money the rights/supplies can cost.")
parser.add_argument("--renew_rights_every", default=1, type=int, help="Every _ steps to renew rights according to fairness.")
parser.add_argument("--trade_cycles", default=48, type=int, help="Average market results over _ steps.")
parser.add_argument("--in_stock_supply_reward", default=Buyers[:,2], type=float, help="The reward gained by buyer if whole consumption is met.")
parser.add_argument("--missing_supply_reward", default=0, type=float, help="The reward gained by buyer if whole consumption is missing.")
parser.add_argument("--end_money_reward", default=1, type=float, help="The reward gained by buyer per unit of money at the end of an episode.")
parser.add_argument("--end_supply_reward", default=-0.01, type=float, help="The reward gained by seller for each unit of supply at the end of a step.")
parser.add_argument("--final_supply_reward", default=0.1, type=float, help="The reward gained by seller for each unit of supply at the end of an episode.")

args = parser.parse_args([] if "__file__" not in globals() else None)

# args.missing_supply_reward = 0
