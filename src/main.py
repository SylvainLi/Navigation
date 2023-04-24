#!/usr/bin/python

from machine_learning.environment import Marketplace
import datetime
from parametrize import *
from equilibrium.approximation_algorithm import *
from parser import parse_json
import numpy as np

dir_name = f'{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time())[:5].replace(":", "-")}'

usage = """usage: python name.py <path> <mode>
    <path> is the path to the input json file.
    <mode> is either "advice", which will run the equilibrium algorithm and return the adviced price for the commodity,
    "predict-train", which will run the machine learning program which will predict the prices according to the json file,
    or "predict-load", which loads a Crisis with two sellers and two buyers with earnings (1/2, 3/2) and demands (3/2, 1/2).
    """
data = parse_json(args)

if args.mode == 'predict-load':
    args.num_sellers = 2

market = Marketplace(args, dir_name, multiagent=True, eval=False)
market.resupply(args)
market.seller_states[0][0][0][1] = args.seller_earning_per_day 
market.seller_states[0][0][0][2] = 1
market.distribute_right()
args = market.args

if args.mode == "advice":

    price = market_equilibrium_approx(market, args, 0.0001, 0)
    print(f"We advice the price: {price}.")
    print('Under this price, the buyers should buy the following amounts of goods: ')
    for b in range(args.num_buyers):
        demand, demand_string = get_buyer_demand(b, market, price, args)
        company_id = data["productDemand"][b]["company_id"]
        bid_id = data["productDemand"][b]["id"]
        print(f"Buyer {company_id} (id {bid_id}) should buy {demand} items of good.")

elif args.mode == "predict-train":

    # Run machine learning model.
    for crisis in range(args.episodes):
        market.simulate_crisis(args)

    seller_states = np.ones(shape=(args.num_sellers, 1))  # The actual states go here
    buyer_states = np.ones(shape=(args.num_buyers, 2))
    prices = market.price_good(seller_states, buyer_states)
    print(f"We predict the price: {prices}.")

elif args.mode == 'predict-load':
    market.load()

    seller_states = np.ones(shape=(args.num_sellers, 1))  # The actual states go here
    buyer_states = np.ones(shape=(args.num_buyers, 2))
    prices = market.price_good(seller_states, buyer_states)
    print(f"We predict the price: {prices}.")

else:
    print(usage)