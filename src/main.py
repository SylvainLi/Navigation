#!/usr/bin/python

from mip import *
from machine_learning.environment import Market
import datetime
from parametrize import *
from equilibrium.model import *
from parser import parse_json

dir_name = f'{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time())[:5].replace(":", "-")}'

usage = """usage: python name.py <path> <mode>
    <path> is the path to the input json file.
    <mode> is either "advice", which will run the equilibrium algorithm and return the adviced price for the commodity,
    or "predict", which will run the machine learning program which will predict the prices according to the json file.
    """
parse_json(args)

market = Market(args, args.consumptions, args.earnings, dir_name)
market.resupply(args)
args = market.args

if args.mode == "advice":

    price = market_equilibrium_approx(market, args, 0.0001, 0)
    print(f"We advice the price: {price}.")
elif args.mode == "predict":

    # Run machine learning model.
    #             TODO
    print("predict")
else:
    print(usage)
