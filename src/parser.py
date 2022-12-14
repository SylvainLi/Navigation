import numpy as np
import json

def get_max_selling_price(data):
    return np.max([offer["price"] for offer in data["productSupply"]])


def get_total_supply(data):
    return np.sum([offer["amount"] for offer in data["productSupply"]])


def get_number_of_bids(data):
    return len(data["productSupply"])


def price_paid(demand, price, rights):
    return price * demand + price * max(0, (demand-rights))


def utility_for_good(x, max_demand, max_price):
    return max_price / (
        np.log(max_demand+1)) * np.log(x+1)


def get_buyers_money(data):
    return [demand["credit"] for demand in data["productDemand"]]


def get_buyers_demand(data):
    return [demand["amount"] for demand in data["productDemand"]]


def parse_json(args):
    path = args.path
    data_json = open(path)
    data = json.load(data_json)

    args.num_buyers = get_number_of_bids(data)
    args.demands = get_buyers_demand(data)
    args.earnings = get_buyers_money(data)

    args.buyer_starting_money = 0
    args.buyer_starting_supply = 0
    args.seller_starting_money = 0
    args.seller_starting_supply = 0
    args.seller_resupply_model = "constant"

    args.num_sellers = 1
    args.seller_earning_per_day = get_total_supply(data)
    args.max_trade_price = get_max_selling_price(data)
    
    return data
