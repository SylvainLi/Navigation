import numpy as np
from equilibrium.utility import *
from mip import *


def clean_price(market, demands, args, verbose=1):
    utility_G = get_utility_G(market, args)

    model = Model(sense=MAXIMIZE, solver_name=CBC)

    # model.max_seconds = maxTime
    model.threads = args.threads
    model.verbose = 0

    # Variables
    price = model.add_var(var_type=CONTINUOUS, name="price")

    # Objective
    model.objective = price

    # Constraints
    # Price upper bound.
    model += price <= args.max_trade_price, "price upper bound"

    for b in range(args.num_buyers):
        i = demands[b]
        # if args.fairness_model == 'eternal_talmud':
        bundle_multiplicator = 2 * \
            (i * args.scale) - market.buyers[b].rights  # double
        if args.fairness_model == "free_market":
            bundle_multiplicator = i*args.scale  # free market
        # The constraints are useless otherwise (and the solver is sad when I put a boolean in the constraint)
        if bundle_multiplicator != 0:

            # Budget constraint.
            model += price * bundle_multiplicator <= market.buyers[
                b].money, f"Budget constraint b{b} buys {i}"

            # Right amount bought constraint.
            model += price * args.end_money_reward * bundle_multiplicator <= (utility_G(
                b, i+market.buyers[b].supply)-utility_G(b, market.buyers[b].supply)), f"Right amount constraint b{b} buys {i}"
    status = model.optimize()

    # verbose = 1 print the optimal solution, verbose = 2 print also the slack of the constraints.
    items_bought = []
    if verbose > 1:
        for c in model.constrs:
            if c.slack < 1 and "once" not in c.name and "all supply sold" not in c.name:
                print(c.name, c.slack)
    if status == OptimizationStatus.OPTIMAL:
        if verbose > 0:
            print(f"Optimal price: {price.x}\n")
    else:
        print("model infeasible.")
    return price.x


def get_buyer_demand(b, market, price, args):
    utility_G = get_utility_G(market, args)
    demand = 0
    while demand <= market.get_total_supply() + 1 and price * get_coef_bundle(market, b, demand+1) <= market.buyers[b].money and price * args.end_money_reward * get_coef_bundle(market, b, demand+1) <= (utility_G(b, demand+1+market.buyers[b].supply)-utility_G(b, market.buyers[b].supply)):
        demand += 1
    demand_string = str(b) + " buys " + str(demand) + " at price: " + str(price * args.end_money_reward *
                                                                          get_coef_bundle(market, b, demand)) + " starting with: " + str(market.buyers[b].money) + " and "+str(market.buyers[b].rights) + "rights \n"
    return demand, demand_string


def get_demand_given_price(market, price,  args):
    total_demand = 0
    all_demand_str = ""

    for b in range(args.num_buyers):
        demand, demand_string = get_buyer_demand(b, market, price, args)
        all_demand_str += demand_string
        total_demand += demand

    return total_demand, all_demand_str


def market_equilibrium_approx(market, args, epsilon=0.0001, verbose=0):
    lower_bound = 0
    upper_bound = args.max_trade_price
    total_supply = market.get_total_supply()
    str = ""
    
    if verbose > 0:
        print("total supply: ", total_supply)
    while upper_bound - lower_bound > epsilon:
        price = (upper_bound + lower_bound)/2
        nb_items_sold, str = get_demand_given_price(market, price, args)
        if verbose > 0:
            print("items sold: ", nb_items_sold, "price: ", price)

        if nb_items_sold >= total_supply:
            if verbose > 0:
                print("lower bound: ", price, "upper bound: ", upper_bound)
            lower_bound = price

        elif nb_items_sold < total_supply:
            if verbose > 0:
                print(" lower bound: ", lower_bound, "upper bound: ", price)
            upper_bound = price
    print("")
    return price
