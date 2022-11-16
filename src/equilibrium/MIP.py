import numpy as np
import sympy
from mip import *
from equilibrium.utility import *


def market_equilibrium_MIP(market, args, verbose=1):
    total_supply = market.get_total_supply()

    utility_G = get_utility_G(market, args)
    # Array used by the MIP, storing all values of the utility for good for each feasible quantity.
    utilities_good_array = [
        [utility_G(buyer, x+market.buyers[buyer].supply)-utility_G(buyer, market.buyers[buyer].supply) for x in range(int(args.min_trade_volume), int(args.max_trade_volume) + 1, args.scale)] for buyer in range(args.num_buyers)]

    # Utility fonction for money, given a buyer and a quantity.
    # Carefull it is only used to derivate the willingness to pay, only final_money_reward is used in the model.
    sym_p = sympy.Symbol("sym_p")
    def sym_utility_M(buyer): return args.final_money_reward*sym_p

    def willingness_to_pay(buyer, x): return utility_G(
        buyer, x) - sym_utility_M(buyer)
    w_array = [[sympy.solveset(willingness_to_pay(buyer, x)).sup for buyer in range(
        args.num_buyers)] for x in range(total_supply+1)]
    model = Model(sense=MAXIMIZE, solver_name=CBC)

    # model.max_seconds = maxTime
    model.threads = args.threads
    model.verbose = 0

    # Variables

    #   If the buyer b is buying i items b_i = 1, 0 otherwise.
    x = np.array([[model.add_var(var_type=BINARY, name=(f"x[{b},{i}]")) for i in range(total_supply + 1)] for b in
                  range(args.num_buyers)])
    price = model.add_var(var_type=CONTINUOUS, name="price")

    # Objective

    model.objective = price

    # Constraints

    # Buy all supply.
    model += xsum([xsum([x[b][i] * i for i in range(total_supply + 1)]) for b in
                   range(args.num_buyers)]) == total_supply, "all supply sold"

    # Price upper bound.
    model += price <= args.max_trade_price, "price upper bound"

    for b in range(args.num_buyers):

        # Buy only once.
        model += xsum([x[b][i] for i in range(total_supply + 1)]
                      ) == 1, f"{b} buy only once"

        for i in range(total_supply + 1):

            # if args.fairness == 'talmud':
            bundle_multiplicator = 2 * \
                (i * args.scale) - market.buyer_states[0][0][b][3]  # double
            if args.fairness == "free":
                bundle_multiplicator = i*args.scale  # free market
            # The constraints are useless otherwise (and the solver is sad when I put a boolean in the constraint)
            if bundle_multiplicator != 0:

                # Budget constraint.
                model += price * bundle_multiplicator - (1 - x[b][i]) * abs(
                    bundle_multiplicator) * args.max_trade_price <= market.buyers[
                    b].money, f"Budget constraint b{b} buys {i}"

#  #                Budget constraint for same time transactions
                # if i <= market.buyer_states[0][0][b][3]:
                #     if i*args.scale > 0:
                #         model += price * i*args.scale - (1 - x[b][i]) * abs(
                #             i*args.scale) * args.max_trade_price <= market.buyers[
                #                     b].money, f"Budget constraint b{b} buys {i}"
                # else:
                #     model += price * bundle_multiplicator - (1 - x[b][i]) * abs(
                #         bundle_multiplicator) * args.max_trade_price <= market.buyers[
                #                 b].money, f"Budget constraint b{b} buys {i}"
                # Right amount bought constraint.
                model += price * args.final_money_reward * bundle_multiplicator - (1 - x[b][i]) * args.max_trade_price * args.final_money_reward * abs(
                    bundle_multiplicator) <= utilities_good_array[b][i], f"Right amount constraint b{b} buys {i}"
    status = model.optimize()

    # verbose = 1 print the optimal solution, verbose = 2 print also the slack of the constraints
    items_bought = []
    if verbose > 1:
        for c in model.constrs:
            if c.slack < 1 and "once" not in c.name and "all supply sold" not in c.name:
                print(c.name, c.slack)
    if status == OptimizationStatus.OPTIMAL:
        if verbose > 0:
            print(f"Optimal price: {price.x}\n")
        for b in range(args.num_buyers):
            for i in range(total_supply + 1):
                if x[b][i].x > 0:
                    items_bought += [i]
                    if verbose > 0:
                        print(
                            f"buyer {b} buys {i} items starting with {market.buyer_states[0][0][b][3]} rights.")
    else:
        print("model infeasible.")
    return price.x, items_bought, w_array
