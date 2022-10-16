import numpy as np

def get_coef_bundle(market, b, demand):
    if market.args.fairness_model == "free_market":
        return demand*market.args.scale
    return 2 * (demand * market.args.scale) - market.buyers[b].rights  # With fairness.


def get_utility_G(market, args):
    # Utility fonction for good, given a buyer and a quantity.
    utility_G = lambda buyer, x: args.missing_supply_reward + x * (-args.missing_supply_reward + args.in_stock_supply_reward[buyer]) / args.consumptions[buyer]

    if args.utility_type == "total":
        utility_G = lambda buyer, x: args.in_stock_supply_reward[buyer] / (
            np.log(2*args.consumptions[buyer] + 1)) * np.log(2*x + 1)/market.get_total_supply()
    if args.utility_type == "test":
        utility_G = lambda buyer, x: args.in_stock_supply_reward[buyer] / (
            np.sqrt(args.consumptions[buyer])) * np.sqrt(x)
    if args.utility_type == "sqrt":
        utility_G = lambda buyer, x: args.in_stock_supply_reward[buyer] / (
            np.sqrt(np.sqrt(args.consumptions[buyer]))) * np.sqrt(np.sqrt(x))
    if args.utility_type == "ln":
        utility_G = lambda buyer, x: args.in_stock_supply_reward[buyer] / (
            np.log(args.consumptions[buyer]+1)) * np.log(x+1)

    if args.utility_type == "mixed":
        def mixed(buyer, x):
            if x <= market.buyers[buyer].supply:
                return args.missing_supply_reward + x * (
                -args.missing_supply_reward - market.buyers[buyer].supply + args.in_stock_supply_reward) / \
                                 args.consumptions[buyer]
            return args.in_stock_supply_reward / (
            np.sqrt(np.sqrt(args.consumptions[buyer]))) * np.sqrt(np.sqrt(x))
        utility_G = mixed
    return utility_G
    