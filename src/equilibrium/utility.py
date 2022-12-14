import numpy as np

# The amount of items and right b would have to buy for a quantity demand of items of good.
def get_coef_bundle(market, args, b, demand):
    if market.args.fairness == "free":
        return demand*market.args.scale
    b_right = market.buyer_states[0][0][b][3]
    return 2 * (demand * market.args.scale) - b_right  # With fairness.


def get_utility_G(market, args):
    # Utility fonction for good, given a buyer and a quantity.
    utility_G = lambda buyer, x: args.missing_supply_reward + x * (-args.missing_supply_reward + args.earnings[buyer]) / args.demands[buyer]

    if args.utility_type == "sqrt":
        utility_G = lambda buyer, x: args.earnings[buyer] / (
            np.sqrt(np.sqrt(args.demands[buyer]))) * np.sqrt(np.sqrt(x))
    if args.utility_type == "ln":
        utility_G = lambda buyer, x: args.earnings[buyer] / (
            np.log(args.demands[buyer]+1)) * np.log(x+1)

    if args.utility_type == "mixed":
        def mixed(buyer, x):
            if x <= market.buyers[buyer].supply:
                return args.missing_supply_reward + x * (
                -args.missing_supply_reward - market.buyers[buyer].supply + args.earnings[buyer]) / \
                                 args.demands[buyer]
            return args.earnings[buyer] / (
            np.sqrt(np.sqrt(args.demands[buyer]))) * np.sqrt(np.sqrt(x))
        utility_G = mixed
    return utility_G
    