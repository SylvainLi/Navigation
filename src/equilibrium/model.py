# import sys
# import time
# import json
# import numpy as np
# from mip import *
from equilibrium.utility import *
from equilibrium.MIP import *
from equilibrium.approximation_algorithm import *


# def solve_sequence(market, args):
#     prices = []
#     solutions = []
#     rights = []
#     moneys = []
#     willingness_to_pay = 0
#     for i in range(1):
#         print(f"\nMarket {i}")
#         rights += [[b.rights for b in market.buyers]]
#         moneys += [[b.money for b in market.buyers]]
#         display_state(market)
#         t0 = time.time()
#         # Solve the sequence using the MIP.
#         price, solution, willingness_to_pay = market_equilibrium_MIP(
#             market, args, 1)
#         print("MIP in: ", time.time()-t0)
#         # epsilon = 0.0000001
#         # t0 = time.time()
#         # if abs(price - market_equilibrium_approx(market, args, epsilon))> epsilon:
#         #     print("approx in: ", time.time()-t0)
#         #     print("LALALALALALALALALAL Error: ", price)
#         #     return
#         # print("approx in: ", time.time()-t0)

#         if price == None:
#             print("Error occured.")
#             return 0
#         print("I: ", i, "\n")
#         if i < args.steps_in_episode-1:
#             market.resupply(args)
#         print(f"price {price}")
#         prices += [price]
#         solutions += [solution]
#     display_state(market)
#     plot_sequence(np.array(moneys), np.array(prices), np.array(
#         solutions), np.array(rights), willingness_to_pay, args)