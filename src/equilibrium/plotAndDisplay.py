import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_sequence(moneys, prices, solutions, rights, willingness_to_pay, args, type=1):
    # colors = [(ax._get_lines.prop_cycler)['color'] for b in range(len(solutions))]
    fig, axs= plt.subplots(3, 3)


    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.3  # the amount  of width reserved for blank space between subplots
    hspace = 0.3  # the amount of height reserved for white space between subplots

    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    axs[0][0].plot(range(len(prices)), prices, 'o-', label="price")
    axs[0][0].set_xlabel("Time in market")
    axs[0][0].set_ylabel("money in arb. unit")
    axs[0][0].set_title("Price evolution during the crisis.")
    axs[0][0].grid(True)
    axs[0][0].set_ylim(bottom=0)
    axs[0][0].legend()
    axs[0][0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0][0].yaxis.set_major_locator(MaxNLocator(integer=True))

    axs[0][1].plot(range(len(moneys)), moneys, 'o-')
    axs[0][1].set_title('Buyers money evolution during the crisis.')
    axs[0][1].set_xlabel("Time in market")
    axs[0][1].set_ylabel("money in arb. unit")
    axs[0][1].grid(True)
    axs[0][1].set_ylim(bottom=0)
    axs[0][1].legend([f"buyer {i}" for i in range(len(solutions[0]))])
    axs[0][1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0][1].yaxis.set_major_locator(MaxNLocator(integer=True))

    axs[0][2].plot(range(len(solutions)), solutions, 'o-')
    axs[0][2].set_title('Items bought per buyer during the crisis.')
    axs[0][2].set_xlabel("Time in market")
    axs[0][2].set_ylabel("items bought")
    axs[0][2].grid(True)
    axs[0][2].set_ylim(bottom=0)
    axs[0][2].legend([f"buyer {i}" for i in range(len(solutions[0]))])
    axs[0][2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0][2].yaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1][0].plot(range(len(willingness_to_pay)), willingness_to_pay, 'o-')
    axs[1][0].set_title("Willingness to pay.")
    axs[1][0].set_xlabel("Number of items")
    axs[1][0].set_ylabel("Price")
    axs[1][0].grid(True)
    axs[1][0].set_ylim(bottom=0)
    axs[1][0].legend([f"buyer {i}" for i in range(len(solutions[0]))])
    axs[1][0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1][0].yaxis.set_major_locator(MaxNLocator(integer=True))

    axs[2][1].text(0.05, -0.4, f"consumptions: {args.consumptions}\nearnings: {args.earnings}\nsupply: {args.seller_resupply_model}\nseller earning: {args.seller_earning_per_day}\nuG(0)={args.missing_supply_reward}\nuG(demand)={args.in_stock_supply_reward}\n uG type: {args.utility_type}",
               transform=axs[1][1].transAxes, fontsize=14, verticalalignment='top')

    if args.fairness_model != "free_market":
        axs[1][1].plot(range(len(rights)), rights, 'o-')
        axs[1][1].set_title("Items of right in the beginning of market.")
        axs[1][1].set_xlabel("Time in market")
        axs[1][1].set_ylabel("Number of items")
        axs[1][1].grid(True)
        axs[1][1].set_ylim(bottom=0)
        axs[1][1].legend([f"buyer {i}" for i in range(len(solutions[0]))])
        axs[1][1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1][1].yaxis.set_major_locator(MaxNLocator(integer=True))

        axs[1][2].plot(range(len(solutions)), (solutions - rights), 'o-')
        axs[1][2].set_title("Items of right bought.")
        axs[1][2].set_xlabel("Time in market")
        axs[1][2].set_ylabel("Number of items")
        axs[1][2].grid(True)
        axs[1][2].legend([f"buyer {i}" for i in range(len(solutions[0]))])
        axs[1][2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1][2].yaxis.set_major_locator(MaxNLocator(integer=True))


    stocks = []
    for b in range(args.num_buyers):
        b_stock = [0]
        for i in range(len(solutions)):
            b_stock += [max(0, b_stock[-1]+solutions[i][b]-args.consumptions[b])]
        stocks += [b_stock[1:]]
    stocks = np.array(stocks).transpose()
    axs[2][0].plot(range(len(solutions)), stocks, 'o-')
    axs[2][0].set_title("Buyer stock")
    axs[2][0].set_xlabel("Number of items")
    axs[2][0].set_ylabel("Time in market")
    axs[2][0].grid(True)
    axs[2][0].set_ylim(bottom=0)
    axs[2][0].legend([f"buyer {i}" for i in range(len(solutions[0]))])
    axs[2][0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2][0].yaxis.set_major_locator(MaxNLocator(integer=True))

    axs[2][2].plot(range(len(solutions)), (rights - solutions)*((rights - solutions)>=0), 'o-')
    axs[2][2].set_title("Frustration.")
    axs[2][2].set_ylabel("frustration")
    axs[2][2].set_xlabel("Time in market")
    axs[2][2].grid(True)
    axs[2][2].legend([f"buyer {i}" for i in range(len(solutions[0]))])
    axs[2][2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2][2].yaxis.set_major_locator(MaxNLocator(integer=True))

    # if type < 3:
    #     plt.gca().set_prop_cycle(None)
    #     axs[1].plot(range(len(solutions)), solutions, 'o-', label=["buyer " + str(i) for i in range(len(solutions[0]))])
    #     axs[1].set_xlabel = "market ID"
    #     axs[1].set_ylabel = "number of items bought",
    #     axs[1].set_title = "price plot"
    #     if type == 2:
    #         plt.gca().set_prop_cycle(None)
    #         axs[1].plot(range(len(solutions)), rights, 'o-')
    # if type == 3:
    #     axs[1].plot(range(len(solutions)), solutions - rights, 'o-')
    plt.show()


def display_state(market):
    print(f"\ntotal supply: {int(sum([seller.supply for seller in market.sellers]))}")
    print(f"Money, Stock, Rights, Consumption")
    for i, b in enumerate(market.buyers):
        print(f"b{i} m: {b.money:.2f}, s: {b.supply}, r: {b.rights}, c {b.need}")
