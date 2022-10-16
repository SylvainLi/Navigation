import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patches as mpatches
import pickle
from machine_learning.environment import Market

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

matplotlib.use('Agg')


def draw(buyer_states, seller_states, last_sim, time, optimal_buyer, optimal_seller, args, greedy=True):
	dir_path = os.getcwd()

	plt.grid()
	returns = np.sum(buyer_states[:last_sim, :, :, 10], axis=1)
	mean = np.mean(returns, axis=-1)
	std = np.std(returns, axis=-1)
	plt.plot(time, mean, 'r', label='Return of the buyers in the episode.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	if np.all(args.consumptions[0] == args.consumptions):
		plt.axhline(y=optimal_buyer, color='b', linestyle='--', label='Optimal return.')
	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Return')
	plt.savefig('buyers_return.png')

	plt.clf()

	plt.grid()
	returns = np.sum(seller_states[:last_sim, :, :, 4], axis=1)
	mean = np.mean(returns, axis=-1)
	std = np.std(returns, axis=-1)
	plt.plot(time, mean, 'r', label='Return of the sellers in the episode.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	plt.axhline(y=optimal_seller, color='b', linestyle='--', label='Optimal return.')
	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Return')
	plt.savefig('sellers_return.png')

	plt.clf()

	plt.grid()
	price = np.mean(seller_states[:last_sim, :, :, 3], axis=1) * args.max_trade_price
	mean = np.mean(price, axis=-1)
	std = np.std(price, axis=-1)
	plt.plot(time, mean, 'r', label='Price of supplies.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	des_price = np.mean(buyer_states[:last_sim, :, :, 7], axis=1) * args.max_trade_price
	mean = np.mean(des_price, axis=-1)
	std = np.std(des_price, axis=-1)
	plt.plot(time, mean, 'b', label='Desired price of supplies.')
	plt.fill_between(time, mean - std, mean + std, facecolor='b', alpha=0.5, interpolate=True)

	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Price')
	plt.savefig('price_supply.png')

	plt.clf()

	plt.grid()
	volume = np.mean(seller_states[:last_sim, :, :, 2] * seller_states[:last_sim, :, :, 1], axis=1)
	mean = np.mean(volume, axis=-1)
	std = np.std(volume, axis=-1)
	plt.plot(time, mean, 'r', label='Volume of supplies offered.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	des_volume = np.mean(buyer_states[:last_sim, :, :, 6], axis=1) * args.max_trade_volume
	mean = np.mean(des_volume, axis=-1)
	std = np.std(des_volume, axis=-1)
	plt.plot(time, mean, 'b', label='Desired volume of supplies.')
	plt.fill_between(time, mean - std, mean + std, facecolor='b', alpha=0.5, interpolate=True)

	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Volume')
	plt.savefig('volume_supply.png')

	plt.clf()

	plt.grid()
	volume = np.mean(buyer_states[:last_sim, :, :, 4] * buyer_states[:last_sim, :, :, 3], axis=1)
	mean = np.mean(volume, axis=-1)
	std = np.std(volume, axis=-1)
	plt.plot(time, mean, 'r', label='Volume of rights offered.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	des_volume = np.mean(buyer_states[:last_sim, :, :, 8], axis=1) * args.max_trade_volume
	mean = np.mean(des_volume, axis=-1)
	std = np.std(des_volume, axis=-1)
	plt.plot(time, mean, 'b', label='Desired volume of rights.')
	plt.fill_between(time, mean - std, mean + std, facecolor='b', alpha=0.5, interpolate=True)

	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Volume')
	plt.savefig('volume_rights.png')

	plt.clf()

	plt.grid()
	price = np.mean(buyer_states[:last_sim, :, :, 5], axis=1) * args.max_trade_price
	mean = np.mean(price, axis=-1)
	std = np.std(price, axis=-1)
	plt.plot(time, mean, 'r', label='Price of rights.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	des_price = np.mean(buyer_states[:last_sim, :, :, 9], axis=1) * args.max_trade_price
	mean = np.mean(des_price, axis=-1)
	std = np.std(des_price, axis=-1)
	plt.plot(time, mean, 'b', label='Desired price of rights.')
	plt.fill_between(time, mean - std, mean + std, facecolor='b', alpha=0.5, interpolate=True)

	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Price')
	plt.savefig('price_rights.png')

	plt.clf()

	plt.grid()
	plt.axhline(y=0, color='b', linestyle='--')
	supply = seller_states[:last_sim, -1, :, 1]
	mean = np.mean(supply, axis=-1)
	std = np.std(supply, axis=-1)
	plt.plot(time, mean, 'r', label='Unsold supplies at the end of episode.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Supply')
	plt.savefig('ending_supply.png')

	plt.clf()

	plt.grid()
	plt.axhline(y=0, color='b', linestyle='--')
	supply = buyer_states[:last_sim, -1, :, 1]
	mean = np.mean(supply, axis=-1)
	std = np.std(supply, axis=-1)
	plt.plot(time, mean, 'r', label='Unused supplies at the end of episode.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Supply')
	plt.savefig('ending_supply_buyers.png')

	plt.clf()

	plt.grid()
	plt.axhline(y=0, color='b', linestyle='--')
	supply = buyer_states[:last_sim, -1, :, 2]
	mean = np.mean(supply, axis=-1)
	std = np.std(supply, axis=-1)
	plt.plot(time, mean, 'r', label='Money at the end of episode.')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)

	plt.legend()
	plt.xlim((1, time[-1]))
	plt.xlabel('Episode')
	plt.ylabel('Money')
	plt.savefig('ending_money.png')

	plt.clf()

	grid_size = int(np.sqrt(args.num_buyers))

	returns = np.sum(buyer_states[:last_sim, :, :, 10], axis=1)
	max_retrun = np.max(returns)
	min_return = np.min(returns)
	fig, axs = plt.subplots(grid_size, grid_size)
	for i in range(grid_size):
		for j in range(grid_size):
			axs[i, j].grid()
			axs[i, j].plot(returns[:, grid_size * i + j])
			if np.all(args.consumptions[0] == args.consumptions):
				axs[i, j].axhline(y=optimal_buyer, color='b', linestyle='--')
			axs[i, j].set_ylim(min_return, max_retrun)

			if j == 0 and i == grid_size - 1:
				axs[i, j].set(xlabel='Time', ylabel='Return')
			elif j == 0:
				axs[i, j].set(ylabel='Return')
			elif i == grid_size - 1:
				axs[i, j].set(xlabel='Time')

	plt.savefig(f'buyer_return_split.png')

	plt.clf()

	money = buyer_states[:last_sim, -1, :, 2]
	max_money = np.max(money)
	min_money = np.min(money)
	fig, axs = plt.subplots(grid_size, grid_size)
	for i in range(grid_size):
		for j in range(grid_size):
			axs[i, j].grid()
			axs[i, j].plot(money[:, grid_size * i + j])
			axs[i, j].set_ylim(1.02*min_money, 1.02*max_money)

			if j == 0 and i == grid_size - 1:
				axs[i, j].set(xlabel='Time', ylabel='Money')
			elif j == 0:
				axs[i, j].set(ylabel='Money')
			elif i == grid_size - 1:
				axs[i, j].set(xlabel='Time')

	plt.savefig(f'ending_money_split.png')

	plt.clf()
	consumption = args.consumptions / np.mean(args.consumptions)
	earnings = args.earnings / np.mean(args.earnings)

	fig, axs = plt.subplots(grid_size, grid_size)
	for i in range(grid_size):
		for j in range(grid_size):
			axs[i, j].hist([1, 2], weights=[consumption[grid_size * i + j], earnings[grid_size * i + j]])
			axs[i, j].set_xlim(left=0.5, right=2.5)
			axs[i, j].set_ylim(bottom=0, top=1.2 * max(np.max(consumption), np.max(earnings)))

			if j == 0 and i == grid_size - 1:
				axs[i, j].set(xlabel='Need, Earning', ylabel='')
			elif j == 0:
				axs[i, j].set(ylabel='')
			elif i == grid_size - 1:
				axs[i, j].set(xlabel='Need, Earning')

	plt.savefig('stats.png')
	plt.clf()

	frustrations = buyer_states[1:last_sim+1, 0, :, 11]
	max_frustrations = buyer_states[:last_sim, -1, :, 11]

	fig, axs = plt.subplots(grid_size, grid_size)
	for i in range(grid_size):
		for j in range(grid_size):
			axs[i, j].grid()
			axs[i, j].plot(frustrations[:, grid_size * i + j])
			axs[i, j].plot(max_frustrations[:, grid_size * i + j])

			if j == 0 and i == grid_size - 1:
				axs[i, j].set(xlabel='Episode', ylabel='Frustration')
			elif j == 0:
				axs[i, j].set(ylabel='Frustration')
			elif i == grid_size - 1:
				axs[i, j].set(xlabel='Episode')

	plt.savefig('frustration.png')
	plt.clf()

	if greedy:
		traded_rights = buyer_states[:last_sim, :, :, 12]
		traded_supply = buyer_states[:last_sim, :, :, 13]
		price_supply = seller_states[:last_sim, :, :, 3] * args.max_trade_price
		des_price_supply = buyer_states[:last_sim, :, :, 7] * args.max_trade_price
		price_rights = buyer_states[:last_sim, :, :, 5] * args.max_trade_price
		des_price_rights = buyer_states[:last_sim, :, :, 9] * args.max_trade_price
		available_supply = np.sum(seller_states[:last_sim, :, :, 1], axis=2)

		max_traded_rights = np.max(traded_rights)
		max_traded_supply = np.max(traded_supply)
		max_reward = np.max(buyer_states[:last_sim, :, :, 10])
		max_money = np.max(buyer_states[:last_sim, :, :, 2])

		os.chdir(dir_path)
		if not os.path.isdir('prices'):
			os.mkdir('prices')
		os.chdir('prices')

		num_figures = len(os.listdir(os.getcwd()))

		for greedy_sim in range(num_figures, last_sim):
			fig, axs = plt.subplots(grid_size, grid_size)
			for i in range(grid_size):
				for j in range(grid_size):
					axs[i, j].grid()
					axs[i, j].plot(np.mean(price_supply[greedy_sim], axis=-1), label='Price')
					axs[i, j].plot(des_price_supply[greedy_sim, :, grid_size * i + j], label='Desired Price')
					axs[i, j].plot(price_rights[greedy_sim, :, grid_size * i + j], label='Price Rights')
					axs[i, j].plot(des_price_rights[greedy_sim, :, grid_size * i + j], label='Desired Price Rights')
					axs[i, j].set_ylim((-0.02 * args.max_trade_price, 1.02 * args.max_trade_price))

					if j == 0 and i == grid_size - 1:
						axs[i, j].set(xlabel='Time', ylabel='Price')
					elif j == 0:
						axs[i, j].set(ylabel='Price')
					elif i == grid_size - 1:
						axs[i, j].set(xlabel='Time')

			lines_labels = [axs[0, 0].get_legend_handles_labels()]
			lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
			fig.legend(lines, labels, ncol=4)
			plt.savefig(f'episode_{greedy_sim}.png')

			plt.clf()

		os.chdir(dir_path)
		if not os.path.isdir('traded_rights'):
			os.mkdir('traded_rights')
		os.chdir('traded_rights')
		for greedy_sim in range(num_figures, last_sim):
			fig, axs = plt.subplots(grid_size, grid_size)
			for i in range(grid_size):
				for j in range(grid_size):
					axs[i, j].grid()
					axs[i, j].plot(traded_rights[greedy_sim, :, grid_size * i + j])
					axs[i, j].set_ylim((-0.02 * max_traded_rights, 1.02 * max_traded_rights))

					if j == 0 and i == grid_size - 1:
						axs[i, j].set(xlabel='Time', ylabel='Rights')
					elif j == 0:
						axs[i, j].set(ylabel='Rights')
					elif i == grid_size - 1:
						axs[i, j].set(xlabel='Time')

			plt.savefig(f'rights_{greedy_sim}.png')

			plt.clf()

		os.chdir(dir_path)
		if not os.path.isdir('rewards'):
			os.mkdir('rewards')
		os.chdir('rewards')
		for greedy_sim in range(num_figures, last_sim):
			fig, axs = plt.subplots(grid_size, grid_size)
			for i in range(grid_size):
				for j in range(grid_size):
					axs[i, j].grid()
					axs[i, j].plot(buyer_states[greedy_sim, :, grid_size * i + j, 10])
					axs[i, j].set_ylim((1.02 * args.missing_supply_reward, 1.02 * max_reward))

					if j == 0 and i == grid_size - 1:
						axs[i, j].set(xlabel='Time', ylabel='Rewards')
					elif j == 0:
						axs[i, j].set(ylabel='Rewards')
					elif i == grid_size - 1:
						axs[i, j].set(xlabel='Time')

			plt.savefig(f'rewards_{greedy_sim}.png')

			plt.clf()

		for greedy_sim in range(num_figures, last_sim):
			fig, axs = plt.subplots(grid_size, grid_size)
			for i in range(grid_size):
				for j in range(grid_size):
					axs[i, j].grid()
					axs[i, j].plot(buyer_states[greedy_sim, :, grid_size * i + j, 2])
					axs[i, j].set_ylim((0, 1.02 * max_money))

					if j == 0 and i == grid_size - 1:
						axs[i, j].set(xlabel='Time', ylabel='Money')
					elif j == 0:
						axs[i, j].set(ylabel='Money')
					elif i == grid_size - 1:
						axs[i, j].set(xlabel='Time')

			plt.savefig(f'money_{greedy_sim}.png')

			plt.clf()

		os.chdir(dir_path)
		if not os.path.isdir('traded_supply'):
			os.mkdir('traded_supply')
		os.chdir('traded_supply')
		for greedy_sim in range(num_figures, last_sim):
			fig, axs = plt.subplots(grid_size, grid_size)
			for i in range(grid_size):
				for j in range(grid_size):
					axs[i, j].grid()
					axs[i, j].plot(traded_supply[greedy_sim, :, grid_size * i + j])
					axs[i, j].set_ylim((-0.2, 1.1 * max_traded_supply))

					if j == 0 and i == grid_size - 1:
						axs[i, j].set(xlabel='Time', ylabel='Supply')
					elif j == 0:
						axs[i, j].set(ylabel='Supply')
					elif i == grid_size - 1:
						axs[i, j].set(xlabel='Time')

			fig.suptitle('Supplies Purchased by Buyers')
			plt.savefig(f'supply_{greedy_sim}.png')
			plt.clf()

			plt.grid()
			plt.plot(available_supply[greedy_sim])
			plt.ylabel('Total Supply')
			plt.xlabel('Time')
			plt.ylim((0, np.max(available_supply)))
			plt.title('Total Available Supply')
			plt.savefig(f'available_supply_{greedy_sim}.png')

			plt.clf()

def produce_output(dir_name):
	path = os.getcwd()
	os.chdir('Results')
	os.chdir(dir_name)
	args = pickle.load(open('args.pickle', 'rb'))
	seller_states = np.load('seller_states.npy').reshape((-1, args.steps_in_episode, args.num_sellers, 5))
	greedy_seller_states = np.load('greedy_seller_states.npy').reshape((-1, args.steps_in_episode, args.num_sellers, 5))
	buyer_states = np.load('buyer_states.npy').reshape((-1, args.steps_in_episode, args.num_buyers, 14))
	greedy_buyer_states = np.load('greedy_buyer_states.npy').reshape((-1, args.steps_in_episode, args.num_buyers, 14))
	dir_path = os.getcwd()

	# Compute optimal return of a buyer is all supplies are bought
	avr_supply_per_day = args.num_sellers * (args.seller_starting_supply + args.steps_in_episode * args.seller_earning_per_day)
	avr_supply_per_day += args.num_buyers * args.buyer_starting_supply
	avr_supply_per_day /= args.steps_in_episode * args.num_buyers
	missing_supply = max(0, args.consumption_on_step - avr_supply_per_day)
	available_supply = min(avr_supply_per_day, args.consumption_on_step)
	reward = args.in_stock_supply_reward * available_supply + args.missing_supply_reward * missing_supply
	optimal_buyer = reward * args.steps_in_episode

	optimal_seller = args.buyer_starting_money + args.steps_in_episode * args.buyer_earning_per_day
	optimal_seller *= args.num_buyers
	optimal_seller /= args.num_sellers

	try:
		last_sim = np.min(np.where(np.sum(seller_states[:, :, 0, 3], axis=1) == 0))
	except:
		last_sim = 10_000
	last_greedy_sim = last_sim // args.eval_every

	# Start with greedy plots
	if not os.path.isdir('greedy'):
		os.mkdir('greedy')
	os.chdir('greedy')

	greedy_time = np.arange(last_greedy_sim) + 1
	draw(greedy_buyer_states, greedy_seller_states, last_greedy_sim, greedy_time, optimal_buyer, optimal_seller, args)

	# Draw the training data
	os.chdir(dir_path)
	time = np.arange(last_sim) + 1
	draw(buyer_states, seller_states, last_sim, time, optimal_buyer, optimal_seller, args, greedy=False)

	os.chdir(path)

dir_name = '2022-04-03_16-19'
produce_output(dir_name)


