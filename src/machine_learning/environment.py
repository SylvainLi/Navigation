import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import os
import datetime
import pickle
import random

from machine_learning.agents import Buyer, Seller
from machine_learning.agents import Buyer_Actor, Seller_Actor

def get_talmud_split(needs, total_supply):
	needs = np.array(needs)
	if np.sum(needs) / 2 == total_supply:
		return needs / 2

	elif np.sum(needs) / 2 > total_supply:  # not even half way full
		previous_demand = 0
		claims = np.zeros(len(needs))
		sorted_indexes = np.argsort(needs)
		for i, demand in enumerate(needs[sorted_indexes]):
			if total_supply - (len(needs) - i) * (
					demand - previous_demand) / 2 > 0:  # something is left over after this step
				total_supply -= (len(needs) - i) * (demand - previous_demand) / 2
				claims[i:] += (demand - previous_demand) / 2
				previous_demand = demand
			else:  # there is not enough liquid to bet to the top of the next cylinder
				claims[i:] += total_supply / (len(needs) - i)
				break

		out = np.zeros(len(needs))
		out[sorted_indexes] = claims
		return out

	else:  # more then half is full
		previous_demand = 0
		losses = np.zeros(len(needs))
		sorted_indexes = np.argsort(needs)
		total_loss = np.sum(needs) - total_supply
		for i, demand in enumerate(np.array(needs)[sorted_indexes]):
			if total_loss - (len(needs) - i) * (
					demand - previous_demand) / 2 > 0:  # something is left over after this step
				total_loss -= (len(needs) - i) * (demand - previous_demand) / 2
				losses[i:] += (demand - previous_demand)/2
				previous_demand = demand
			else:  # there is not enough liquid to bet to the top of the next cylinder
				losses[i:] += total_loss / (len(needs) - i)
				break

		out = np.zeros(len(needs))
		out[sorted_indexes] = losses
		return needs - out

class Market():
	def __init__(self, args, consumptions, earnings, dir_name):
		self.dir_name = dir_name
		# Each seller knows the supply of all sellers, and supply, money and rights of all buyers
		seller_actor = Seller_Actor(args)
		self.sellers = [Seller(args, self, id, seller_actor) for id in range(args.num_sellers)]

		self.consumptions = consumptions
		self.earnings = earnings
		# When buying supply, each buyer knows offered volume and price
		# for all sellers and supply, money and rights of all buyers
		self.buyer_observation_shape_buy = (2 * args.num_sellers + 3 * args.num_buyers + 1, )
		# When selling rights, the buyer looks at the the same thing, but
		# on top of that, they know the offered price and volume of rights
		# by the other buyers.
		self.buyer_observation_shape_sell = (2 * args.num_buyers + 3 * args.num_buyers + 1, )

		self.buyers = []
		actors = {}
		for i, (need, earning) in enumerate(zip(consumptions, earnings)):
			if (need, earning) in actors:
				self.buyers.append(Buyer(args, self, need, earning, i, actors[(need, earning)]))
			else:
				actors[(need, earning)] = Buyer_Actor(args)
				self.buyers.append(Buyer(args, self, need, earning, i, actors[(need, earning)]))

		self.step_num = 0
		self.greedy_step_num = 0
		self.total_step_num = 0
		self.step_in_episode = 0
		self.args = args
		self.variance = args.variance
		self.mask_prob = args.mask_prob
		self.log = {'buyer_mean_return': [],
					'buyer_std_return': [],
					'seller_mean_return': [],
					'seller_std_return': [],
					'supply_mean_price': [],
					'supply_std_price': [],
					'supply_mean_volume': [],
					'supply_std_volume': [],
					'right_mean_volume': [],
					'right_std_volume': [],
					'right_mean_price': [],
					'right_std_price': [],
					'des_supply_mean_volume': [],
					'des_supply_std_volume': [],
					'des_right_mean_volume': [],
					'des_right_std_volume': [],
					'des_supply_mean_price': [],
					'des_supply_std_price': [],
					'des_right_mean_price': [],
					'des_right_std_price': [],
					'frustration_max': [],
					'frustration_mean': [],
					'frustration_min': [],
					'total_frustration': [],
					'traded_rights': [],
					'traded_supply': [],
					'ending_supply_mean': [],
					'ending_supply_std': [],
					'ending_buyer_supply_mean': [],
					'ending_buyer_supply_std': [],
					'ending_money_mean': [],
					'ending_money_std': []}
		self.greedy_log = {'buyer_mean_return': [],
					'buyer_std_return': [],
					'seller_mean_return': [],
					'seller_std_return': [],
					'supply_mean_price': [],
					'supply_std_price': [],
					'supply_mean_volume': [],
					'supply_std_volume': [],
					'right_mean_volume': [],
					'right_std_volume': [],
					'right_mean_price': [],
					'right_std_price': [],
					'des_supply_mean_volume': [],
					'des_supply_std_volume': [],
					'des_right_mean_volume': [],
					'des_right_std_volume': [],
					'des_supply_mean_price': [],
					'des_supply_std_price': [],
					'des_right_mean_price': [],
					'des_right_std_price': [],
					'frustration_max': [],
					'frustration_mean': [],
					'frustration_min': [],
					'total_frustration': [],
					'traded_rights': [],
					'traded_supply': [],
					'ending_supply_mean': [],
					'ending_supply_std': [],
					'ending_buyer_supply_mean': [],
					'ending_buyer_supply_std': [],
					'ending_money_mean': [],
					'ending_money_std': []}

		self.memory_len = args.max_memory_len
		# States of sellers - for each seller we remember step_num(0), supply(1), offered volume(2) and price(3) and reward (4=-1)
		self.seller_states = np.zeros(shape=(self.memory_len, args.num_sellers, 5), dtype=np.float32)
		self.greedy_seller_states = np.zeros(shape=(self.memory_len, args.num_sellers, 5))

		# For buyers we remember their
		# step_num(0), supply(1), money(2), rights(3), offered volume(4) and price(5) of rights,
		# desired volume (6) and price (7) of supplies, vol (8) and price (9) of rights and reward(10)
		# frustration (11), traded rights (12) and traded supply (13)
		self.buyer_states = np.zeros(shape=(self.memory_len, args.num_buyers, 14), dtype=np.float32)
		self.greedy_buyer_states = np.zeros(shape=(self.memory_len, args.num_buyers, 14), dtype=np.float32)

		self.traded_rights = np.zeros(shape=(args.steps_in_episode, ))
		self.traded_supply = np.zeros(shape=(args.steps_in_episode, ))

	@tf.function
	def make_offers_tf(self, buyer_states, seller_states):
		offers_volume = tf.TensorArray(tf.float32, size=self.args.num_sellers)
		offers_price = tf.TensorArray(tf.float32, size=self.args.num_sellers)
		for id, seller in enumerate(self.sellers):
			action = seller.predict(buyer_states[tf.newaxis, ...], seller_states[tf.newaxis, id, ...])[0]

			offers_volume = offers_volume.write(id, action[0])
			offers_price = offers_price.write(id, action[1])

		offers_volume = offers_volume.stack()
		offers_price = offers_price.stack()
		return offers_volume, offers_price

	def make_offers(self, greedy):
		if greedy:
			for i, seller in enumerate(self.sellers):
				assert seller.supply >= 0, 'Seller supplies negative'
				self.greedy_seller_states[self.greedy_step_num, i, 1] = seller.supply

			for i, buyer in enumerate(self.buyers):
				assert buyer.supply >= 0, 'Buyer supplies negative'
				assert buyer.money >= 0, 'Buyer money negative'
				assert buyer.rights >= 0, 'Buyer rights negative'
				self.greedy_buyer_states[self.greedy_step_num, i, 1] = buyer.supply
				self.greedy_buyer_states[self.greedy_step_num, i, 2] = buyer.money
				self.greedy_buyer_states[self.greedy_step_num, i, 3] = buyer.rights

			buyer_states = self.greedy_buyer_states[self.greedy_step_num, :, 1:4]
			seller_states = self.greedy_seller_states[self.greedy_step_num, :, 1:2]

		else:
			for i, seller in enumerate(self.sellers):
				assert seller.supply >= 0, 'Seller supplies negative'
				self.seller_states[self.step_num, i, 1] = seller.supply

			for i, buyer in enumerate(self.buyers):
				assert buyer.supply >= 0, 'Buyer supplies negative'
				assert buyer.money >= 0, 'Buyer money negative'
				assert buyer.rights >= 0, 'Buyer rights negative'
				self.buyer_states[self.step_num, i, 1] = buyer.supply
				self.buyer_states[self.step_num, i, 2] = buyer.money
				self.buyer_states[self.step_num, i, 3] = buyer.rights

			buyer_states = self.buyer_states[self.step_num, :, 1:4]
			seller_states = self.seller_states[self.step_num, :, 1:2]

		offers_volume, offers_price = self.make_offers_tf(buyer_states, seller_states)

		if greedy:
			self.greedy_seller_states[self.greedy_step_num, :, 2] = offers_volume
			self.greedy_seller_states[self.greedy_step_num, :, 3] = offers_price
		else:
			mask = np.random.choice([self.args.mask_const, 1], size=offers_volume.shape, p=[self.mask_prob, 1-self.mask_prob])
			noise = np.random.normal(loc=0, scale=self.variance, size=offers_volume.shape)
			offers_volume = mask * np.clip(offers_volume + noise, a_min=0, a_max=1)
			mask = np.random.choice([self.args.mask_const, 1], size=offers_price.shape, p=[self.mask_prob, 1-self.mask_prob])
			noise = np.random.normal(loc=0, scale=self.variance, size=offers_price.shape)
			offers_price = mask * np.clip(offers_price + noise, a_min=0, a_max=1)

			self.seller_states[self.step_num, :, 2] = offers_volume
			self.seller_states[self.step_num, :, 3] = offers_price

	@tf.function
	def offer_rights_tf(self, seller_offers, buyer_states):
		actions = tf.TensorArray(tf.float32, size=self.args.num_buyers)
		for i, buyer in enumerate(self.buyers):
			action = buyer.predict_sell_action(seller_offers[tf.newaxis, ...], buyer_states[tf.newaxis, i, ...])[0]

			actions = actions.write(i, action)

		actions = actions.stack()
		return actions

	def offer_rights(self, greedy):
		if greedy:
			seller_states = self.greedy_seller_states[self.greedy_step_num, :, 2:4].copy()
			seller_states[:, 0] = seller_states[:, 0] * self.greedy_seller_states[self.greedy_step_num, :, 1].copy()
			seller_states[:, 1] = seller_states[:, 1] * self.args.max_trade_price

			buyer_states = self.greedy_buyer_states[self.greedy_step_num, :, 1:4].copy()

		else:
			seller_states = self.seller_states[self.step_num, :, 2:4].copy()
			seller_states[:, 0] = seller_states[:, 0] * self.seller_states[self.step_num, :, 1].copy()
			seller_states[:, 1] = seller_states[:, 1] * self.args.max_trade_price

			buyer_states = self.buyer_states[self.step_num, :, 1:4].copy()

		actions = self.offer_rights_tf(seller_states, buyer_states)

		if greedy:
			self.greedy_buyer_states[self.greedy_step_num, :, 4:6] = actions
		else:
			mask = np.random.choice([self.args.mask_const_rights, 1], size=actions.shape, p=[self.mask_prob, 1-self.mask_prob])
			noise = np.random.normal(loc=0, scale=self.variance, size=actions.shape)
			actions = mask * np.clip(actions + noise, a_min=0, a_max=1)

			self.buyer_states[self.step_num, :, 4:6] = actions

	@tf.function
	def order_supply_tf(self, seller_offers, buyer_states, rights_offers):
		actions = tf.TensorArray(tf.float32, size=self.args.num_buyers)
		for i, buyer in enumerate(self.buyers):
			other_rights = tf.concat([rights_offers[:i], rights_offers[i+1:]], axis=0)
			action = buyer.predict_buy_action(seller_offers[tf.newaxis, ...], buyer_states[tf.newaxis, i, ...],
											  other_rights[tf.newaxis, ...])[0]

			actions = actions.write(i, action)

		actions = actions.stack()
		return actions

	def order_supply(self, greedy):
		if greedy:
			buyer_states = self.greedy_buyer_states[self.greedy_step_num, :, 1:4].copy()

			seller_states = self.greedy_seller_states[self.greedy_step_num, :, 2:4].copy()
			seller_states[:, 0] = seller_states[:, 0] * self.greedy_seller_states[self.greedy_step_num, :, 1].copy()
			seller_states[:, 1] = seller_states[:, 1] * self.args.max_trade_price

			rights_offers = self.greedy_buyer_states[self.greedy_step_num, :, 4:6].copy()
			rights_offers[:, 0] = rights_offers[:, 0] * self.greedy_buyer_states[self.greedy_step_num, :, 3].copy()
			rights_offers[:, 1] = rights_offers[:, 1] * self.args.max_trade_price

		else:
			buyer_states = self.buyer_states[self.step_num, :, 1:4].copy()

			seller_states = self.seller_states[self.step_num, :, 2:4].copy()
			seller_states[:, 0] = seller_states[:, 0] * self.seller_states[self.step_num, :, 1].copy()
			seller_states[:, 1] = seller_states[:, 1] * self.args.max_trade_price

			rights_offers = self.buyer_states[self.step_num, :, 4:6].copy()
			rights_offers[:, 0] = rights_offers[:, 0] * self.buyer_states[self.step_num, :, 3].copy()
			rights_offers[:, 1] = rights_offers[:, 1] * self.args.max_trade_price

		actions = self.order_supply_tf(seller_states, buyer_states, rights_offers)

		if greedy:
			self.greedy_buyer_states[self.greedy_step_num, :, 6:10] = actions
		else:
			mask = np.random.choice([self.args.mask_const, 1], size=actions.shape, p=[self.mask_prob, 1-self.mask_prob])
			noise = np.random.normal(loc=0, scale=self.variance, size=actions.shape)
			actions = mask * np.clip(actions + noise, a_min=0, a_max=1)

			self.buyer_states[self.step_num, :, 6:10] = actions

	def trade_rights(self, greedy):
		if not self.args.fairness_model[-6:] == 'talmud':
			return

		if greedy:
			offered_volume = self.greedy_buyer_states[self.greedy_step_num, :, 4].copy()
			offered_volume = offered_volume * np.array([buyer.rights for buyer in self.buyers]) #self.buyer_states[self.step_num, :, 3].copy()
			offered_price = self.greedy_buyer_states[self.greedy_step_num, :, 5].copy() * self.args.max_trade_price

			desired_volume = self.greedy_buyer_states[self.greedy_step_num, :, 8].copy() * self.args.max_trade_volume
			desired_price = self.greedy_buyer_states[self.greedy_step_num, :, 9].copy() * self.args.max_trade_price

		else:
			offered_volume = self.buyer_states[self.step_num, :, 4].copy()
			offered_volume = offered_volume * np.array([buyer.rights for buyer in self.buyers]) #self.buyer_states[self.step_num, :, 3].copy()
			offered_price = self.buyer_states[self.step_num, :, 5].copy() * self.args.max_trade_price

			desired_volume = self.buyer_states[self.step_num, :, 8].copy() * self.args.max_trade_volume
			desired_price = self.buyer_states[self.step_num, :, 9].copy() * self.args.max_trade_price

		sold = np.zeros(shape=(self.args.trade_cycles, len(self.buyers), 2))
		bought = np.zeros(shape=(self.args.trade_cycles, len(self.buyers), 2))
		buyer_money = np.repeat([np.array([buyer.money for buyer in self.buyers])], self.args.trade_cycles, axis=0)
		for cycle in range(self.args.trade_cycles):
			for buyer_id in np.random.permutation(len(self.buyers)):
				for seller_id in np.random.permutation(len(self.buyers)):
					if buyer_id == seller_id:
						continue
					if desired_price[buyer_id] < offered_price[seller_id]:
						continue

					leftover_volume = offered_volume[seller_id] - sold[cycle, seller_id, 0]
					still_desired_volume = desired_volume[buyer_id] - bought[cycle, buyer_id, 0]

					if offered_price[seller_id] > 0:
						affordable_vol = buyer_money[cycle, buyer_id] / offered_price[seller_id]
						vol_to_buy = min(affordable_vol, leftover_volume, still_desired_volume) - 1e-4
					else:
						vol_to_buy = min(leftover_volume, still_desired_volume) - 1e-4

					if vol_to_buy > self.args.min_trade_volume:
						sold[cycle, seller_id, 0] += vol_to_buy
						bought[cycle, buyer_id, 0] += vol_to_buy

						money_to_pay = vol_to_buy * offered_price[seller_id]
						sold[cycle, seller_id, 1] += money_to_pay
						bought[cycle, buyer_id, 1] += money_to_pay
						buyer_money[cycle, buyer_id] -= money_to_pay

		total_trade_buyer = np.mean(bought, axis=0)
		total_trade_seller = np.mean(sold, axis=0)

		for buyer_id in range(self.args.num_buyers):
			traded_rights = total_trade_buyer[buyer_id, 0]
			self.buyers[buyer_id].rights += traded_rights
			self.buyers[buyer_id].money -= total_trade_buyer[buyer_id, 1]
			if greedy:
				self.greedy_buyer_states[self.greedy_step_num, buyer_id, 12] = traded_rights
			else:
				self.buyer_states[self.step_num, buyer_id, 12] = traded_rights

		for seller_id in range(self.args.num_buyers):
			self.buyers[seller_id].rights -= total_trade_seller[seller_id, 0]
			self.buyers[seller_id].money += total_trade_seller[seller_id, 1]

		self.traded_rights[self.step_in_episode] = np.sum(total_trade_buyer[:, 0])

	def trade_supply(self, greedy):
		if greedy:
			offered_volume = self.greedy_seller_states[self.greedy_step_num, :, 2].copy()
			offered_volume = offered_volume * np.array([seller.supply for seller in self.sellers])
			offered_price = self.greedy_seller_states[self.greedy_step_num, :, 3].copy() * self.args.max_trade_price

			desired_volume = self.greedy_buyer_states[self.greedy_step_num, :, 6].copy() * self.args.max_trade_volume
			desired_price = self.greedy_buyer_states[self.greedy_step_num, :, 7].copy() * self.args.max_trade_price

		else:
			offered_volume = self.seller_states[self.step_num, :, 2].copy()
			offered_volume = offered_volume * np.array([seller.supply for seller in self.sellers])
			offered_price = self.seller_states[self.step_num, :, 3].copy() * self.args.max_trade_price

			desired_volume = self.buyer_states[self.step_num, :, 6].copy() * self.args.max_trade_volume
			desired_price = self.buyer_states[self.step_num, :, 7].copy() * self.args.max_trade_price

		sold = np.zeros(shape=(self.args.trade_cycles, len(self.buyers), 2))
		bought = np.zeros(shape=(self.args.trade_cycles, len(self.buyers), 2))
		buyer_money = np.repeat([np.array([buyer.money for buyer in self.buyers])], self.args.trade_cycles, axis=0)
		buyer_rights = np.repeat([np.array([buyer.rights for buyer in self.buyers])], self.args.trade_cycles, axis=0)
		for cycle in range(self.args.trade_cycles):
			for buyer_id in np.random.permutation(len(self.buyers)):
				for seller_id in np.random.permutation(len(self.sellers)):
					if desired_price[buyer_id] < offered_price[seller_id]:
						continue

					right_volume = buyer_rights[cycle, buyer_id] if self.args.fairness_model[-6:] == 'talmud' else self.args.max_trade_volume
					leftover_volume = offered_volume[seller_id] - sold[cycle, seller_id, 0]
					still_desired_volume = desired_volume[buyer_id] - bought[cycle, buyer_id, 0]

					if offered_price[seller_id] > 0:
						affordable_vol = buyer_money[cycle, buyer_id] / offered_price[seller_id]
						vol_to_buy = min(affordable_vol, right_volume, leftover_volume, still_desired_volume) - 1e-4
					else:
						vol_to_buy = min(right_volume, leftover_volume, still_desired_volume) - 1e-4

					if vol_to_buy > self.args.min_trade_volume:
						sold[cycle, seller_id, 0] += vol_to_buy
						bought[cycle, buyer_id, 0] += vol_to_buy

						money_to_pay = vol_to_buy * offered_price[seller_id]
						sold[cycle, seller_id, 1] += money_to_pay
						bought[cycle, buyer_id, 1] += money_to_pay

						buyer_money[cycle, buyer_id] -= money_to_pay
						buyer_rights[cycle, buyer_id] -= vol_to_buy

		total_trade_buyer = np.mean(bought, axis=0)
		total_trade_seller = np.mean(sold, axis=0)

		for buyer_id in range(self.args.num_buyers):
			traded_supply = total_trade_buyer[buyer_id, 0]
			self.buyers[buyer_id].supply += traded_supply
			self.buyers[buyer_id].frustration -= traded_supply
			self.buyers[buyer_id].money -= total_trade_buyer[buyer_id, 1]
			if greedy:
				self.greedy_buyer_states[self.greedy_step_num, buyer_id, 13] = traded_supply
			else:
				self.buyer_states[self.step_num, buyer_id, 13] = traded_supply

		for seller_id in range(self.args.num_sellers):
			self.sellers[seller_id].supply -= total_trade_seller[seller_id, 0]

		seller_rewards = total_trade_seller[:, 1]

		self.traded_supply[self.step_in_episode] = np.sum(total_trade_buyer[:, 0])

		return seller_rewards

	def resupply(self, args):
		# Give sellers rewards for selling and add new supply to them
		new_supplies = 0
		for seller in self.sellers:
			if args.seller_resupply_model == 'constant':
				new_supply = args.seller_earning_per_day
			elif args.seller_resupply_model == 'rand_constant':
				new_supply = int(np.round(np.random.normal(loc=args.seller_earning_per_day,
											  scale=args.seller_earning_per_day / 10)))
			elif args.seller_resupply_model == 'cos':
				cos_factor = (np.cos(self.step_in_episode * 2 * np.pi / args.steps_in_episode) + 2) / 2
				new_supply = args.seller_earning_per_day * cos_factor
			elif args.seller_resupply_model == 'rand_cos':
				cos_factor = (np.cos(self.step_in_episode * 2 * np.pi / args.steps_in_episode) + 2) / 2
				new_supply = np.random.normal(loc=args.seller_earning_per_day * cos_factor,
											  scale=args.seller_earning_per_day / 10)
			elif args.seller_resupply_model == 'array':
				# assert len(args.resupply_array) == args.num_sellers and [len(args.resupply_array[i]) == args.steps_in_episode for i in range(args.num_sellers) ].all()
				assert(len(args.resupply_array) == args.steps_in_episode)
				print(self.step_num,  args.steps_in_episode,"\n")
				new_supply = args.resupply_array[self.step_num]
			else:
				raise NotImplementedError('Unknown seller resupply model.')

			new_supplies += args.renew_rights_every * max(new_supply, 0)
			if args.perishing_supplies:
				seller.supply = args.renew_rights_every * max(new_supply, 0)
			else:
				seller.supply += args.renew_rights_every * max(new_supply, 0)

		total_supply = 0
		for seller in self.sellers:
			total_supply += seller.supply

		# Distribute rights
		if self.args.fairness_model[:7] == 'eternal':
			new_rights = get_talmud_split(self.consumptions - np.array([min(b.need, b.supply) for b in self.buyers ]), new_supplies)
		else:
			new_rights = get_talmud_split(self.consumptions - np.array([min(b.need, b.supply) for b in self.buyers ]), total_supply)

		for earning, buyer, new_right in zip(self.earnings, self.buyers, new_rights):
			buyer.money += earning * args.buyer_earning_per_day * args.renew_rights_every
			buyer.rights = new_right
			buyer.frustration += new_right
			buyer.total_frustration += new_right

	def get_total_supply(self):
		return int(sum([seller.supply for seller in self.sellers]))

	def reward_agents(self, seller_rewards, args, greedy):
		if greedy:
			for reward, seller in zip(seller_rewards, self.sellers):
				self.greedy_seller_states[self.greedy_step_num, seller.id, 4] = reward + self.args.end_supply_reward * seller.supply

			for id, (consumption, buyer) in enumerate(zip(self.consumptions, self.buyers)):
				missing_supply = max(0, args.consumption_on_step - buyer.supply / consumption)
				available_supply = min(buyer.supply / consumption, args.consumption_on_step)
				reward = args.in_stock_supply_reward * available_supply + args.missing_supply_reward * missing_supply
				self.greedy_buyer_states[self.greedy_step_num, id, 10] = reward

		else:
			for reward, seller in zip(seller_rewards, self.sellers):
				self.seller_states[self.step_num, seller.id, 4] = reward + self.args.end_supply_reward * seller.supply

			for id, (consumption, buyer) in enumerate(zip(self.consumptions, self.buyers)):
				missing_supply = max(0, args.consumption_on_step - buyer.supply / consumption)
				available_supply = min(buyer.supply / consumption, args.consumption_on_step)
				reward = args.in_stock_supply_reward * available_supply + args.missing_supply_reward * missing_supply
				self.buyer_states[self.step_num, id, 10] = reward

	def reward_buyers_at_end(self, greedy):
		if greedy:
			for buyer in self.buyers:
				self.greedy_buyer_states[self.greedy_step_num, buyer.id, 10] += self.args.end_money_reward * buyer.money

			for seller in self.sellers:
				self.greedy_seller_states[self.greedy_step_num, seller.id, 4] += self.args.final_supply_reward * seller.supply

		else:
			for buyer in self.buyers:
				self.buyer_states[self.step_num, buyer.id, 10] += self.args.end_money_reward * buyer.money

			for seller in self.sellers:
				self.seller_states[self.step_num, seller.id, 4] += self.args.final_supply_reward * seller.supply

	def consume_supply(self, args):
		for consumption, buyer in zip(self.consumptions, self.buyers):
			buyer.supply = max(buyer.supply - consumption * args.consumption_on_step, 0)

	def reset_episode(self, args, greedy):
		self.step_in_episode = 0

		buyer_money = [buyer.money for buyer in self.buyers]
		buyer_supply = [buyer.supply for buyer in self.buyers]
		seller_supply = [seller.supply for seller in self.sellers]

		if greedy:
			log = self.greedy_log
			s = self.greedy_step_num
			if s == 0:
				s = self.memory_len
			seller_states = self.greedy_seller_states.copy()
			buyer_states = self.greedy_buyer_states.copy()
		else:
			log = self.log
			s = self.step_num
			if s == 0:
				s = self.memory_len
			seller_states = self.seller_states.copy()
			buyer_states = self.buyer_states.copy()

		log['ending_money_mean'].append(np.mean(buyer_money))
		log['ending_money_std'].append(np.std(buyer_money))

		log['ending_supply_mean'].append(np.mean(seller_supply))
		log['ending_supply_std'].append(np.std(seller_supply))

		log['ending_buyer_supply_mean'].append(np.mean(buyer_supply))
		log['ending_buyer_supply_std'].append(np.std(buyer_supply))

		buyer_returns = np.sum(buyer_states[s-100:s, :, -1], axis=0)
		log['buyer_mean_return'].append(np.mean(buyer_returns))
		log['buyer_std_return'].append(np.std(buyer_returns))
		seller_returns = np.sum(seller_states[s-100:s, :, -1], axis=0)
		log['seller_mean_return'].append(np.mean(seller_returns))
		log['seller_std_return'].append(np.std(seller_returns))

		supply_volume = np.mean(seller_states[s-100:s, :, 2] * seller_states[s-100:s, :, 1], axis=0)
		log['supply_mean_volume'].append(np.mean(supply_volume))
		log['supply_std_volume'].append(np.std(supply_volume))
		supply_price = np.mean(seller_states[s-100:s, :, 3], axis=0) * self.args.max_trade_price
		log['supply_mean_price'].append(np.mean(supply_price))
		log['supply_std_price'].append(np.std(supply_price))
		rights_volume = np.mean(buyer_states[s-100:s, :, 4] * buyer_states[s-100:s, :, 3], axis=0)
		log['right_mean_volume'].append(np.mean(rights_volume))
		log['right_std_volume'].append(np.std(rights_volume))
		rights_price = np.mean(buyer_states[s-100:s, :, 5], axis=0) * self.args.max_trade_price
		log['right_mean_price'].append(np.mean(rights_price))
		log['right_std_price'].append(np.std(rights_price))

		des_sup_volume = np.mean(buyer_states[s-100:s, :, 6], axis=0) * self.args.max_trade_volume
		log['des_supply_mean_volume'].append(np.mean(des_sup_volume))
		log['des_supply_std_volume'].append(np.std(des_sup_volume))
		des_supply_price = np.mean(buyer_states[s-100:s, :, 7], axis=0) * self.args.max_trade_price
		log['des_supply_mean_price'].append(np.mean(des_supply_price))
		log['des_supply_std_price'].append(np.std(des_supply_price))
		des_right_volume = np.mean(buyer_states[s-100:s, :, 8], axis=0) * self.args.max_trade_volume
		log['des_right_mean_volume'].append(np.mean(des_right_volume))
		log['des_right_std_volume'].append(np.std(des_right_volume))
		des_right_price = np.mean(buyer_states[s-100:s, :, 9], axis=0) * self.args.max_trade_price
		log['des_right_mean_price'].append(np.mean(des_right_price))
		log['des_right_std_price'].append(np.std(des_right_price))

		log['frustration_max'].append(np.max([buyer.frustration for buyer in self.buyers]))
		log['total_frustration'].append(np.max([buyer.total_frustration for buyer in self.buyers]))

		log['traded_rights'].append(np.sum(self.traded_rights))
		self.traded_rights = np.zeros(shape=(args.steps_in_episode, ))
		log['traded_supply'].append(np.sum(self.traded_supply))
		self.traded_supply = np.zeros(shape=(args.steps_in_episode, ))

		if greedy:
			for i, buyer in enumerate(self.buyers):
				self.greedy_buyer_states[self.greedy_step_num-1, i, 11] = buyer.frustration
				self.greedy_buyer_states[self.greedy_step_num, i, 11] = buyer.total_frustration
		else:
			for i, buyer in enumerate(self.buyers):
				self.buyer_states[self.step_num-1, i, 11] = buyer.frustration
				self.buyer_states[self.step_num, i, 11] = buyer.total_frustration

		for buyer in self.buyers:
			buyer.on_episode_end(args)

		for seller in self.sellers:
			seller.on_episode_end(args)

		self.reward_buyers_at_end(greedy)
		self.adjust_exploration()

	def adjust_exploration(self):
		decay = (self.args.variance - self.args.min_variance) / self.args.min_variance_at
		if self.variance > self.args.min_variance + decay:
			self.variance -= decay
		else:
			self.variance = self.args.min_variance

		decay = (self.args.mask_prob - self.args.min_mask_prob) / self.args.min_mask_prob_at
		if self.mask_prob > self.args.min_mask_prob + decay:
			self.mask_prob -= decay
		else:
			self.mask_prob = self.args.min_mask_prob

	def train_all(self):
		# Seller training
		samples = np.random.randint(0, min(self.total_step_num, self.args.max_memory_len) - 1, self.args.batch_size)
		next_samples = samples + 1
		buyer_data = self.buyer_states[samples]
		buyer_next_data = self.buyer_states[next_samples]
		seller_data = self.seller_states[samples]
		seller_next_data = self.seller_states[next_samples]

		if self.total_step_num % (2 * self.args.train_every * self.args.num_sellers) == 0:
			self.train_sellers_actor(buyer_data, seller_data)
		self.train_sellers_critics(buyer_data, seller_data, buyer_next_data, seller_next_data)

		# Buyer training
		samples = np.random.randint(0, min(self.total_step_num, self.args.max_memory_len) - 1, self.args.batch_size)
		next_samples = samples + 1
		buyer_data = self.buyer_states[samples]
		buyer_next_data = self.buyer_states[next_samples]
		seller_data = self.seller_states[samples]
		seller_next_data = self.seller_states[next_samples]

		if self.total_step_num % (2 * self.args.train_every * self.args.num_buyers) == 0:
			self.train_buyers_actor(buyer_data, seller_data)
		self.train_buyers_critics(buyer_data, seller_data, buyer_next_data, seller_next_data)

	@tf.function
	def train_sellers_actor(self, buyer_states, seller_states):
		for seller in self.sellers:
			seller.train_actor(buyer_states, seller_states)

	@tf.function
	def train_sellers_critics(self, buyer_states, seller_states, next_buyer_states, next_seller_states):
		for seller in self.sellers:
			seller.train_critics(buyer_states, seller_states, next_buyer_states, next_seller_states)

	@tf.function
	def train_buyers_actor(self, buyer_states, seller_states):
		for buyer in self.buyers:
			buyer.train_actor(buyer_states, seller_states)

	@tf.function
	def train_buyers_critics(self, buyer_states, seller_states, next_buyer_states, next_seller_states):
		for buyer in self.buyers:
			buyer.train_critics(buyer_states, seller_states, next_buyer_states, next_seller_states)

	def save_all(self):
		path = os.getcwd()
		os.chdir('Results/'+self.dir_name)
		np.save('seller_states.npy', self.seller_states)
		np.save('greedy_seller_states.npy', self.greedy_seller_states)
		np.save('buyer_states.npy', self.buyer_states)
		np.save('greedy_buyer_states.npy', self.greedy_buyer_states)

		dir_name = f'models'
		if not os.path.isdir(dir_name):
			os.mkdir(dir_name)
		os.chdir(dir_name)
		for id, buyer in enumerate(self.buyers):
			buyer.save(id)

		for id, seller in enumerate(self.sellers):
			seller.save(id)
		os.chdir(path)

	def load_all(self, load_from=-1):
		path = os.getcwd()
		os.chdir('./Results')
		folders = os.listdir('.')
		os.chdir(f'./{folders[load_from]}/models')
		for id, buyer in enumerate(self.buyers):
			buyer.load(id)

		for id, seller in enumerate(self.sellers):
			seller.load(id)
		os.chdir(path)

	def step(self, args, greedy):
		# Resupply sellers and buyers
		if self.step_in_episode % self.args.renew_rights_every == 0:
			self.resupply(args)
		if greedy:
			self.greedy_seller_states[self.greedy_step_num, :, 0] = self.step_in_episode / (args.steps_in_episode - 1)
			self.greedy_buyer_states[self.greedy_step_num, :, 0] = self.step_in_episode / (args.steps_in_episode - 1)

		else:
			self.seller_states[self.step_num, :, 0] = self.step_in_episode / (args.steps_in_episode - 1)
			self.buyer_states[self.step_num, :, 0] = self.step_in_episode / (args.steps_in_episode - 1)

		# First, sellers make their offers
		self.make_offers(greedy)

		# Second, the buyers react by offering rights to sell
		self.offer_rights(greedy)

		# Lastly, order all
		self.order_supply(greedy)

		# Actually trade rights
		self.trade_rights(greedy)

		# Trade supplies
		seller_rewards = self.trade_supply(greedy)

		# Reward (only if we added the state for training also)
		self.reward_agents(seller_rewards, args, greedy)

		self.consume_supply(args)

		if self.total_step_num > 2 * self.args.batch_size and self.total_step_num % args.train_every == 0:

			self.train_all()

		if greedy:
			self.greedy_step_num += 1
			self.step_in_episode += 1
		else:
			self.total_step_num += 1
			self.step_num = self.total_step_num % self.memory_len
			if self.step_num == 0:
				print('Buffer full!')
			self.step_in_episode += 1



