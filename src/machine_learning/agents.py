import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp

import numpy as np

class Buyer_Actor():
	def __init__(self, args):
		seller_offers = tf.keras.Input(shape=(args.num_sellers, 2))  # Volume and price of offered goods
		my_state = tf.keras.Input(shape=(3, ))  # This buyers amount of rights, money and supply stored plus step num
		rights_offers = tf.keras.Input(shape=(args.num_buyers - 1, 2))  # What amount of rights at what price are the others offering

		flat_seller_offers = tf.keras.layers.Flatten()(seller_offers)
		hidden_sell = tf.keras.layers.Concatenate()([flat_seller_offers, my_state])
		#hidden_sell = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu', kernel_regularizer=l2(args.l2))(hidden_sell)
		hidden_sell = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu', kernel_regularizer=l2(args.l2))(hidden_sell)

		to_sell = tf.keras.layers.Dense(2, activation='sigmoid', kernel_regularizer=l2(args.l2))(hidden_sell)

		flat_rights_offers = tf.keras.layers.Flatten()(rights_offers)
		hidden_buy = tf.keras.layers.Concatenate()([flat_seller_offers, my_state, flat_rights_offers, to_sell])
		#hidden_buy = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu', kernel_regularizer=l2(args.l2))(hidden_buy)
		hidden_buy = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu', kernel_regularizer=l2(args.l2))(hidden_buy)

		to_buy = tf.keras.layers.Dense(4, activation='sigmoid')(hidden_buy)

		self.actor = tf.keras.Model(inputs={'supply_offers': seller_offers, 'my_state': my_state, 'rights_offered': rights_offers},
										   outputs={'sell': to_sell, 'buy': to_buy})

		self.actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, global_clipnorm=args.clip_norm))
		self.target_actor = tf.keras.models.clone_model(self.actor)

class Seller_Actor():
	def __init__(self, args):
		buyer_state = tf.keras.Input(shape=(args.num_buyers, 3))
		my_state = tf.keras.Input(shape=(1, ))

		flat_state = tf.keras.layers.Flatten()(buyer_state)
		hidden = tf.keras.layers.Concatenate()([flat_state, my_state])
		hidden = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu', kernel_regularizer=l2(args.l2))(hidden)
		#hidden = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu', kernel_regularizer=l2(args.l2))(hidden)
		to_sell = tf.keras.layers.Dense(2, activation='sigmoid')(hidden)  # Volume and price for supply to sell

		self.actor = tf.keras.Model(inputs={'b_state': buyer_state, 'my_state': my_state},
									outputs=to_sell)

		self.actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, global_clipnorm=args.clip_norm))
		self.target_actor = tf.keras.models.clone_model(self.actor)

class Buyer():
	def __init__(self, args, env, need, earning, id, actor):
		self.supply = args.buyer_starting_money
		self.money = args.buyer_starting_supply
		self.rights = 0
		self.frustration = 0
		self.total_frustration = 0

		self.need = need
		self.earning = earning
		self.args = args
		self.id = id

		self.actor = actor.actor
		self.target_actor = actor.target_actor
		self.get_critics(args, env)

	def train_actor(self, buyer_states, seller_states):
		seller_offers = seller_states[:, :, 2:4]

		my_state = buyer_states[:, self.id, :4]
		offered_rights = tf.concat([buyer_states[:, :self.id, 4:6], buyer_states[:, self.id+1:, 4:6]], axis=-2)

		seller_s = seller_states[..., 1:4]
		buyer_s = tf.concat([buyer_states[:, :self.id, 1:10], buyer_states[:, self.id+1:, 1:10]], axis=-2)

		with tf.GradientTape() as actor_tape_sell:
			action = self.actor({'supply_offers': seller_offers, 'my_state': my_state[..., 1:],
								 'rights_offered': offered_rights}, training=True)

			values = self.critic({'seller_states': seller_s, 'buyer_states': buyer_s, 'my_state': my_state,
									  'action_sell': action['sell'], 'action_buy': action['buy']}, training=True)
			actor_loss_sell = - tf.reduce_mean(values)
			for var in self.actor.trainable_variables:
				actor_loss_sell += self.args.l2 * tf.nn.l2_loss(var)

			actor_loss_sell += self.args.c_mean * tf.keras.losses.MeanSquaredError()(y_true=0.5, y_pred=action['sell'])
			actor_loss_sell += self.args.c_mean * tf.keras.losses.MeanSquaredError()(y_true=0.5, y_pred=action['buy'])

		self.actor.optimizer.minimize(actor_loss_sell, self.actor.trainable_variables, tape=actor_tape_sell)

		for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

		for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

	def train_critics(self, buyer_states, seller_states, next_buyer_states, next_seller_states):
		rewards = buyer_states[:, self.id, 10]
		dones = tf.cast(buyer_states[:, self.id, 0] >= 1., tf.float32)
		returns = rewards + self.args.gamma * (1-dones) * self.predict_value(next_buyer_states, next_seller_states)

		buy_actions = buyer_states[:, self.id, 6:10]
		sell_actions = buyer_states[:, self.id, 4:6]
		seller_s = seller_states[..., 1:4]
		buyer_s = tf.concat([buyer_states[..., :self.id, 1:10], buyer_states[..., self.id+1:, 1:10]], axis=-2)
		my_state = buyer_states[:, self.id, :4]

		with tf.GradientTape() as critic_tape:
			pred_values = self.critic({'seller_states': seller_s, 'buyer_states': buyer_s, 'my_state': my_state,
									   'action_sell': sell_actions, 'action_buy': buy_actions}, training=True)
			critic_loss = self.critic.compiled_loss(y_true=returns, y_pred=pred_values)
			for var in self.critic.trainable_variables:
				critic_loss += self.args.l2 * tf.nn.l2_loss(var)
		
		self.critic.optimizer.minimize(critic_loss, self.critic.trainable_variables, tape=critic_tape)

	def get_critics(self, args, env):
		seller_state = tf.keras.Input(shape=(args.num_sellers, 3))
		buyer_states = tf.keras.Input(shape=(args.num_buyers - 1, 9))
		my_state = tf.keras.Input(shape=(4,))
		action_sell = tf.keras.Input(shape=(2,))
		action_buy = tf.keras.Input(shape=(4,))

		flat_seller_offers = tf.keras.layers.Flatten()(seller_state)
		flat_rights_offers = tf.keras.layers.Flatten()(buyer_states)
		hidden = tf.keras.layers.Concatenate()([flat_seller_offers, flat_rights_offers, action_sell, action_buy, my_state])

		x = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(hidden)
		x = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(x)
		#x = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(x)

		value_1 = tf.keras.layers.Dense(1, activation='linear')(x)[:, 0]

		y = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(hidden)
		y = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(y)
		#y = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(y)

		value_2 = tf.keras.layers.Dense(1, activation='linear')(y)[:, 0]

		value = tf.minimum(value_1, value_2)

		self.critic = tf.keras.Model(inputs={'seller_states': seller_state, 'buyer_states': buyer_states,
											   'action_sell': action_sell, 'action_buy': action_buy, 'my_state': my_state},
									   outputs=value)
		self.critic.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, global_clipnorm=args.clip_norm),
							  loss=tf.keras.losses.MeanSquaredError())
		self.target_critic = tf.keras.models.clone_model(self.critic)

	def predict_sell_action(self, seller_offers, my_state):
		rights = tf.zeros((1, self.args.num_buyers - 1, 2))
		actions = self.target_actor({'supply_offers': seller_offers, 'my_state': my_state, 'rights_offered': rights})['sell']
		return actions

	def predict_buy_action(self, seller_offers, my_state, rights_offers):
		actions = self.target_actor({'supply_offers': seller_offers, 'my_state': my_state, 'rights_offered': rights_offers})['buy']
		return actions

	def predict_value(self, buyer_states, seller_states):
		my_state = buyer_states[:, self.id, :4]
		rights_offers = tf.concat([buyer_states[:, :self.id, 4:6], buyer_states[:, self.id+1:, 4:6]], axis=-2)
		seller_offers = seller_states[..., 2:4]

		actions = self.target_actor({'supply_offers': seller_offers, 'my_state': my_state[..., 1:],
									 'rights_offered': rights_offers})

		other_s_states = seller_states[..., 1:4]
		other_b_states = tf.concat([buyer_states[..., :self.id, 1:10], buyer_states[..., self.id+1:, 1:10]], axis=-2)
		values = self.target_critic({'seller_states': other_s_states, 'buyer_states': other_b_states,
									 'my_state': my_state, 'action_sell': actions['sell'], 'action_buy': actions['buy']})
		return values

	def on_episode_end(self, args):
		self.money = args.buyer_starting_money
		self.supply = args.buyer_starting_supply
		self.frustration = 0
		self.total_frustration = 0

	def save(self, id):
		self.actor.save_weights(f'b_actor_{id}.model')
		self.target_actor.save_weights(f'b_target_actor_{id}.model')
		self.critic.save_weights(f'b_critic_{id}.model')
		self.target_critic.save_weights(f'b_target_critic_{id}.model')

	def load(self, id):
		self.actor.load_weights(f'b_actor_{id}.model')
		self.target_actor.load_weights(f'b_target_actor_{id}.model')
		self.critic.load_weights(f'b_critic_{id}.model')
		self.target_critic.load_weights(f'b_target_critic_{id}.model')

class Seller():
	def __init__(self, args, env, id, actor):
		self.supply = args.seller_starting_money
		self.money = args.seller_starting_supply
		
		self.args = args
		self.id = id

		self.actor = actor.actor
		self.target_actor = actor.target_actor
		self.get_critics(args, env)

	def train_actor(self, buyer_states, seller_states):
		ac_b_state = buyer_states[..., 1:4]
		my_state = seller_states[:, self.id, :2]
		b_states = buyer_states[..., 1:10]
		s_states = tf.concat([seller_states[:, :self.id, 1:4], seller_states[:, self.id+1:, 1:4]], axis=-2)
		with tf.GradientTape() as actor_tape:
			action = self.actor({'b_state': ac_b_state, 'my_state': my_state[..., 1:]}, training=True)

			values = self.critic({'buyer_state': b_states, 'seller_state': s_states,
								  'action': action, 'my_state': my_state}, training=True)
			actor_loss = - tf.reduce_mean(values)
			for var in self.actor.trainable_variables:
				actor_loss += self.args.l2 * tf.nn.l2_loss(var)

			actor_loss += self.args.c_mean * tf.keras.losses.MeanSquaredError()(y_true=0.5, y_pred=action)

		self.actor.optimizer.minimize(actor_loss, self.actor.trainable_variables, tape=actor_tape)
		
		for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

		for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

	def train_critics(self, buyer_states, seller_states, next_buyer_states, next_seller_states):
		rewards = seller_states[:, self.id, 4]
		dones = tf.cast(seller_states[:, self.id, 0] >= 1., tf.float32)
		returns = rewards + self.args.gamma * (1-dones) * self.predict_value(next_buyer_states, next_seller_states)

		action = seller_states[:, self.id, 2:4]
		b_states = buyer_states[..., 1:10]
		s_states = tf.concat([seller_states[:, :self.id, 1:4], seller_states[:, self.id+1:, 1:4]], axis=-2)
		my_state = seller_states[:, self.id, :2]

		if False:
			tf.print()
			tf.print(returns)
			tf.print(rewards)
			tf.print(self.predict_value(buyer_states, seller_states))
			tf.print(self.predict_value(next_buyer_states, next_seller_states))

		with tf.GradientTape() as critic_tape:
			pred_values = self.critic({'buyer_state': b_states, 'seller_state': s_states,
									   'action': action, 'my_state': my_state}, training=True)
			critic_loss = self.critic.compiled_loss(y_true=returns, y_pred=pred_values)
			for var in self.critic.trainable_variables:
				critic_loss += self.args.l2 * tf.nn.l2_loss(var)
		
		self.critic.optimizer.minimize(critic_loss, self.critic.trainable_variables, tape=critic_tape)
	
	def get_critics(self, args, env):
		b_state = tf.keras.Input(shape=(args.num_buyers, 9))
		s_state = tf.keras.Input(shape=(args.num_sellers - 1, 3))
		my_state = tf.keras.Input(shape=(2,))
		action = tf.keras.Input(shape=(2,))

		flat_s_state = tf.keras.layers.Flatten()(s_state)
		flat_b_state = tf.keras.layers.Flatten()(b_state)
		hidden = tf.keras.layers.Concatenate()([flat_s_state, flat_b_state, action, my_state])
		x = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(hidden)
		x = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(x)
		#x = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(x)

		value_1 = tf.keras.layers.Dense(1, activation='linear')(x)[:, 0]

		y = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(hidden)
		y = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(y)
		#y = tf.keras.layers.Dense(args.buyer_hidden_layer_size, activation='relu')(y)

		value_2 = tf.keras.layers.Dense(1, activation='linear')(y)[:, 0]

		value = tf.minimum(value_1, value_2)

		self.critic = tf.keras.Model(inputs={'buyer_state': b_state, 'seller_state': s_state,
											 'action': action, 'my_state': my_state},
									 outputs=value)
		self.critic.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, global_clipnorm=args.clip_norm),
							  loss=tf.keras.losses.MeanSquaredError())
		self.target_critic = tf.keras.models.clone_model(self.critic)
	
	def predict(self, buyer_states, my_state):
		actions = self.target_actor({'b_state': buyer_states, 'my_state': my_state})
		return actions
	
	def predict_value(self, buyer_states, seller_states):
		my_state = seller_states[:, self.id, :2]
		actions = self.target_actor({'b_state': buyer_states[..., 1:4], 'my_state': my_state[..., 1:]})

		b_states = buyer_states[..., 1:10]
		s_states = tf.concat([seller_states[:, :self.id, 1:4], seller_states[:, self.id+1:, 1:4]], axis=-2)
		values = self.target_critic({'buyer_state': b_states, 'seller_state': s_states,
									 'action': actions, 'my_state': my_state})
		return values
	
	def on_episode_end(self, args):
		self.money, self.supply = args.seller_starting_money, args.seller_starting_money

	def save(self, id):
		self.actor.save_weights(f's_actor_{id}.model')
		self.target_actor.save_weights(f's_target_actor_{id}.model')
		self.critic.save_weights(f's_critic_{id}.model')
		self.target_critic.save_weights(f's_target_critic_{id}.model')

	def load(self, id):
		self.actor.load_weights(f's_actor_{id}.model')
		self.target_actor.load_weights(f's_target_actor_{id}.model')
		self.critic.load_weights(f's_critic_{id}.model')
		self.target_critic.load_weights(f's_target_critic_{id}.model')
