from env_obstacle import CycleBalancingEnv
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.agents import DDPGAgent
import time

from tensorflow.keras.layers import *

from actor import get_actor
from critic import get_critic
from plot_moving_average import plot_moving_average

from tensorflow.keras.optimizers import Adam

import keras


env = CycleBalancingEnv()
env.reset()

states = env.observation_space.shape # Shape of our observation space
nb_actions = env.action_space.shape[0] # shape of our action space

# Getting our actor model for the DDPG algorithm
actor = get_actor(env, nb_actions)
print(actor.summary())

# Getting our critic network for the DDPG algorithm
action_input = Input(shape=(nb_actions,), name='action_input')
critic = get_critic(env, nb_actions, action_input)
print(critic.summary())

# Defining our DDPG agent
memory = SequentialMemory(limit=50000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta= 0.1, mu=0, sigma=.2)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, batch_size=1024,
                  memory=memory, nb_steps_warmup_critic=20, nb_steps_warmup_actor=20,
                  random_process=random_process, gamma=0.95, target_model_update=1e-3)

agent.compile([Adam(lr=.0001, clipnorm=1.0), Adam(lr=.001, clipnorm=1.0)], metrics=['mae'])

#Loading Weights
#agent.load_weights('ddpg_{}_weights.h5f'.format('32_3_rays_final'))

history = agent.fit(env, nb_steps=200, visualize=True, verbose=2, nb_max_episode_steps=1000)
episode_reward += history.history['episode_reward']

#Plot Episode reward
plt.plot(episode_reward)

#Ploting Moving Average of episode reward
avg_reward = plot_moving_average(episode_reward)
plt.plot(avg_reward)

#Saving Weights
# agent.save_weights('ddpg_{}_weights.h5f'.format('32_3_rays_final'), overwrite=True)