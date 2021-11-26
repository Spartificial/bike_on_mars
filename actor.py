import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

def get_actor(env, nb_actions):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32, kernel_initializer='he_uniform'))
    actor.add(Activation('relu'))
    actor.add(Dense(32, kernel_initializer='he_uniform'))
    actor.add(Activation('relu'))
    actor.add(Dense(32, kernel_initializer='he_uniform'))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))

    return actor