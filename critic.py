import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import keras

def get_critic(env, nb_actions, action_input):
    observation_input = tf.keras.Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32, kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = tf.keras.Model(inputs=[action_input, observation_input], outputs=x)

    return critic