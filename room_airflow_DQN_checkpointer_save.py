#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:59:09 2021

@author: Hajime Ikeda

リフティング問題のDQNプログラム Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
を参考にしてプログラムを作成している。このプログラムは、Tf-agentsのドキュメントを参考にしている、

"""

import os
# import logger

import tensorflow as tf
from tensorflow import keras

from tf_agents.environments import suite_gym

#from tensorflow.keras import layers
#from tensorflow.keras import Input

from tf_agents.environments import gym_wrapper, py_environment, tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver

from tf_agents.policies import random_tf_policy

tf.compat.v1.enable_v2_behavior()

import numpy as np
import random
import gym
import envCFD

#ネットワーククラスの設定
# class MyQNetwork(network.QNetwork):
#   def __init__(self, observation_spec, action_spec, n_hidden_channels=50, name='QNetwork'):
#     super(MyQNetwork, self).__init__(
#       input_tensor_spec=observation_spec, 
#       state_spec=(), 
#       name=name
#     )
#     n_action = action_spec.maximum - action_spec.minimum + 1
#     self.model = keras.Sequential(
#       [
#         keras.layers.Dense(n_hidden_channels, activation='tanh'),
#         keras.layers.Dense(n_hidden_channels, activation='tanh'),
#         keras.layers.Dense(n_action),
#       ]
#     )
#   def call(self, observation, step_type=None, network_state=(), training=True):
#     actions = self.model(observation, training=training)
#     return actions, network_state


def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)





def train():

#環境の設定
  #env_py = suite_gym.load('FDM-v0')
  #env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env_py))
  train_py_env = suite_gym.load('FDM-v0')
  train_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(train_py_env))
  

  
  print('action_spec:', train_env .action_spec())
  print('time_step_spec.observation:', train_env .time_step_spec().observation)
  print('time_step_spec.step_type:', train_env .time_step_spec().step_type)
  print('time_step_spec.discount:', train_env .time_step_spec().discount)
  print('time_step_spec.reward:', train_env .time_step_spec().reward)
  
  print("create netwak")
#ネットワークの設定
  # primary_network = MyQNetwork(env.observation_spec(),  env.action_spec())
  
  q_net = q_network.QNetwork(train_env.observation_spec(),  train_env.action_spec(),fc_layer_params=(100,))
  print('train_env.observation_spec():', train_env.observation_spec())
  
  
  print("setting agent")
#エージェントの設定
  n_step_update = 1
  agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-3),
    n_step_update=n_step_update,
    epsilon_greedy=1.0,
    target_update_tau=1.0,
    target_update_period=100,
    gamma=0.99,
    td_errors_loss_fn = common.element_wise_squared_loss,
    train_step_counter = tf.Variable(0)
  )
  agent.initialize()
  agent.train = common.function(agent.train)
  
  print("setting action(policy)")
#行動の設定
  policy = agent.collect_policy
#データの保存の設定 
  print("setting replay buffer")
  # replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
  #   data_spec=agent.collect_data_spec,
  #   batch_size=env.batch_size,
  #   max_length=10**6
  # )
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
     data_spec=agent.collect_data_spec,
     batch_size=train_env.batch_size,
     max_length=10**6
   )  
  print("setting dataset")
  print("Timestep",train_env.time_step_spec())
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    sample_batch_size=128,
    num_steps=n_step_update+1
  ).prefetch(tf.data.experimental.AUTOTUNE)
  iterator = iter(dataset)
  
  print("setting env reset")
#事前データの設定
  train_env.reset()
  
  print("setting driver")
  # driver = dynamic_episode_driver.DynamicEpisodeDriver(
  #    train_env, 
  #    policy, 
  #    observers=[replay_buffer.add_batch], 
  #    num_episodes = 10, # 100
  #  )
  # driver.run(maximum_iterations=15)
  # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
  #                                               train_env.action_spec())
  
  # print("setting driver(collect data)")
  # collect_data(train_env, random_policy, replay_buffer, steps=100)

  num_episodes = 20 # 500
  
  epsilon = np.linspace(start=1.0, stop=0.0, num=num_episodes+1)
  
  print("setting policy saver")
  tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)#ポリシーの保存設定


  # make folder for saving the airflow temperature images
  os.mkdir('result')


  reward_list = []
  average_loss_list = []

  # for policy saver
  train_checkpointer = common.Checkpointer(
    ckpt_dir='checkpointer',
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=agent.train_step_counter
  )

  #train_checkpointer.initialize_or_restore()

  for episode in range(num_episodes+1):
    episode_rewards = 0#報酬の計算用
    episode_average_loss = []#lossの計算用
    policy._epsilon = epsilon[episode]#ランダム行動の確率
    time_step = train_env.reset()#環境の初期化
    #env_py.set_temperature_init()
    #train_env.set_temperature_init()
    train_py_env.set_temperature_init()
    
    reward_log = []

    calculating_episode = episode
    str_calculating_episode= str(calculating_episode).zfill(4)
    current_result_folder = 'result' + str_calculating_episode

    save_folder = 'result/' + current_result_folder
    print(current_result_folder)
    os.makedirs(save_folder)


    angle_fixed_cycles = 200
    number_of_action_chances = 20


    for action_chance in range(0,number_of_action_chances):

      # render 
      time = angle_fixed_cycles * action_chance
      time_limit_cycle = number_of_action_chances * angle_fixed_cycles
      
      calculating_episode = episode
      
      save_folder = 'result/' + current_result_folder
      train_py_env.save_fig(calculating_episode, save_folder, time, time_limit_cycle, reward_log)

      policy_step = policy.action(time_step)#状態から行動の決定

      next_time_step = train_env.step(policy_step.action)#行動による状態の遷移

      traj =  trajectory.from_transition(time_step, policy_step, next_time_step)#データの生成
      replay_buffer.add_batch(traj)#データの保存

      experience, _ = next(iterator)#学習用データの呼び出し
      loss_info = agent.train(experience=experience)#学習

      #R = next_time_step.reward.numpy().astype('int').tolist()[0]#報酬
      R = next_time_step.reward.numpy().astype('float').tolist()[0]#報酬
      episode_average_loss.append(loss_info.loss.numpy())#lossの計算
      episode_rewards += R#報酬の合計値の計算

      time_step = next_time_step#次の状態を今の状態に設定

      average_loss_list.append(loss_info.loss.numpy())
      reward_list.append(R)
                    

    
    print(f'Episode:{episode:4.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}, PE:{policy._epsilon:.6f}')

  tf_policy_saver.save(export_dir='policy')#ポリシーの保存

  # save checkpointer  
  train_checkpointer.save(global_step=agent.train_step_counter)

  train_env.close()
  
  return reward_list, average_loss_list
  
if __name__ == '__main__':
  reward, average_loss = train()
