import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from enum import Enum,IntEnum
from numba import jit

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


from tf_agents.trajectories import time_step as ts



class Actions(IntEnum):
    UP = 0
    STAY = 1
    DOWN = 2
    LEFT =3
    RIGHT = 4
    

class FDMEnv(gym.Env):


    def __init__(self):
        """
        OpenAI gymに有限差分法の環境を登録する
        
        """
        self.Re = 20
        self.Gr = 11.2
        self.Pr = 0.7
        self.dt = 0.01

        self.flag_hybrid_method = 1

        self.n_x = 32 
        self.n_y = 20
        self.n_z = 32


        ACTION_NUM = 5
        self.action_space = spaces.Discrete(ACTION_NUM) 

        """
        observation_high = np.array([
          np.finfo(np.float32).max,
          np.finfo(np.float32).max,
          np.finfo(np.float32).max,
          np.finfo(np.float32).max])

        self.observation_space = spaces.Box(-observation_high, observation_high)
        """
        observation_high = 1.0
        
        # 部屋の温度点数のすべてを状態として定義する。
        # self.observation_space = spaces.Box(low=-observation_high,high=observation_high,shape=(self.n_x * self.n_y * self.n_z,))

        #　床の温度分布を状態として定義する。
        self.observation_space = spaces.Box(low=-observation_high,high=observation_high,shape=(self.n_x * self.n_z,))
        # self.observation_space = spaces.MultiBinary(self.n_x * self.n_z,dtype=float32)
        

        self._seed()
        self._reset()

 

        # the inlet is exist (0,position_inlet) ~ (0,position_outlet)
        self.position_inlet_high = 16
        self.position_inlet_low = 15

        # the outlet is exist (0,position_outlet_low) ~ (0, position_outlet_high )
        self.position_outlet_high = 18 
        self.position_outlet_low = 17

        self.position_inlet_left = 14
        self.position_inlet_right = 19
        self.position_outlet_left = 14
        self.position_outlet_right = 19
        self.position_inlet_left

        self.room_length = 3.6
        self.room_height = 2.25
        self.room_width = 3.6


        self.max_iteration_pressure_calc = 2000

        self.velocity_mag = 4
        self.angle = 60
        self.delta_angle = 2
        
        #左右方向の風向　範囲は -90deg ~ +90deg
        self.angle_width = 0
        self.delta_angle_width = 4
        
        self.u_inlet = self.velocity_mag * np.cos(self.angle/180 * np.pi) * np.cos(self.angle_width/180 * np.pi)
        self.v_inlet = self.velocity_mag * (-1.0) * np.sin(self.angle/180 * np.pi)
        self.w_inlet = self.velocity_mag *  np.sin(self.angle_width/180 * np.pi)

        self.u_outlet = self.velocity_mag * (-1.0)
        self.v_outlet = self.velocity_mag * 0.0
        self.w_outlet = 0


        self.T_inlet = 0.0
        self.T_initial = 1.0
        #heat transfer rate of the walls
        self.h_wall = 0.01


        """
        Staggered grid
        number of u = (n_x +1) * n_y * n_z ->   0~(n_x) * 0~(n_y-1) * 0~(n_z-1)
        number of v = n_x * (n_y +1) * n_z ->   0~(n_x-1) * 0~(n_y) * 0~(n_z-1)
        number of w = n_x * n_y * (n_z +1) ->   0~(n_x-1) * 0~(n_y-1) * 0~(n_z)
        number of p = n_x * n_y * n_z      ->   0~(n_x-1) * 0~(n_y-1) * 0~(n_z-1)
        number of T = n_x * n_y * n_z      ->   0~(n_x-1) * 0~(n_y-1) * 0~(n_z-1)
        the outer side of pressure-point is virtual point for boundary condition
        """

        self.u = np.zeros((self.n_x+1, self.n_y, self.n_z))
        self.v = np.zeros((self.n_x, self.n_y+1, self.n_z))
        self.w = np.zeros((self.n_x, self.n_y, self.n_z+1))
        
        self.p = np.zeros((self.n_x  , self.n_y , self.n_z ))
        self.T = np.zeros((self.n_x  , self.n_y , self.n_z ))


        self.T[:][:][:] = self.T_initial
        #self.T_wall = 1.0

        self.u_new = np.zeros((self.n_x+1, self.n_y, self.n_z))
        self.v_new = np.zeros((self.n_x, self.n_y+1, self.n_z))
        self.w_new = np.zeros((self.n_x, self.n_y, self.n_z+1))
        
        self.p_new = np.zeros((self.n_x , self.n_y , self.n_z ))
        self.T_new = np.zeros((self.n_x , self.n_y , self.n_z ))

        self.u_at_pressure = np.zeros((self.n_x  , self.n_y ))
        self.v_at_pressure = np.zeros((self.n_x  , self.n_y ))
        self.w_at_pressure = np.zeros((self.n_x  , self.n_y ))
        
        
        self.x_at_pressure = np.zeros(self.n_x)
        self.y_at_pressure = np.zeros(self.n_y)
        self.z_at_pressure = np.zeros(self.n_z)

        #right-hand equation of poisson's equation
        self.right_hand_poisson= np.zeros((self.n_x ,self.n_y, self.n_z))

        #uniform mesh
        #x,y denotes pressure-points
        self.x = np.linspace(0,self.room_length, self.n_x )
        self.y = np.linspace(0,self.room_height, self.n_y )
        self.z = np.linspace(0,self.room_width, self.n_z )

        self.x_u = np.zeros(self.n_x +1)
        self.y_u = np.zeros(self.n_y )
        self.w_u = np.zeros(self.n_z )

        self.x_v = np.zeros(self.n_x)
        self.y_v = np.zeros(self.n_y+1)
        self.z_v = np.zeros(self.n_z)
        
        self.x_w = np.zeros(self.n_x)
        self.y_w = np.zeros(self.n_y)
        self.w_w = np.zeros(self.n_z+1)
        

        self.delta_x = self.x[1] - self.x[0]
        self.delta_y = self.y[1] - self.y[0]
        self.delta_z = self.z[1] - self.z[0]
        
        self.sensor_j=3
        self.T1_i = 15
        self.T2_i = 13
        self.T3_i = 11
                
        #objective points
        self.objective_j = 10
        self.O_low = 4
        self.O_high = 13

        """
        self.calculating_episode = calculating_episode
        str_calculating_episode = str(self.calculating_episode).zfill(4)
        self.current_result_folder = 'result' + str_calculating_episode
        os.mkdir(self.current_result_folder)
        """

        self.obserabation = 0
        
        # self.var_angle_for_state = var_angle_for_state
                            
        #PID control
        self.pid_kp = 10
        self.pid_ki = 10
        self.pid_ref = 1.0
        self.pid_observe = 0
        self.pid_error_p = 0.0 
        self.pid_error_i = 0.0
        self.pid_error_d = 0.0

        return

    #初期化
    # def _reset(self):
    
    #     init_state = np.zeros(self.n_x  * self.n_y * self.n_z )
    #     return init_state        
        
    def _reset(self):
        
        done = False
        info = {}
        
        init_state = np.zeros(self.n_x * self.n_z)
        
        discount = 0.1
        
        return discount, init_state, 0.0, done, info    
    
    
    def set_temperature_init(self):
        
        self.T[:][:][:] = self.T_initial
        self.u[:][:][:] = 0
        self.v[:][:][:] = 0
        self.w[:][:][:] = 0
        self.p[:][:][:] = 0
        
        return
        

    """
    def observation_spec(self):
        return self._observation_spec
  
    def action_spec(self):
        return self._action_spec
    """

    def _step(self,action):
        """
        

        """
        time_start, time_end = 0, 200
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        done = False
        info = []
        
        self.u_inlet, self.v_inlet, self.w_inlet, self.velocity_mag, self.angle,self.angle_width = self.set_airflow_control(self.angle,self.angle_width,self.velocity_mag,action)
        
        self.update_mac_staggerered_grid(action,time_start,time_end)
        
        # observation = self.judge_state(self.var_angle_for_state)
        observation = self.observe_temperature()

        reward = self.compute_reward()


        step_type = 0
        discount = 0.1
        
                
        #return ts.transition(observation, reward, self.done, info)

        return discount, observation, reward, done, info 



        
        
    #乱数の設定
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_saved_data_for_initial_condition(self):

        self.u = np.load('./saved/u.npy')
        self.v = np.load('./saved/v.npy')
        self.w = np.load('./saved/w.npy')
        self.T = np.load('./saved/temperature.npy')
        self.p = np.load('./saved/pressure.npy')

        return

    def solve_poisson_equation(self):

        error = 0.0

        for k in range(1,self.n_z-1):
            for j in range(1,self.n_y-1):
                for i in range(1,self.n_x-1):

                    p1 = (2/self.delta_x)*(2/self.delta_x) + (2/self.delta_y)*(2/self.delta_y) + (2/self.delta_z)*(2/self.delta_z)
                        
                    p2 = (self.p[i+1][j][k] + self.p[i-1][j][k]) /(self.delta_x * self.delta_x) + \
                        (self.p[i][j+1][k] + self.p[i][j-1][k]) /(self.delta_y * self.delta_y) + \
                            (self.p[i][j][k+1] + self.p[i][j][k-1]) /(self.delta_z * self.delta_z)

                    diff_p = (1/p1) * (p2 - self.right_hand_poisson[i][j][k]) - self.p[i][j][k]

                    error = error + diff_p * diff_p

                    self.p[i][j][k] = diff_p + self.p[i][j][k]

        return error

    @jit
    def solve_poisson_equation_fast(self):

        p1 = (2/self.delta_x)*(2/self.delta_x) + (2/self.delta_y)*(2/self.delta_y) + (2/self.delta_z)*(2/self.delta_z)

        p2 = np.zeros_like(self.p)
        diff_p = np.zeros_like(self.p) 
        error_p = np.zeros_like(self.p)
        error = 0.0

        p2[1:-2,1:-2,1:-2] = (self.p[2:-1,1:-2,1:-2] + self.p[0:-3,1:-2,1:-2]) /(self.delta_x * self.delta_x) + \
            (self.p[1:-2,2:-1,1:-2] + self.p[1:-2,0:-3,1:-2]) /(self.delta_y * self.delta_y) + \
                (self.p[1:-2,1:-2,2:-1] + self.p[1:-2,1:-2,0:-3]) /(self.delta_z * self.delta_z)


        diff_p[1:-2,1:-2,1:-2] = (1/p1) * (p2[1:-2,1:-2,1:-2] - self.right_hand_poisson[1:-2,1:-2,1:-2]) - self.p[1:-2,1:-2,1:-2]

        error_p[1:-2,1:-2,1:-2] =  diff_p[1:-2,1:-2,1:-2] * diff_p[1:-2,1:-2,1:-2]


        self.p[1:-2,1:-2,1:-2] = diff_p[1:-2,1:-2,1:-2] + self.p[1:-2,1:-2,1:-2]

        error = np.sum(error_p)

        return  error 



    def solve_pressure_equation_staggeredgrid(self):

        # print("poisson equation calculation")

        # right hand of pressure equation        
        '''
        for i in range(1,self.n_x-1):
            for j in range(1,self.n_y-1):
                for k in range(1,self.n_z-1):

                    #p1=(u_x)^2
                    p1 = ((self.u[i+1][j][k] - self.u[i][j][k]) / (self.delta_x)) *  \
                    ((self.u[i+1][j][k] - self.u[i][j][k]) /(self.delta_x))
    
                    #p2=(v_y)^2
                    p2 = ((self.v[i][j+1][k] - self.v[i][j][k]) / (self.delta_y)) * \
                    ((self.v[i][j+1][k] - self.v[i][j][k]) / (self.delta_y))
    
                    #p3=(w_z)^2
                    p3 = ((self.w[i][j][k+1] - self.w[i][j][k]) / (self.delta_z))* \
                    ((self.w[i][j][k+1] - self.w[i][j][k]) / (self.delta_z))
    
                    #p4 = (v_x) * (u_y) 
                    p4 = (self.v[i+1][j][k] - self.v[i][j][k]) / (self.delta_x) * \
                    (self.u[i][j+1][k] - self.u[i][j][k]) / (self.delta_y) 
                    
                    #p5 = (v_z)*(w_y)
                    p5 = (self.w[i][j+1][k] - self.w[i][j][k]) /(self.delta_y) * \
                    (self.v[i][j][k+1]  - self.v[i][j][k]) / (self.delta_z) 
                                
                    #p6 = (w_x)*(u_z)
                    p6 = (self.u[i][j][k+1] - self.u[i][j][k]) /(self.delta_z) * \
                    (self.w[i+1][j][k] - self.w[i][j][k]) / (self.delta_x) \

                    #(u_x + v_y + w_z) / dt                                      
                    p7 = ((self.u[i+1][j][k] - self.u[i][j][k]) / (self.delta_x) +  \
                    (self.v[i][j+1][k] - self.v[i][j][k]) / (self.delta_y) + \
                    (self.w[i][j][k+1] - self.w[i][j][k]) / (self.delta_z)) \
                    / self.dt
                        
    
                    #T1 = (T_y)                        
                    #T1 = (self.T[i][j+1][k] - self.T[i][j][k]) / self.delta_y
                    T_at_v2 = (self.T[i][j+1][k] + self.T[i][j][k]) /2
                    T_at_v1 = (self.T[i][j][k] + self.T[i][j-1][k]) /2
                    
                    #T1 = (T_y) 
                    T1 = (T_at_v2  - T_at_v1) / self.delta_z
    
                    self.right_hand_poisson[i][j][k] = -1.0 * (p1 + p2 + p3  + p3 + 2.0 *(p4  +p5 +p6))  + p7 \
                    + (self.Gr / (self.Re * self.Re)) * T1

        '''

        #p1 = np.zeros((self.n_x+1,self.n_y+1,self.n_z+1))
        p1 = np.zeros_like(self.right_hand_poisson)
        p2 = np.zeros_like(self.right_hand_poisson)
        p3 = np.zeros_like(self.right_hand_poisson)
        p4 = np.zeros_like(self.right_hand_poisson)
        p5 = np.zeros_like(self.right_hand_poisson)
        p6 = np.zeros_like(self.right_hand_poisson)
        p7 = np.zeros_like(self.right_hand_poisson)
        T1 = np.zeros_like(self.T)
        T_at_v2 = np.zeros_like(self.T)
        T_at_v1 = np.zeros_like(self.T)

        #p1=(u_x)^2
        p1[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ((self.u[2:self.n_x, 1:self.n_y-1,1:self.n_z-1] - self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_x)) * \
             ((self.u[2:self.n_x,1:self.n_y-1,1:self.n_z-1]  - self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) /(self.delta_x))
    
        #p2=(v_y)^2
        p2[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ((self.v[1:self.n_x-1, 2:self.n_y,1:self.n_z-1] - self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_y)) * \
             ((self.v[1:self.n_x-1,2:self.n_y,1:self.n_z-1]  - self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_y))
    
        #p3=(w_z)^2
        p3[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ((self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_z)) * \
             ((self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_z))
    
        #p4 = (v_x) * (u_y) 
        p4[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.v[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_x) * \
        (self.u[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_y) 
                    
        #p5 = (v_z)*(w_y)
        p5[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.w[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) /(self.delta_y) * \
            (self.v[1:self.n_x-1,1:self.n_y-1,2:self.n_z]  - self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_z) 
                                
        #p6 = (w_x)*(u_z)
        p6[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.u[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) /(self.delta_z) * \
            (self.w[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_x) 

        #(u_x + v_y + w_z) / dt                                      
        p7[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ((self.u[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_x) +  \
            (self.v[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_y) + \
                (self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]) / (self.delta_z)) \
                    / self.dt

        #T1 = (T_y)                        
        T_at_v2[1:-2,1:-2,1:-2] = (self.T[1:-2,2:-1,1:-2] + self.T[1:-2,1:-2,1:-2]) /2
        T_at_v1[1:-2,1:-2,1:-2] = (self.T[1:-2,1:-2,1:-2] + self.T[1:-2,0:-3,1:-2]) /2
                    
        #T1 = (T_y) 
        T1[1:-2,1:-2,1:-2] = (T_at_v2[1:-2,1:-2,1:-2]  - T_at_v1[1:-2,1:-2,1:-2]) / self.delta_z
    
        self.right_hand_poisson[1:-2,1:-2,1:-2] = -1.0 * (p1[1:-2,1:-2,1:-2] + p2[1:-2,1:-2,1:-2] + p3[1:-2,1:-2,1:-2]  + p3[1:-2,1:-2,1:-2] + 2.0 *(p4[1:-2,1:-2,1:-2]  +p5[1:-2,1:-2,1:-2] +p6[1:-2,1:-2,1:-2]))  + p7[1:-2,1:-2,1:-2] \
            + (self.Gr / (self.Re * self.Re)) * T1[1:-2,1:-2,1:-2] 
                    

        for iteration_pressure in range(0,self.max_iteration_pressure_calc):

            error = 0.0

            #boundary conditions for pressure and temperature
            
            #boundary condition for Y-Z plane
            for k in range(1,self.n_z-1):
                for j in range(1,self.n_y-1):
                    
                    u_c = self.u[2][j][k]
                    u_b = self.u[1][j][k]
                    u_a = self.u[0][j][k]
                    delta_x = self.x[1] - self.x[0]                    
                    self.p[0][j][k] = self.p[1][j][k] + (1/self.Re) * (u_c -2 * u_b + u_a) / delta_x
                                        
                    u_c = self.u[self.n_x-2][j][k]
                    u_b = self.u[self.n_x-1][j][k]
                    u_a = self.u[self.n_x][j][k]
                    delta_x = self.x[1] - self.x[0]                    
                    self.p[self.n_x-1][j][k] = self.p[self.n_x-2][j][k] + (1/self.Re) * (u_c -2 * u_b + u_a) / delta_x
                                        
#                   # Adiabatic wall condition
                    #self.T[0][j][k] = self.T[1][j][k]
                    #self.T[self.n_x-1][j][k] = self.T[self.n_x-2][j][k]

                    # heat transfer condition
                    self.T[0][j][k] = self.T[1][j][k] + self.h_wall * self.delta_x
                    self.T[self.n_x-1][j][k] = self.T[self.n_x-2][j][k] + self.h_wall * self.delta_x



                    #Isothermal wall condition
#                    self.T[0][j][k] = 0
#                    self.T[self.n_x-1][j][k] = 0
                    
            #boundary condition for X-Y plane
            for j in range(1,self.n_y-1):
                for i in range(1,self.n_x-1):

                    w_c = self.w[i][j][2]
                    w_b = self.w[i][j][1]
                    w_a = self.w[i][j][0]
                    delta_z = self.z[1] - self.z[0]
                    self.p[i][j][0] = self.p[i][j][1] +\
                            (1/self.Re) * (w_c -2 * w_b + w_a) / delta_z
                    
                    w_c = self.w[i][j][self.n_z-2]
                    w_b = self.w[i][j][self.n_z-1]
                    w_a = self.w[i][j][self.n_z]
                    delta_z = self.z[1] - self.z[0]
                    self.p[i][j][self.n_z-1] = self.p[i][j][self.n_z-2] +\
                            (1/self.Re) * (w_c -2 * w_b + w_a) / delta_z
                                                                          
#                    #Adiabatic wall condition
#                    self.T[i][j][0] = self.T[i][j][1]
#                    self.T[i][j][self.n_z-1] = self.T[i][j][self.n_z-2]


                    # heat transfer condition
                    self.T[i][j][0] = self.T[i][j][1] + self.h_wall * self.delta_y
                    self.T[i][j][self.n_z-1] = self.T[i][j][self.n_z-2] + + self.h_wall * self.delta_x


                    #Isothermal wall condition
#                    self.T[i][j][0] = 0
#                    self.T[i][j][self.n_z-1] = 0
            
            #boundary condition for X-Z plane
            for k in range(1,self.n_z):
                for i in range(1,self.n_x):

                    v_c = self.v[i][2][k]
                    v_b = self.v[i][1][k]
                    v_a = self.v[i][0][k]
                    delta_y = self.y[1] - self.y[0]
                    self.p[i][0][k] = self.p[i][1][k] + (1/self.Re) * (v_c -2 * v_b + v_a) / delta_y
                    
                    v_c = self.v[i][self.n_y-2][k]
                    v_b = self.v[i][self.n_y-1][k]
                    v_a = self.v[i][self.n_y][k]
                    delta_y = self.y[1] - self.y[0]
                    self.p[i][self.n_y-1][k] = self.p[i][self.n_y-2][k] + (1/self.Re) * (v_c -2 * v_b + v_a) / delta_y

                    
                    #Isothermal wall condition
#                    self.T[i][0][k] = self.T_bottom_wall
#                    self.T[i][self.n_y-1][k] = self.T_top_wall
                    
#                    #Adiabatic wall condition
                    #self.T[i][0][k] = self.T[i][1][k]
                    #self.T[i][self.n_y-1][k] = self.T[i][self.n_y-2][k]

                    # heat transfer condition
                    self.T[i][0][k] = self.T[i][1][k] + self.h_wall * self.delta_z
                    self.T[i][self.n_y-1][k] = self.T[i][self.n_y-2][k] + self.h_wall * self.delta_z


            #Inlet Condition
            #print('Inlet condition_0')
            for k in range(self.position_inlet_left,self.position_inlet_right):
                for j in range(self.position_inlet_low,self.position_inlet_high):
                    # ∂p/∂x = 0
                    self.p[0][j][k] = self.p[1][j][k] 
                    #print('T_inlet_1')
                    self.T[0][j][k] = self.T_inlet 
                    
                        
            #Outlet pressure is 0 (fixed)
            #Outlet pressure is temeratue is same as connected node
            for k in range(self.position_outlet_left,self.position_outlet_right):    
                for j in range(self.position_outlet_low,self.position_outlet_high):
                    self.p[0][j][k] = 0
                    self.T[0][j][k] = self.T[1][j][k]
                    
        
            #error = self.solve_poisson_equation()
            error = self.solve_poisson_equation_fast()
            
            #Outlet pressure is 0 (fixed)
            #Outlet pressure is temeratue is same as connected node
            for k in range(self.position_outlet_left,self.position_outlet_right):    
                for j in range(self.position_outlet_low,self.position_outlet_high):
                    self.p[0][j][k] = 0
                    #print('T_outlet_1')
                    self.T[0][j][k] = self.T[1][j][k]
                    

            if (error < 0.000001):
                # print('pressure is converged cycle_poisson=',k)
                break

        return
    
    

    def update_mac_staggerered_grid(self, action, time_start, time_end):


        for i_time in range(time_start, time_end):
            
            
            #u,v,wの境界条件
            #boundary condition for X-Z plane
            for k in range(0,self.n_z):
                for i in range(0,self.n_x):
                    self.v[i][self.n_y-1][k] = 0 
                    self.v[i][0][k] = 0.0  
                                                                             
            for k in range(0,self.n_z+1):
                for i in range(0,self.n_x):
                    self.w[i][self.n_y-1][k] = -1.0 * self.w[i][self.n_y-2][k]
                    self.w[i][0][k] = -1.0 * self.w[i][1][k] 
                                
            for k in range(0,self.n_z):
                for i in range(0,self.n_x+1):
                    self.u[i][self.n_y-1][k] = -1.0 * self.u[i][self.n_y-2][k]
                    self.u[i][0][k] = -1.0 * self.u[i][1][k]
                                        
                                                            
            #boundary condition for Y-Z plane
            for k in range(0,self.n_z):
                for j in range(0,self.n_y):
                    self.u[self.n_x-1][j][k] = 0.0
                    self.u[0][j][k] = 0.0
                                                            
            for k in range(0,self.n_z+1):
                for j in range(0,self.n_y):
                    self.w[self.n_x-1][j][k] = -1.0 * self.w[self.n_x-2][j][k] 
                    self.w[0][j][k] = -1.0 * self.w[1][j][k] 
                    
            for k in range(0,self.n_z):
                for j in range(0,self.n_y+1):
                    self.v[self.n_x-1][j][k] = -1.0 * self.v[self.n_x-2][j][k]
                    self.v[0][j][k] = -1.0 * self.v[1][j][k]                    


            #boundary condition for X-Y plane
            for j in range(0,self.n_y):
                for i in range(0,self.n_x):                    
                    self.w[i][j][0] = 0
                    self.w[i][j][self.n_z-1] = 0.0                    
                                                                                
            for j in range(0,self.n_y+1):
                for i in range(0,self.n_x):
                    self.v[i][j][self.n_z-1] = -1.0 * self.v[i][j][self.n_z-2]
                    self.v[i][j][0] = -1.0 * self.v[i][j][1]
                    
            for j in range(0,self.n_y):
                for i in range(0,self.n_x+1):
                    self.u[i][j][self.n_z-1] = -1.0 * self.u[i][j][self.n_z-2]
                    self.u[i][j][0] = -1.0 * self.u[i][j][1]
                    
                                                    
            #inlet condition
            for k in range(self.position_inlet_left,self.position_inlet_right): 
                for j in range(self.position_inlet_low, self.position_inlet_high):
                    self.u[1][j][k] = self.u_inlet
                    self.v[1][j+1][k] = self.v_inlet
                    self.w[1][j][k] = 0
                    #print('T_inlet_2')
                    self.T[0][j][k] = self.T_inlet

            #outlet condition
            for k in range(self.position_outlet_left,self.position_outlet_right): 
                for j in range(self.position_outlet_low, self.position_outlet_high):
                    self.u[1][j][k] = self.u_outlet
                    self.v[1][j+1][k] = self.v_outlet
                    self.w[1][j][k] = 0

            #v_new,u_newに境界条件をコピーしておかないと、1回目の　dtの計算で境界条件が消える
            self.u_new = self.u
            self.v_new = self.v
            self.w_new = self.w

            #Solve a Poisson equation
            self.solve_pressure_equation_staggeredgrid()

            #Update velocity
            #Update u[][]
            #range()の範囲が未検証（20190131）
            '''
            for i in range(1,self.n_x-1):
                for j in range(1,self.n_y-1):
                    for k in range(1,self.n_z-1):

                        uu_x = self.u[i][j][k] * ((self.u[i+1][j][k] - self.u[i-1][j][k]) / (2 * self.delta_x))
                        
                        vu_y = (self.v[i-1][j][k] + self.v[i][j][k] + self.v[i-1][j+1][k] + self.v[i][j+1][k])/4 * \
                        (self.u[i][j+1][k] - self.u[i][j-1][k]) /(2 * self.delta_y)
                        
                        wu_z = (self.w[i-1][j][k] + self.w[i][j][k] + self.w[i-1][j][k+1] + self.w[i][j][k+1])/4 * \
                        (self.u[i][j][k+1] - self.u[i][j][k-1]) /(2 * self.delta_z)
                                                
                        p_x = (self.p[i][j][k] - self.p[i-1][j][k]) / self.delta_x
    
                        laplacian_u = ((self.u[i+1][j][k] - 2*self.u[i][j][k]+ self.u[i-1][j][k]) / (self.delta_x * self.delta_x) + \
                              (self.u[i][j+1][k] - 2*self.u[i][j][k]+ self.u[i][j-1][k])/(self.delta_y * self.delta_y) + \
                              (self.u[i][j][k+1] - 2*self.u[i][j][k]+ self.u[i][j][k-1])/(self.delta_z * self.delta_z)) 
    
                        self.u_new[i][j][k] = self.u[i][j][k] + self.dt * (- uu_x - vu_y - wu_z - p_x + laplacian_u / self.Re)
                        
                        
                        # if  inlet or outlet, u[i][j] do not change.
                        if (i == 1) and ((j>=self.position_inlet_low) and (j<=self.position_inlet_high) and \
                                (k>=self.position_inlet_left) and (k<=self.position_inlet_right) ) :
                            self.u_new[i][j][k] = self.u_inlet
                    
                        if (i == 1) and ( (j>=self.position_outlet_low) and (j<=self.position_outlet_high) and \
                                (k>=self.position_inlet_left) and (k<=self.position_inlet_right) ) :                                                         
                            self.u_new[i][j][k] = self.u_outlet
            '''

            uu_x = np.zeros_like(self.u)
            vu_y = np.zeros_like(self.u)
            wu_z = np.zeros_like(self.u)
            p_x = np.zeros_like(self.u)
            laplacian_u = np.zeros_like(self.u)
                        
            uu_x[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] * ((self.u[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - self.u[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1]) / (2 * self.delta_x))
                        
            vu_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.v[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1] + self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.v[0:self.n_x-2,2:self.n_y,1:self.n_z-1] + self.v[1:self.n_x-1,2:self.n_y,1:self.n_z-1])/4 * \
                (self.u[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - self.u[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1]) /(2 * self.delta_y)
                        
            wu_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.w[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1] + self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.w[0:self.n_x-2,1:self.n_y-1,2:self.n_z] + self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z] )/4 * \
                (self.u[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.u[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2] ) /(2 * self.delta_z)
                                                
            p_x[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.p[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] - self.p[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1] ) / self.delta_x
            
            laplacian_u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  = ((self.u[2:self.n_x,1:self.n_y-1,1:self.n_z-1]  - 2*self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.u[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1] ) / (self.delta_x * self.delta_x) + \
                (self.u[1:self.n_x-1,2:self.n_y,1:self.n_z-1]  - 2*self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.u[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1] )/(self.delta_y * self.delta_y) + \
                    (self.u[1:self.n_x-1,1:self.n_y-1,2:self.n_z]  - 2*self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.u[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2] )/(self.delta_z * self.delta_z)) 

            self.u_new[1:-2,1:-2,1:-2] = self.u[1:-2,1:-2,1:-2] + self.dt * (- uu_x[1:-2,1:-2,1:-2] - vu_y[1:-2,1:-2,1:-2] - wu_z[1:-2,1:-2,1:-2] - p_x[1:-2,1:-2,1:-2] + laplacian_u[1:-2,1:-2,1:-2] / self.Re)


        
            # if  inlet or outlet, u[i][j] do not change.
            self.u_new[1,self.position_inlet_low:self.position_inlet_high+1,self.position_inlet_left:self.position_inlet_right+1] = self.u_inlet
            self.u_new[1,self.position_outlet_low:self.position_outlet_high+1,self.position_outlet_left:self.position_outlet_right+1] = self.u_outlet



            #Update v[][]
            #range()の範囲が未検証（20190131）
            '''
            for i in range(1,self.n_x-1):
                for j in range(1,self.n_y-1):
                    for k in range(1,self.n_z-1):
                        
                        uv_x = (self.u[i][j-1][k] + self.u[i][j][k] + self.u[i+1][j-1][k] + self.u[i+1][j][k])/4 * \
                        (self.v[i+1][j][k] - self.v[i-1][j][k]) /(2 * self.delta_x)
                                                                            
                        vv_y = self.v[i][j][k] * ((self.v[i][j+1][k] - self.v[i][j-1][k]) /  (2 * self.delta_y))

                        wv_z = (self.w[i][j-1][k] + self.w[i][j][k] + self.w[i][j-1][k+1] + self.w[i][j][k+1])/4 * \
                        (self.v[i][j][k+1] - self.v[i][j][k-1]) /(2 * self.delta_z)
        
                        p_y =  (self.p[i][j][k] - self.p[i][j-1][k]) / self.delta_y
    
                        laplacian_v = ((self.v[i+1][j][k] - 2 *self.v[i][j][k] + self.v[i-1][j][k]) / (self.delta_x * self.delta_x) + \
                        (self.v[i][j+1][k] - 2*self.v[i][j][k] + self.v[i][j-1][k]) / (self.delta_y * self.delta_y) + \
                        (self.v[i][j][k+1] - 2*self.v[i][j][k]+ self.v[i][j][k-1])/(self.delta_z * self.delta_z))
                        
                        T_force = (self.T[i][j][k] + self.T[i][j-1][k]) * 0.5 * (self.Gr/(self.Re * self.Re))
                    
                        self.v_new[i][j][k] = self.v[i][j][k] + self.dt * (- vv_y -uv_x - wv_z -p_y + laplacian_v /self.Re + T_force)
                                                
                        # if  inlet or outlet, u[i][j] do not change.
                        if (i == 1) and ((j>=self.position_inlet_low) and (j<=self.position_inlet_high) and\
                                (k>=self.position_inlet_left) and (k<=self.position_inlet_right) ) :
                            self.v_new[i][j][k] = self.v_inlet
                    
                        if (i == 1) and ( (j>=self.position_outlet_low) and (j<=self.position_outlet_high) and\
                                (k>=self.position_inlet_left) and (k<=self.position_inlet_right) ) :
                                                         self.v_new[i][j][k] = self.v_outlet
            '''

            uv_x = np.zeros_like(self.v)
            vv_y = np.zeros_like(self.v)
            wv_z = np.zeros_like(self.v)
            p_y = np.zeros_like(self.v)
            laplacian_v = np.zeros_like(self.v)
            T_force =  np.zeros_like(self.v)



            uv_x[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.u[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1] + self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.u[2:self.n_x,0:self.n_y-2,1:self.n_z-1] + self.u[2:self.n_x,1:self.n_y-1,1:self.n_z-1] )/4 * \
                (self.v[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - self.v[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1] ) /(2 * self.delta_x)
                                                                            
            vv_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] * ((self.v[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - self.v[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1]) /  (2 * self.delta_y))

            wv_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.w[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1] + self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.w[1:self.n_x-1,0:self.n_y-2,2:self.n_z]+ self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z] )/4 * \
                        (self.v[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.v[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2]) /(2 * self.delta_z)
        
            p_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] =  (self.p[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] - self.p[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1]) / self.delta_y
    
            laplacian_v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ((self.v[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - 2 *self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.v[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1]) / (self.delta_x * self.delta_x) + \
                (self.v[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - 2*self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.v[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1]) / (self.delta_y * self.delta_y) + \
                    (self.v[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - 2*self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]+ self.v[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2])/(self.delta_z * self.delta_z))
                        
            T_force[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.T[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.T[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1]) * 0.5 * (self.Gr/(self.Re * self.Re))
                    
            self.v_new[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  = self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  + self.dt * (- vv_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  -uv_x[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  - wv_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] -p_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  + laplacian_v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  /self.Re + T_force[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] )


            self.v_new[1,self.position_inlet_low:self.position_inlet_high+1,self.position_inlet_left:self.position_inlet_right+1] = self.v_inlet
            self.v_new[1,self.position_outlet_low:self.position_outlet_high+1,self.position_outlet_left:self.position_outlet_right+1] = self.v_outlet
                                                


                        
                        
            #update v
            '''                
            for i in range(1,self.n_x-1):
                for j in range(1,self.n_y-1):
                    for k in range(1,self.n_z-1):
                        
                        uw_x = (self.u[i][j][k-1] + self.u[i][j][k] + self.u[i+1][j][k-1] + self.u[i][j+1][k])/4 * \
                        (self.w[i+1][j][k] - self.w[i-1][j][k]) /(2 * self.delta_x)
                        
                        vw_y = (self.v[i][j][k-1] + self.v[i][j][k] + self.v[i][j+1][k-1] + self.v[i][j+1][k])/4 * \
                        (self.w[i][j+1][k] - self.w[i][j-1][k]) /(2 * self.delta_y)
                                                    
                        ww_z = self.w[i][j][k]  * (self.w[i][j][k+1] - self.w[i][j][k-1]) / (2 * self.delta_z) 
        
                        p_z = (self.p[i][j][k] - self.p[i][j][k-1]) / self.delta_z
    
                        laplacian_w = ((self.w[i+1][j][k] - 2 *self.w[i][j][k] + self.w[i-1][j][k]) / (self.delta_x * self.delta_x) + \
                        (self.w[i][j+1][k] - 2*self.w[i][j][k] + self.w[i][j-1][k]) / (self.delta_y * self.delta_y) + \
                        (self.w[i][j][k+1] - 2*self.w[i][j][k]+ self.w[i][j][k-1])/(self.delta_z * self.delta_z))
        
                        self.w_new[i][j][k] = self.w[i][j][k] + self.dt * (- uw_x -vw_y - ww_z -p_z  + laplacian_w /self.Re)

                        # if  inlet or outlet, u[i][j] do not change.
                        if (i == 1) and ((j>=self.position_inlet_low) and (j<=self.position_inlet_high) and\
                                (k>=self.position_inlet_left) and (k<=self.position_inlet_right) ) :
                            self.w_new[i][j][k] = 0
                    
                        if (i == 1) and ( (j>=self.position_outlet_low) and (j<=self.position_outlet_high) and\
                                (k>=self.position_inlet_left) and (k<=self.position_inlet_right) ) :
                                                         self.w_new[i][j][k] = 0                       
            '''

            uw_x = np.zeros_like(self.w)
            vw_y = np.zeros_like(self.w)
            ww_z = np.zeros_like(self.w)
            p_z = np.zeros_like(self.w)
            laplacian_w = np.zeros_like(self.w)


            uw_x[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.u[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2] + self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.u[2:self.n_x,1:self.n_y-1,0:self.n_z-2] + self.u[1:self.n_x-1,2:self.n_y,1:self.n_z-1])/4 * \
                (self.w[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - self.w[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1]) /(2 * self.delta_x)
                        
            vw_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.v[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2] + self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.v[1:self.n_x-1,2:self.n_y,0:self.n_z-2] + self.v[1:self.n_x-1,2:self.n_y,1:self.n_z-1])/4 * \
                        (self.w[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - self.w[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1]) /(2 * self.delta_y)
                                                    
            ww_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  * (self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.w[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2]) / (2 * self.delta_z) 
        
            p_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.p[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] - self.p[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2]) / self.delta_z
    
            laplacian_w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ((self.w[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - 2 *self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.w[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1]) / (self.delta_x * self.delta_x) + \
                (self.w[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - 2*self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.w[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1]) / (self.delta_y * self.delta_y) + \
                    (self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - 2*self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]+ self.w[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2])/(self.delta_z * self.delta_z))
        
            self.w_new[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  = self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  + self.dt * (- uw_x[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  -vw_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  - ww_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  -p_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]   + laplacian_w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1]  /self.Re)

            self.w_new[1,self.position_inlet_low:self.position_inlet_high+1,self.position_inlet_left:self.position_inlet_right+1] = 0
            self.w_new[1,self.position_outlet_low:self.position_outlet_high+1,self.position_outlet_left:self.position_outlet_right+1] = 0




                        
            #Temperature
            '''
            for j in range(1,self.n_y-1):
                for i in range(1,self.n_x-1):
                    for k in range(1,self.n_z-1):
    
                        U_at_Tijk = (self.u[i][j][k] + self.u[i+1][j][k])/2
                        T_1 = self.T[i+1][j][k]
                        T_2 = self.T[i-1][j][k]
                        T_x = (T_1 - T_2)/(2 * self.delta_x)
                        
                        V_at_Tijk = (self.v[i][j][k] + self.v[i][j+1][k])/2
                        T_1 = self.T[i][j+1][k]
                        T_2 = self.T[i][j-1][k]
                        T_y = (T_1 - T_2)/(2*self.delta_y)
                        
                        W_at_Tijk = (self.w[i][j][k] + self.w[i][j][k+1])/2
                        T_1 = self.T[i][j][k+1]
                        T_2 = self.T[i][j][k-1]                        
                        T_z = (T_1 - T_2)/(2*self.delta_z)
        
                        laplacian_T = (self.T[i+1][j][k] - 2.0* self.T[i][j][k] + self.T[i-1][j][k]) / (self.delta_x * self.delta_x) \
                        + (self.T[i][j+1][k] - 2.0* self.T[i][j][k] + self.T[i][j-1][k]) / (self.delta_y * self.delta_y) \
                        + (self.T[i][j][k+1] - 2.0* self.T[i][j][k] + self.T[i][j][k-1]) / (self.delta_z * self.delta_z)
    
                        self.T_new[i][j][k] = self.T[i][j][k] + self.dt * (-1.0 * (U_at_Tijk  * T_x + V_at_Tijk  * T_y  + W_at_Tijk  * T_z ) + laplacian_T/(self.Re * self.Pr))
            

            '''
            U_at_Tijk = np.zeros_like(self.T)
            T_x = np.zeros_like(self.T)
            V_at_Tijk = np.zeros_like(self.T)
            T_y = np.zeros_like(self.T)
            W_at_Tijk = np.zeros_like(self.T)
            T_z = np.zeros_like(self.T)
            laplacian_T = np.zeros_like(self.T)

            vw_y = np.zeros_like(self.w)
            ww_z = np.zeros_like(self.w)

            U_at_Tijk[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.u[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.u[2:self.n_x,1:self.n_y-1,1:self.n_z-1])/2
            T_x[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ( self.T[2:self.n_x,1:self.n_y-1,1:self.n_z-1] - self.T[0:self.n_x-2,1:self.n_y-1,1:self.n_z-1])/(2 * self.delta_x)


            V_at_Tijk[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.v[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.v[1:self.n_x-1,2:self.n_y,1:self.n_z-1])/2
            T_y[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ( self.T[1:self.n_x-1,2:self.n_y,1:self.n_z-1] - self.T[1:self.n_x-1,0:self.n_y-2,1:self.n_z-1])/(2 * self.delta_y)
                        

            W_at_Tijk[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = (self.w[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] + self.w[1:self.n_x-1,1:self.n_y-1,2:self.n_z])/2
            T_z[1:self.n_x-1,1:self.n_y-1,1:self.n_z-1] = ( self.T[1:self.n_x-1,1:self.n_y-1,2:self.n_z] - self.T[1:self.n_x-1,1:self.n_y-1,0:self.n_z-2])/(2 * self.delta_z)                        

            laplacian_T[1:-1,1:-1,1:-1] = (self.T[2:,1:-1,1:-1] - 2.0* self.T[1:-1,1:-1,1:-1] + self.T[0:-2,1:-1,1:-1]) / (self.delta_x * self.delta_x) \
                        + (self.T[1:-1,2:,1:-1] - 2.0* self.T[1:-1,1:-1,1:-1] + self.T[1:-1,0:-2,1:-1]) / (self.delta_y * self.delta_y) \
                        + (self.T[1:-1,1:-1,2:] - 2.0* self.T[1:-1,1:-1,1:-1] + self.T[1:-1,1:-1,0:-2]) / (self.delta_z * self.delta_z)

            self.T_new[1:-1,1:-1,1:-1] = self.T[1:-1,1:-1,1:-1] + self.dt * (-1.0 * (U_at_Tijk[1:-1,1:-1,1:-1]  * T_x[1:-1,1:-1,1:-1] + V_at_Tijk[1:-1,1:-1,1:-1]  * T_y[1:-1,1:-1,1:-1]  + W_at_Tijk[1:-1,1:-1,1:-1]  * T_z[1:-1,1:-1,1:-1] ) + laplacian_T[1:-1,1:-1,1:-1] /(self.Re * self.Pr))
            


            self.u = self.u_new
            self.v = self.v_new
            self.w = self.w_new
            self.T = self.T_new

        return



    def set_airflow_control(self,angle,angle_width,velocity_mag,action):

        delta_angle = self.delta_angle
        delta_angle_width = self.delta_angle_width
        
        #keep STAY
        #action == 2 

        #UP        
        if action == Actions.UP and (angle>(0 + delta_angle)): 
            angle = angle - delta_angle * (1.0)
            print("Action is UP")                  
        #DOWN
        elif action == Actions.DOWN and (angle<(90 - delta_angle)): 
            angle = angle + delta_angle * (1.0)
            print("Action is DOWNS")                
        #LEFT
        elif action == Actions.LEFT and (angle_width < 90): 
            angle_width = angle_width + delta_angle_width * (1.0)
            print("Action is LEFT")  
        #RIGHT
        elif action == Actions.RIGHT and (angle_width > -90 ): 
            angle_width = angle_width - delta_angle_width * (1.0)
            print("Action is RIGHT") 
        #STAY
        else:
            angle = angle + delta_angle * (0.0)
            print("Action is STAY")
        
        
        
        
        
        u_inlet = velocity_mag * np.cos(angle / 180 * np.pi) * np.cos(angle_width/180 * np.pi)

        v_inlet = -1.0 * velocity_mag * np.sin(angle / 180 * np.pi)
        w_inlet = velocity_mag *  np.sin(angle_width/180 * np.pi)

        print('angle=',angle,'angle_width=',angle_width)
                         
        return u_inlet, v_inlet ,w_inlet, velocity_mag, angle, angle_width





    def step_temperature(self,action,time_start,time_end):

        self.update_SMAC_temperature(action,time_start,time_end)

        return




    def compute_reward(self):
        
        
        # 20220111
        # 部屋の空気温度の標準偏差の逆数を報酬にする
        reward = 1 / np.std(self.T)

        if reward > 100:
            reward = 100
        
        
        #20211207
        # sigma = 0.01
        
        # position_middle = int((self.position_inlet_left + self.position_inlet_right)/2)
        # position_length = 19

        # Temperature_objective_high = self.T[position_length][self.O_high][position_middle]
        # Temperature_objective_low = self.T[position_length][self.O_low][position_middle]
        
        # ratio_wall = Temperature_objective_high / Temperature_objective_low
        
        # reward_wall = np.exp(-((ratio_wall-1.0)*(ratio_wall-1.0))/(2 * 1 * 1)) / (np.sqrt(2 * np.pi * sigma))
        

        # Temperature_floor_1 =  self.T[3][1][position_middle]
        # Temperature_floor_2 =  self.T[16][1][position_middle]
        
        # ratio_floor = Temperature_floor_1 / Temperature_floor_2
        
        # reward_floor = np.exp(-((ratio_floor-1.0)*(ratio_floor-1.0))/(2 * 1 * 1)) / (np.sqrt(2 * np.pi * sigma))
        
        
        # reward = reward_floor + reward_wall
        
        
        #20210123
        #reward = np.abs((Temperature_objective_high - Temperature_objective_low)/Temperature_objective_high)
        
        #20210124
        #reward = np.abs(Temperature_objective_high/(Temperature_objective_high - Temperature_objective_low))
        
        
        #print("Reward =", reward, "ratio_wall= ",ratio_wall,"ratio_floor= ",ratio_floor)

        return reward



    def PID_control(self):
        """
        PID control 

        """

        
        position_middle = int((self.position_inlet_left + self.position_inlet_right)/2)
        position_length = 4

        Temperature_objective_high = self.T[position_length][self.O_high][position_middle]
        Temperature_objective_low = self.T[position_length][self.O_low][position_middle]

        self.pid_observe = Temperature_objective_high / Temperature_objective_low
        self.pid_ref = 1.0

        self.pid_error_p = self.pid_observe - self.pid_ref
        self.pid_error_i += (self.pid_observe - self.pid_ref) 

        u_angle_pid = self.pid_kp * self.pid_error_p + self.pid_ki * self.pid_error_i

        print('u_angle_pid = ', u_angle_pid)

        if u_angle_pid >= 0:
            action_pid = Actions.DOWN
        else:
            action_pid = Actions.UP 

        return action_pid 
  
      
    def judge_state(self,var_angle_for_state):
        """
        吹出風の風向とセンサー3点の温度の順列で状態を定義している。

        The state is defined by the airflow direction and the permutation of the temperature of the three sensor points.
        """
        
        position_middle = int((self.position_inlet_left + self.position_inlet_right)/2)
        
        T1 = self.T[self.T1_i][self.sensor_j][position_middle]
        T2 = self.T[self.T2_i][self.sensor_j][position_middle]
        T3 = self.T[self.T3_i][self.sensor_j][position_middle]

        n_angle = self.angle / var_angle_for_state
        
        if (T1>=T2) and (T2>=T3):
            s_mod = 0
        elif  (T1>=T2) and (T2>=T3):
            s_mod = 1
        elif  (T1>=T3) and (T3>=T2):
            s_mod = 2
        elif  (T1>=T3) and (T3>=T2):
            s_mod = 3
        elif  (T2>=T1) and (T1>=T3):
            s_mod = 4
        else:
            s_mod = 5
 
        s_q = int(n_angle + s_mod)

        return s_q 

    def observe_temperature(self):
        """
        部屋の温度点数のすべてを状態として定義する。
        Define all of the temperature points in the room as a state.
        
        床面の温度分布を状態として定義する（2021.12/2）。
        """
        
        #  部屋の温度点数のすべてを状態として定義する。
        # return self.T.ravel()
        
        T_floor = self.T[:,1,:]
        T_floor = T_floor.T
        
        return T_floor.ravel()


    def _render(self, mode='human', close=False):
        
        #if close:
        #  if self.viewer is not None:
        #    self.viewer.close()
        #    self.viewer = None
        #  return
      
        return 



    def save_fig(self,calculating_episode, current_result_folder, time, time_limit_cycle, reward_log):
        
        time_filename = time
        self.current_result_folder = current_result_folder
        
        position_middle = int((self.position_inlet_left + self.position_inlet_right)/2)
        
        #calc u,v,x,y at pressure grid
        for i in range(0,self.n_x):
             for j in range(0,self.n_y):
                 self.u_at_pressure[i][j] = (self.u[i][j][position_middle] + self.u[i-1][j][position_middle]) /2
                 self.v_at_pressure[i][j] = (self.v[i][j][position_middle] + self.v[i][j-1][position_middle]) /2
                 self.w_at_pressure[i][j] = (self.w[i][j][position_middle] + self.w[i][j][position_middle+1]) /2

        for i in range(0,self.n_x):
            self.x_at_pressure[i] = self.delta_x * 0.5 + self.delta_x * i

        for j in range(0,self.n_y):
            self.y_at_pressure[j] = self.delta_y * 0.5 + self.delta_y * j
            
        for k in range(0,self.n_z):
            self.z_at_pressure[k] = self.delta_z * 0.5 + self.delta_z * k

        xx, yy = np.meshgrid(self.x_at_pressure, self.y_at_pressure)

        
        T_2D = self.T[:,:,position_middle]
        T_2D = T_2D.T
        
        T_floor = self.T[:,1,:]
        T_floor = T_floor.T
        
        norm_velocity = np.sqrt(np.power(self.u_at_pressure, 2) + np.power(self.v_at_pressure, 2))
        norm_velocity = norm_velocity.T

        fig,ax =  plt.subplots(1,4,figsize=(24, 2))

        #plt.axes().set_aspect('equal')
        levels_velocity = [0,0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        contour1 = ax[0].contourf(xx, yy, norm_velocity,levels_velocity,cmap='Blues')
        contour1.clabel(fmt='%1.1f', fontsize=14)
        fig.colorbar(contour1,ax=ax[0])

        u_at_pressure_T = self.u_at_pressure.T
        v_at_pressure_T = self.v_at_pressure.T
        
        #u and v is opposite.
        ax[0].quiver(xx,yy,u_at_pressure_T ,v_at_pressure_T,scale=5)
        ax[0].set_title('Episode=' + str(calculating_episode) + ',Time=' + str(time_filename))

        #plot temperature field
        levels_temperature = [0,0.01, 0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        
        contour2 = ax[1].contourf(xx, yy, T_2D, levels_temperature,cmap='jet',alpha=0.8)
        contour2.clabel(fmt='%1.1f', fontsize=14)
        fig.colorbar(contour2,ax=ax[1])

        # ax[1].quiver(xx,yy,u_at_pressure_T,v_at_pressure_T ,angles='xy',scale_units='xy',scale=5)
        ax[1].quiver(xx,yy,u_at_pressure_T,v_at_pressure_T ,scale=5)


        xx, zz = np.meshgrid(self.x_at_pressure, self.z_at_pressure)        

        levels_temperature_floor = [0,0.01, 0.05, 0.1,0.2,0.3,0.4,0.5]
        contour3 = ax[2].contourf(xx, zz, T_floor , levels_temperature_floor,cmap='jet',alpha=0.8)
        contour3.clabel(fmt='%1.1f', fontsize=14)
        fig.colorbar(contour3,ax=ax[2])

        
        ##plot reward history
        time_data = list(range(len(reward_log[(calculating_episode -1) * time_limit_cycle:])))
        # #del time_data[0:(time -1) * time_limit_cycle]
        
        time_reward =  list(map(lambda x: x * 20, time_data))
        
        ax[3].set_ylim(0,0.5)
        ax[3].set_xlim(0,time_limit_cycle)
        ax[3].grid()
        ax[3].plot(time_reward, reward_log[(calculating_episode -1) * time_limit_cycle:])


        time_filename_zeropadding = str(time_filename).zfill(4)
        filename = 'airflow' + time_filename_zeropadding + '.png' 
        
        # self.current_result_folder = 'result' + str(calculating_episode).zfill(4)
        
        plt.savefig('./' + self.current_result_folder + '/' + filename)


        # release the memory
        plt.clf()
        plt.close()

        
        
        #Save Temperature
        filename_Temperature = 'Temperature' + time_filename_zeropadding
        filename_Pressure = 'Pressure' +  time_filename_zeropadding
        filename_Velocity_U = 'Velocity_U' +  time_filename_zeropadding
        filename_Velocity_V = 'Velocity_V' +  time_filename_zeropadding
        filename_Velocity_W = 'Velocity_W' +  time_filename_zeropadding




        
        print('saving numpy file')
        np.save('./'+self.current_result_folder + '/' + filename_Temperature,self.T)
        np.save('./'+self.current_result_folder + '/' + filename_Pressure,self.p)
        np.save('./'+self.current_result_folder + '/' + filename_Velocity_U,self.u)
        np.save('./'+self.current_result_folder + '/' + filename_Velocity_V,self.v)
        np.save('./'+self.current_result_folder + '/' + filename_Velocity_W,self.w)
        

        return xx,yy,norm_velocity
        

