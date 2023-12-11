import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
import matplotlib.pyplot as plt
import sumolib
import sys
import traci
import scipy.io as scio
from structure_car_follow import Actor
from structure_car_follow import Critic
from structure_car_follow import DDPG_Memory
import pandas as pd
np.random.seed(1)
tf.set_random_seed(1)
sess = tf.Session()
data_path1 = 'D:\\Wechat\\file\\Data_Standard Driving Cycles\\China_city+HWFET.mat'
data1 = scio.loadmat(data_path1)
y1 = data1['speed_vector'][0]
data_path2 = 'D:\\Wechat\\file\\Data_Standard Driving Cycles\\UDDS+US06_2.mat'
data2 = scio.loadmat(data_path2)
y2 = data2['speed_vector'][0]
y=[]
y = list(y1)+list(y2)
base_mean_speed = sum(y)/len(y)



# MAX_EPISODES = 99
# MAX_EPISODES = 150
MAX_EPISODES = 20
MAX_EP_STEPS = len(y)-16
# print(MAX_EP_STEPS)
LR_A = 1e-3  # learning rate for actor
last_LR_A = 8e-4+5e-5
# last_LR_A = 1e-3
LR_C = 1e-4  # learning rate for critic
last_LR_C = 8e-5+5e-6
# last_LR_C = 1e-4
GAMMA = 0.9  # reward discount
# REPLACE_ITER_A = 800
# REPLACE_ITER_C = 700
REPLACE_ITER_A = 3865
REPLACE_ITER_C = 3000
MEMORY_CAPACITY = 3865
BATCH_SIZE = 128
VAR_MIN = 0.1
RENDER = False
LOAD = True
# LOAD = False
DISCRETE_ACTION = False
STATE_DIM = 4
ACTION_DIM = 1
ACTION_BOUND = [-1,1]

# print(base_mean_speed)





# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A,last_LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C,last_LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = DDPG_Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
# path ='./9.24_DDPG'
# path = './10.13-2-1 NEW_DDPG'
# path = './10.7-9 NEW_DDPG'
# path = './1118 NEW_DDPG'
path = './1126 NEW_DDPG'
if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

if_sumo_gui = True
if not if_sumo_gui:
    sumoBinary = sumolib.checkBinary('sumo')
else:
    sumoBinary = sumolib.checkBinary('sumo-gui')
sumocfgfile = "D:\\Project_codes of pycharm\\2021.6.29\\car-follow-5.24\\car-lc.sumocfg"
traci.start([sumoBinary, "-c", sumocfgfile])

# var = 2.  # control exploration
var = 0.1
ep_collision_list = []
ep_list = []
ep_reward = []
ep_mean_speed = []
ep_base_mean_speed = []
kkk=0
last_speed = 11.6
last_jerk = 1.2
first_time = 1
for i in range(MAX_EPISODES):
    traci.load(["-c", "D:\\Project_codes of pycharm\\2021.6.29\\car-follow-5.24\\car-lc.sumocfg"])  # 调用sumo文件
    traci.simulationStep(5)
    traci.vehicle.setSpeed("002", 1)
    traci.vehicle.setSpeed("001", 2)
    traci.vehicle.setSpeedMode("001", 12)
    traci.vehicle.setSpeedMode("002", 12)
    traci.simulationStep(14)  # 仿真12步，确保所有车辆已经进入道路
    all_r = []  # 奖励列表
    all_jerk_value = [] #jerk列表
    rear_v_list = []  # 速度列表
    l_v_list = []  # 前车速度列表
    ep_step_list = []
    distance_headway_list = []
    danger_time = []
    total_r = 0  # 单回合的总奖励
    r0 = 0  # 换道惩罚初始化
    t = 0
    DDPG_s = np.zeros(4)  # 生成跟驰状态空间
    DDPG_s_ = np.zeros(4)  # 生成跟驰的下一状态空间
    s = np.zeros(4)
    s_ = np.zeros(4)
    ep_step = 0
    l_a = 0
    collision_list = []
    dis_safe_list = []
    for j in range(MAX_EP_STEPS):
        # while True:
        Distance_list = []
        r_step = 1
        m = j
        r_dis = 0
        danger = 0
        collision = 0
        # traci.vehicle.setSpeedMode("002", 14)

        r_v = traci.vehicle.getSpeed("002")  # 获取主车车速，状态0
        l_v = traci.vehicle.getSpeed("001")
        r_distance_value = traci.vehicle.getDistance("002")  # 获得主车行驶里程
        position = traci.vehicle.getPosition("002")  # 获得主车坐标
        r_danger = 0

        l_a = y[m + 15] - l_v

        DDPG_s[0] = r_v
        DDPG_s[1] = l_v
        DDPG_s[2] = traci.vehicle.getPosition("001")[0] - traci.vehicle.getPosition("002")[0] - 5
        # DDPG_s[2] = traci.vehicle.getPosition("001")[0] - traci.vehicle.getPosition("002")[0] - 2.5
        DDPG_s[3] = l_a
        if DDPG_s[2]>150:
            DDPG_s[1]=33
            DDPG_s[2]=150
            DDPG_s[3]=0.01
        s =  DDPG_s
        # s[0] = s[0]/33
        # s[1] = s[1] / 33
        # s[2] = s[2] / 150
        # s[3] = s[3] / 3
        a = actor.choose_action(s)
        a = np.clip(np.random.normal(a, var), *ACTION_BOUND)  # add randomness to action selection for exploration
        DDPG_action = a
        update_speed = min(max(DDPG_s[0] + 3 * DDPG_action, 0.2),33)  # 速度更新
        traci.vehicle.setSpeed("002", update_speed)  # 主车车速更新
        traci.vehicle.setSpeed("001", y[m + 15])  # 虚拟前车车速设定

        traci.simulationStep(m + 15)
        DDPG_s_[0] = traci.vehicle.getSpeed("002")
        DDPG_s_[1] = traci.vehicle.getSpeed("001")
        DDPG_s_[2] = (traci.vehicle.getPosition("001")[0] - traci.vehicle.getPosition("002")[0] - 5)
        # DDPG_s_[2] = (traci.vehicle.getPosition("001")[0] - traci.vehicle.getPosition("002")[0] - 2.5)
        DDPG_s_[3] = y[m + 16] - traci.vehicle.getSpeed("001")
        if DDPG_s_[2]>150:
            DDPG_s_[1]=33
            DDPG_s_[2]=150
            DDPG_s_[3]=0.01
        distance_headway = DDPG_s_[2]
        # dis_safe = (0.36 * DDPG_s_[0] + (DDPG_s_[0] * DDPG_s_[0]) / 6 + 0.0001)
        dis_safe = (0.36 * DDPG_s_[0] + (DDPG_s_[0] * DDPG_s_[0]) * 0.1)
        if distance_headway > 0:
            # r_dis = 1 - min(abs(np.log(distance_headway) / np.log(2 * dis_safe)), 5)
            # r_dis = 1
            r_dis = 1.12*(150-distance_headway)/75   ####时距

            # r_dis = 3*(distance_headway/150-1)*(distance_headway/150-1)
            # if distance_headway < min(dis_safe,30):
            #     # r_dis = -10
            #     r_dis = 1.5
            if distance_headway >= 150:
                # r_dis = min(-30,150-distance_headway)
                r_dis = -30
        if distance_headway <= 0:
            # r_dis = min(10 * distance_headway,-20)
            r_dis = min(10 * distance_headway, -500)
            collision = 1
            # if i >=150:
            #     break
        # if distance_headway == 0:
        #     r_dis = -10
        #     collision = 1
        r_efficient = DDPG_s_[0] / 33

        # r = 3*r_efficient + r_dis
        # r = 10 * r_efficient + r_dis - 6*abs(DDPG_s_[0]-DDPG_s[0])
        # r = 17.5 * r_efficient + r_dis - 2*abs(DDPG_s_[0]-DDPG_s[0])
        # r = 16.5 * r_efficient + r_dis - 2 * abs(DDPG_s_[0] - DDPG_s[0])

        # r = 33 * r_efficient * 0.6 + r_dis - 2.25 * (DDPG_s_[0] - DDPG_s[0])*(DDPG_s_[0] - DDPG_s[0])
        r = 33 * r_efficient * 0.6  - 2.25 * (DDPG_s_[0] - DDPG_s[0]) * (DDPG_s_[0] - DDPG_s[0])
        # r =  3*r_dis - 2.25 * (DDPG_s_[0] - DDPG_s[0]) * (DDPG_s_[0] - DDPG_s[0])
        # r = 33 * r_efficient * 0.6 + r_dis - 4*abs(DDPG_s_[0] - DDPG_s[0])
        jerk_value=abs(DDPG_s_[0]-DDPG_s[0])

        # total_step += 1

        ep_step += 1
        # print(DDPG_s_)
        s = DDPG_s
        a = DDPG_action
        s_ = DDPG_s_
        # s_[0] = s_[0] / 33
        # s_[1] = s_[1] / 33
        # s_[2] = s_[2] / 150
        # s_[3] = s_[3] / 3

        # M.store_transition(s, a, r, s_)
        # if M.pointer > MEMORY_CAPACITY:
        #     var = max([var * 0.999995, VAR_MIN])  # decay the action randomness
        #     # var = var * 0.99995
        #     b_M = M.sample(BATCH_SIZE)
        #     b_s = b_M[:, :STATE_DIM]
        #     b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
        #     b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
        #     b_s_ = b_M[:, -STATE_DIM:]
        #
        #     critic.learn(b_s, b_a, b_r, b_s_)
        #     actor.learn(b_s)

        all_r.append(r)
        collision_list.append(collision)
        if collision==1:
            # print('episode=%s' % i,'step=%s' % j,'dangerious!')0
            traci.vehicle.moveToXY('002',"gneE0", traci.vehicle.getLaneIndex('002'), traci.vehicle.getPosition('001')[0]-22.5, traci.vehicle.getPosition('001')[1],keepRoute=1)
            # break
        # total_step += 1
        ep_step_list.append(ep_step)
        rear_v_list.append(DDPG_s_[0])
        l_v_list.append(DDPG_s_[1])
        distance_headway_list.append(DDPG_s_[2])
        dis_safe_list.append(dis_safe)
        danger_time.append(danger)
        # collision_list.append(collision)
        all_jerk_value.append(jerk_value)
    ep_list.append(i)
    ep_collision_list.append(sum(collision_list))
    ep_reward.append(sum(all_r))
    mean_speed = sum(rear_v_list) / MAX_EP_STEPS
    ep_mean_speed.append(mean_speed)
    ep_base_mean_speed.append(base_mean_speed)
    mean_r = sum(all_r)/MAX_EP_STEPS
    mean_jerk=sum(all_jerk_value)/MAX_EP_STEPS
    mean_headway = sum(distance_headway_list)/len(distance_headway_list)
    print('episode=%s' % i, 'mean_reward=%s' % mean_r,'total_collision=%s' % sum(collision_list), 'mean_speed=%s' % mean_speed,'mean_jerk_value=%s' % mean_jerk, 'mean_headway=%s' % mean_headway,'A learn=%s' % actor.lr,'C learn=%s' % critic.lr)

    # if mean_speed>=11.62 and mean_jerk<=0.4 and sum(collision_list)==0 and (33 not in rear_v_list) and (33 not in l_v_list):
    if sum(collision_list) == 0 and mean_speed >= 11.6:
        kkk+=1
    if sum(collision_list) != 0:
        kkk=0
    if kkk>=3 and mean_speed >= 11.6  and max(l_v_list) <33:

        plt.plot(ep_step_list, rear_v_list, color='r', label='Ego_car')
        plt.plot(ep_step_list, l_v_list, color='b', label='Leader')
        plt.xlabel('Running time (s)', fontsize=20, color='black')
        plt.ylabel('Velocity (m/s)', fontsize=20, color='black')
        plt.legend()
        plt.show()
        plt.plot(ep_step_list,distance_headway_list)
        plt.show()
        test1=pd.DataFrame({'ep_step_list':ep_step_list,'rear_v_list':rear_v_list,'l_v_list':l_v_list,'distance_headway_list':distance_headway_list})
        test1.to_csv('./figure106/2023-5-30-carfollow.csv')
# plt.plot([i+1 for i in range(len(critic.loss_his))],critic.loss_his)
# plt.show()
    # plt.plot(total_distance,RL.cost_his)
        # path = './1126 NEW_DDPG'
        # if os.path.isdir(path): shutil.rmtree(path)
        # os.mkdir(path)
        # ckpt_path = os.path.join(path, 'DDPG.ckpt')
        # save_path = saver.save(sess, ckpt_path, write_meta_graph=True)
        # print("\nSave Model %s\n" % save_path)
        ###record:100,
        # path = './10.7-10 NEW_DDPG'
        # if os.path.isdir(path): shutil.rmtree(path)
        # os.mkdir(path)
        # ckpt_path = os.path.join(path, 'DDPG.ckpt')
        # save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
        # print("\nSave Model %s\n" % save_path)

        # break
    # if mean_speed >= 11.6 and mean_jerk <= 1 and sum(collision_list) == 0 and i>=80 and first_time ==1:
    #     plt.plot(ep_step_list, rear_v_list, color='r', label='Ego_car')
    #     plt.plot(ep_step_list, l_v_list, color='b', label='Leader')
    #     plt.xlabel('Running time (s)', fontsize=20, color='black')
    #     plt.ylabel('Velocity (m/s)', fontsize=20, color='black')
    #     plt.legend()
    #     plt.show()
    #     plt.plot(ep_step_list,distance_headway_list)
    #     plt.show()
    #     print(i)
    #     last_speed = mean_speed
    #     last_jerk = mean_jerk
    #     first_time = 0
    # if mean_speed>=last_speed and mean_jerk<=last_jerk and first_time == 0:
    #     last_speed = mean_speed
    #     last_jerk = mean_jerk
    #     path = './10.7-2-1 NEW_DDPG'
    #     if os.path.isdir(path): shutil.rmtree(path)
    #     os.mkdir(path)
    #     ckpt_path = os.path.join(path, 'DDPG.ckpt')
    #     save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    #     print("\nSave Model %s\n" % save_path)
        # break
    # if mean_speed>=11.6 and mean_jerk<=0.5 and sum(collision_list)==0 and i>=40:
    #     break
    # if  mean_speed >= 11.62 and mean_jerk <= 0.378 and sum(collision_list) == 0:
    #     plt.plot(ep_step_list, rear_v_list, color='r', label='Ego_car')
    #     plt.plot(ep_step_list, l_v_list, color='b', label='Leader')
    #     plt.xlabel('Running time (s)', fontsize=20, color='black')
    #     plt.ylabel('Velocity (m/s)', fontsize=20, color='black')
    #     plt.legend()
    #     plt.show()
    #     path = './9.24_DDPG'
    #     if os.path.isdir(path): shutil.rmtree(path)
    #     os.mkdir(path)
    #     ckpt_path = os.path.join(path, 'DDPG.ckpt')
    #     save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    #     print("\nSave Model %s\n" % save_path)

traci.close()
# if mean_speed>=11.6 and mean_jerk<=0.5 and sum(collision_list)==0 and i>=40:
# plt.plot(ep_step_list, rear_v_list, color='r', label='Ego_car')
# plt.plot(ep_step_list, l_v_list, color='b', label='Leader')
# plt.xlabel('Running time (s)', fontsize=20, color='black')
# plt.ylabel('Velocity (m/s)', fontsize=20, color='black')
# plt.legend()
# plt.show()
# path = './1118 NEW_DDPG'
# if os.path.isdir(path): shutil.rmtree(path)
# os.mkdir(path)
# ckpt_path = os.path.join(path, 'DDPG.ckpt')
# save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
# print("\nSave Model %s\n" % save_path)

# test1=pd.DataFrame(columns=column1,data=list1)
# test1=pd.DataFrame({'ep_list':ep_list,'ep_collision_list':ep_collision_list,'mean_speed':ep_mean_speed,'ep_reward':ep_reward,'ep_base_mean_speed':ep_base_mean_speed,'ep_base_mean_speed':ep_base_mean_speed})
# test1.to_csv('./figure1.csv')
# test1=pd.DataFrame({'ep_step_list':ep_step_list,'rear_v_list':rear_v_list,'l_v_list':l_v_list,'distance_headway_list':distance_headway_list})
# test1.to_csv('./figure106/1-DDPG-carfollow.csv')