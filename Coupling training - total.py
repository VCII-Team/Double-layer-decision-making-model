import tensorflow as tf
import numpy as np
import shutil
import scipy.io as scio
from structure_car_follow import Actor
from structure_car_follow import Critic
from structure_car_follow import DDPG_Memory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "/device:GPU:0"
# CUDA_VISIBLE_DEVICES=2
import time
import traci
import sumolib
import sys
import math
import numpy as np
import shutil
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from structure_lane_change import DuelingDQNPrioritizedReplay



###读取工况数据作为训练车流环境
data_path1 = 'D:\\Wechat\\file\\Data_Standard Driving Cycles\\China_city+HWFET.mat'
data1 = scio.loadmat(data_path1)
y1 = data1['speed_vector'][0]
data_path2 = 'D:\\Wechat\\file\\Data_Standard Driving Cycles\\UDDS+US06_2.mat'
data2 = scio.loadmat(data_path2)
y2 = data2['speed_vector'][0]
y=[]
y = list(y1)+list(y2)
base_mean_speed = sum(y)/len(y)
np.random.seed(1)
tf.set_random_seed(1)

###此处对下层跟驰模块的DDPG算法进行定义
MAX_EPISODES = 0
LR_A = 1e-3  # learning rate for actor
last_LR_A = 8e-4
LR_C = 1e-4  # learning rate for critic
last_LR_C = 8e-5
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 3865
REPLACE_ITER_C = 3000
MEMORY_CAPACITY = 38650
BATCH_SIZE = 256
VAR_MIN = 0.1
RENDER = False
LOAD1 = True
DISCRETE_ACTION = False
STATE_DIM = 4
ACTION_DIM = 1
ACTION_BOUND = [-1,1]
sess1 = tf.Session()
# Create actor and critic.
actor = Actor(sess1, ACTION_DIM, ACTION_BOUND[1], LR_A,last_LR_A, REPLACE_ITER_A)
critic = Critic(sess1, STATE_DIM, ACTION_DIM, LR_C,last_LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)
M = DDPG_Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)
saver1 = tf.train.Saver()
# path1 = './10.7-9 NEW_DDPG'
# path1 = './1118 NEW_DDPG'
path1 ='./1126 NEW_DDPG'
if LOAD1:
    saver1.restore(sess1, tf.train.latest_checkpoint(path1))
else:
    sess1.run(tf.global_variables_initializer())



###对换道决策模块的DQN算法进行定义
s_dim = 16
DQN_s_dim = 15

a_dim = 3
DDPG_MEMORY_CAPACITY = 38650
TARGET_REP_ITER = 300
sess2 = tf.Session()
E_GREEDY = 1
E_INCREMENT = 0.00001
GAMMA = 0.9
LR = 0.0001
BATCH_SIZE = 128
HIDDEN = [600, 600, 600, 600]
LOAD2 = False
RENDER = True
RL = DuelingDQNPrioritizedReplay(
    n_actions=a_dim, n_features=DQN_s_dim, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
    hidden=HIDDEN, batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,
    memory_size=MEMORY_CAPACITY, e_greedy_increment=E_INCREMENT,)
saver = tf.train.Saver()
saver2 = tf.train.Saver()
path2 = './coupling train 917 2'
if LOAD2:
    saver2.restore(sess2, tf.train.latest_checkpoint(path2))
else:
    sess2.run(tf.global_variables_initializer())


###打开sumo的接口
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
sumocfgfile = "D:\\Project_codes of pycharm\\2021.6.29\\sumo-923\\car-lc.sumocfg"
traci.start([sumoBinary, "-c", sumocfgfile])

# var = 2.  # control exploration
var = 0.1   #此时的跟驰模型探索率
total_step = 0    #总的训练步数，每一步记一次，用在跟驰部分记忆池
total_ep_list = []     #总训练回合数生成的列表，可以用作回合数画图的横坐标
total_lc_r_list = []    #每训练回合的奖励
mean_speed_list = []    #每训练回合的平均速度
total_fleet_mean_speed = []    #每训练回合的车流中位车速
lanechange_learn_start = 0     #开始换道学习
carfollowing_learn_start = 0    #开始跟驰学习
for i in range(MAX_EPISODES):

    fleet_mean_speed = []
    ep_reward = 0
    ep_reward_all = 0
    ep_step = 0

    v = []
    all_r = []
    rear_v_list = []
    l_v_list = []
    all_changelanetimes = []
    # all_changelanetimes.append(0)
    lane_index0_position_x = []
    lane_index0_id_list = []
    lane_index1_position_x = []
    lane_index1_id_list = []
    lane_index2_position_x = []
    lane_index2_id_list = []
    lane_index0_front = []
    lane_index1_front = []
    lane_index2_front = []
    danger_lc = []
    all_cf_r = []
    cf_collision_list = []
    lane_index0_acc_list=[]
    lane_index1_acc_list=[]
    lane_index2_acc_list=[]
    ep_step_list=[]
    distance_headway_list = []
    dis_safe_list = []
    total_r = 0
    r0 = 0  # 换道惩罚初始化
    t = 0
    k = 1


    s = np.zeros(s_dim)
    s_ = np.zeros(s_dim)
    DQN_s = np.zeros(DQN_s_dim)
    DQN_s_ = np.zeros(DQN_s_dim)

    DDPG_s = np.zeros(4)
    DDPG_s_ = np.zeros(4)
    cf_s = np.zeros(4)
    ego_id = 0  # 初始时主车的ID是0#######################################
    horizontal_position_list=[]
    lengthwise_position_list=[]
    order_ego_id_list = []
    r_danger_lc = 0
    r_high_frequency_lc = 1


    MAX_EP_STEPS = len(y)-60
    traci.load(["-c", "D:\\Project_codes of pycharm\\2021.6.29\\sumo-923\\car-lc.sumocfg"])
    traci.simulationStep(50)
    for step in range(10):
        ID_list_all = traci.edge.getLastStepVehicleIDs("gneE0")
        for vehicle in ID_list_all:
            if int(vehicle)!=ego_id:
                traci.vehicle.setSpeed(vehicle, 5)  # 初始速度设定
                traci.vehicle.setSpeedMode(vehicle, 12)
                traci.vehicle.setLaneChangeMode(vehicle, 512)  # 换道机制设定
        traci.simulationStep(50 + step)

    origin_speed_ego_car_list = []
    origin_speed_ego_car=traci.vehicle.getSpeed('%d' % ego_id)
    if i >=0:
        lanechange_learn_start = 1
    if i>=100:
        carfollowing_learn_start = 1
    state_space = []
    lc_action_list = []
    cf_action_list = []
    start_time = time.time()
    already_t = 60
    for j in range(MAX_EP_STEPS):
        all_car_position_x = []
        all_car_position_y = []
        lane_index0_position_x = []
        lane_index0_id_list = []
        lane_index1_position_x = []
        lane_index1_id_list = []
        lane_index2_position_x = []
        lane_index2_id_list = []
        lane_index0_front = []
        lane_index1_front = []
        lane_index2_front = []
        Distance_list = []
        all_speed_list = []
        changelane = 0
        m = j
        r0 = 0
        cf_danger = 0
        left_lanechange_dangerious = 0
        right_lanechange_dangerious = 0
        traci.simulationStep(j + already_t)
        ####这部分的代码作用是获取路网所有车辆，并按照行驶的位置进行前后排序
        ID_list_all = traci.edge.getLastStepVehicleIDs("gneE0")  # 获取主路所有车辆的ID编号
        for x in range(len(ID_list_all)):  # 长度应该为16
            Distance_list.append(traci.vehicle.getDistance(ID_list_all[x]))  # 16个元素跑的里程
        Index = sorted(range(len(Distance_list)), key=lambda k: Distance_list[k], reverse=True)  # 距离表降序排列的索引
        Index = np.array(Index)
        ID_list_all = list(map(int, ID_list_all))  # 字符串转化为数值型
        ID_list_order = np.array(ID_list_all)[Index]  # 按照前后顺序进行车序排列
        ID_list_order_list = ID_list_order.tolist()   #转换为list
        order_ego_id = ID_list_order_list.index(ego_id)   #返回主车在车流的的序号
        ###排序完成之后对主车进行判断（是否主车已经超越所有的车辆成为第一辆车）和操作
        lane_index = traci.vehicle.getLaneIndex('%d' % ego_id)  # 获得主车所在车道index
        position = traci.vehicle.getPosition('%d' % ego_id)  # 获得主车坐标
        position = [max(position[0], 0), max(position[1], -8)]
        if ego_id == ID_list_order[0] or ego_id == ID_list_order[1] or ego_id == ID_list_order[2]:  # 如果主车成为了头车
            ego_id = ID_list_order[-1]  # 主车的控制对象变成了最后一辆车

        # traci.vehicle.setLaneChangeMode('%d' % ego_id, 1621)  # 让主车可以自由进行换道
        # traci.vehicle.setLaneChangeMode('%d' % ego_id, 256)  # 避免碰撞
        # traci.vehicle.setLaneChangeMode('%d' % ego_id, 512)  # 避免碰撞和安全间隙
        # traci.vehicle.setSpeedMode('%d' % ego_id, 31)
        ###获取状态值
        r_v = traci.vehicle.getSpeed('%d' % ego_id)  # 获取主车车速，状态0
        s[0] = r_v
        s[1] = traci.vehicle.getLaneIndex('%d' % ego_id)  # 获取车道数
        s[14] = traci.vehicle.couldChangeLane('%d' % ego_id, 1)  # 左换道可行性
        s[15] = traci.vehicle.couldChangeLane('%d' % ego_id, -1)  # 右换道可行性
        s[5] = 33
        s[6] = 0.01
        s[7] = 150
        s[8] = 33
        s[9] = 0.01
        s[10] = 150
        s[11] = 33
        s[12] = 0.01
        s[13] = 150
        r_distance_value = max(traci.vehicle.getDistance('%d' % ego_id), 0)  # 获得主车行驶里程
        ######此部分代码用于每车道前后车排序
        for p in range(len(ID_list_order)):
            if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -8:
                lane_index0_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])  #
                lane_index0_id_list.append(ID_list_order[p])
            if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -4.8:
                lane_index1_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                lane_index1_id_list.append(ID_list_order[p])
            if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -1.6:
                lane_index2_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                lane_index2_id_list.append(ID_list_order[p])
        ######此部分代码用于找出每一车道上的前车ID
        for p in range(len(lane_index0_position_x)):
            if lane_index0_position_x[p] <= position[0]+5:
                break
            lane_index0_front.append(lane_index0_id_list[p])   #int
        for p in range(len(lane_index1_position_x)):

            if lane_index1_position_x[p] <= position[0]+5:
                break
            lane_index1_front.append(lane_index1_id_list[p])
        for p in range(len(lane_index2_position_x)):

            if lane_index2_position_x[p] <= position[0]+5:
                break
            lane_index2_front.append(lane_index2_id_list[p])

        ###对车辆进行速度设定
        for w in range(len(ID_list_order)):
            # if w==0:
            #     traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
            #     traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 512)  # 换道机制设定
            #     traci.vehicle.setSpeed('%d' % ID_list_order[w], y[ m + already_t - w])
            #     continue
            if ID_list_order[w]==ego_id:###对于主车
                traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
                if lanechange_learn_start==0:
                    traci.vehicle.setLaneChangeMode('%d' % ego_id, 1621)  # 让主车可以自由进行换道
                if lanechange_learn_start==1:
                    traci.vehicle.setLaneChangeMode('%d' % ego_id, 512)  #
                origin_speed_ego_car = y[ m + already_t - w]
                origin_speed_ego_car_list.append(origin_speed_ego_car)
                continue
            if ID_list_order[w]!=ego_id:###对于非主车
                # traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)   ###安全检查开启，但是仍要按照工况设定行驶
                # traci.vehicle.setSpeed('%d' % ID_list_order[w], y[m + 36 - w])
                # traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 512)  # 换道机制设定
                traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)
                traci.vehicle.setSpeed('%d' % ID_list_order[w], -1)
                if -2.5 <=(traci.vehicle.getPosition('%d' % ego_id)[0]-traci.vehicle.getPosition('%d' % ID_list_order[w])[0]-5)<=10:
                    if (traci.vehicle.getLaneIndex('%d' % ego_id) - s[1])==1:
                        left_lanechange_dangerious = 1
                        continue
                    if (traci.vehicle.getLaneIndex('%d' % ego_id) - s[1]) == -1:
                        right_lanechange_dangerious = 1
                        continue

                if lane_index0_front:
                    if ID_list_order[w] in lane_index0_front:
                        # traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
                        # traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 1621)  # 换道机制设定
                        # traci.vehicle.setSpeed('%d' % ID_list_order[w], y[ m + already_t - 2*w])
                        traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)
                        traci.vehicle.setSpeed('%d' % ID_list_order[w], -1)
                        if ID_list_order[w] == (lane_index0_front[0]):
                            traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
                            traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 512)  # 换道机制设定
                            traci.vehicle.setSpeed('%d' % ID_list_order[w], y[ m + already_t - w])
                            continue
                        if ID_list_order[w] == (lane_index0_front[-1]):
                            lane_index0_v = traci.vehicle.getSpeed('%d' % lane_index0_front[-1])
                            # lane_index0_acc = y[ m + already_t - 2*w] - lane_index0_v
                            lane_index0_acc = traci.vehicle.getAcceleration('%d' % lane_index0_front[-1])
                            lane_index0_s = traci.vehicle.getPosition('%d'% ID_list_order[w])[0]-traci.vehicle.getPosition('%d' % ego_id)[0]-5
                            if lane_index0_s <= 150:
                                s[5] = lane_index0_v
                                s[6] = lane_index0_acc
                                s[7] = lane_index0_s
                                if lane_index0_s <=10 and s[0]==1:
                                    right_lanechange_dangerious = 1
                        continue
                if lane_index1_front:
                    if ID_list_order[w] in lane_index1_front:
                        # traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
                        # traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 1621)  # 换道机制设定
                        # traci.vehicle.setSpeed('%d' % ID_list_order[w], y[ m + already_t - 2*w])
                        traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)
                        traci.vehicle.setSpeed('%d' % ID_list_order[w], -1)
                        if ID_list_order[w] == (lane_index1_front[0]):
                            traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
                            traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 512)  # 换道机制设定
                            traci.vehicle.setSpeed('%d' % ID_list_order[w], y[ m + already_t - w])
                            continue
                        if ID_list_order[w] == (lane_index1_front[-1]):
                            lane_index1_v = traci.vehicle.getSpeed('%d' % lane_index1_front[-1])
                            # lane_index1_acc = y[ m + already_t - w] - lane_index1_v
                            lane_index1_acc = traci.vehicle.getAcceleration('%d' % lane_index1_front[-1])
                            lane_index1_s = traci.vehicle.getPosition('%d'% ID_list_order[w])[0]-traci.vehicle.getPosition('%d' % ego_id)[0]-5
                            if lane_index1_s <= 150:
                                s[8] = lane_index1_v
                                s[9] = lane_index1_acc
                                s[10] = lane_index1_s
                                if lane_index1_s <=10:
                                    if s[0]==0:
                                        left_lanechange_dangerious = 1
                                    if s[0]==2:
                                        right_lanechange_dangerious = 1
                        continue
                if lane_index2_front:
                    if ID_list_order[w] in lane_index2_front:
                        # traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
                        # traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 1621)  # 换道机制设定
                        # traci.vehicle.setSpeed('%d' % ID_list_order[w], y[ m + already_t - 2*w])
                        traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)
                        traci.vehicle.setSpeed('%d' % ID_list_order[w], -1)
                        if ID_list_order[w] == (lane_index2_front[0]):
                            traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 12)
                            traci.vehicle.setLaneChangeMode('%d' % ID_list_order[w], 512)  # 换道机制设定
                            traci.vehicle.setSpeed('%d' % ID_list_order[w], y[ m + already_t - w])
                            continue

                        if ID_list_order[w] == (lane_index2_front[-1]):
                            lane_index2_v = traci.vehicle.getSpeed('%d' % lane_index2_front[-1])
                            # lane_index2_acc = y[ m + already_t - w] - lane_index2_v
                            lane_index2_acc = traci.vehicle.getAcceleration('%d' % lane_index2_front[-1])
                            lane_index2_s = traci.vehicle.getPosition('%d'% ID_list_order[w])[0]-traci.vehicle.getPosition('%d' % ego_id)[0]-5
                            if lane_index2_s <= 150:
                                s[11] = lane_index2_v
                                s[12] = lane_index2_acc
                                s[13] = lane_index2_s
                                if lane_index2_s <=10 and s[0]==1:
                                    left_lanechange_dangerious = 1
                        continue

        ###获得前车的ID和车距
        if s[1]==0:
            s[3] = s[6]
            s[4] = s[7]
            s[2] = s[5]
        if s[1] == 1:
            s[3] = s[9]
            s[4] = s[10]
            s[2] = s[8]
        if s[1]==2:
            s[3] = s[12]
            s[4] = s[13]
            s[2] = s[11]


        ### 此部分用于计算奖励时间
        for p in range(len(ID_list_order)):
            all_speed_list.append(traci.vehicle.getSpeed('%d' % ID_list_order[p]))
        sorted(all_speed_list, reverse=False)  ###速度列表升序
        alist = numpy.array(all_speed_list)  ###转换为数组
        q1 = numpy.percentile(alist, 25)
        q2 = numpy.percentile(alist, 50)
        q3 = numpy.percentile(alist, 75)
        iqr = q3 - q1
        q_low = q1 - (1.5 * iqr)
        q_85 = numpy.percentile(alist, 85)  # 85位车速
        v_max_limit = 33
        potential_reward_max = 33
        t0 = (q_85 - s[0]) / 3  # 计算加速到85位车速时间
        t1 = (33 - s[0]) / 3
        s1 = (s[0] + 33) * t1 * 0.5
        ##############车道0的优势函数计算
        ss1 = s[7] + t1 * s[5] - s1  # 加速阶段结束车间距
        if s[0] <= s[5]:  # 主车车速小于前车
            if ss1 >= 0:  # 加速阶段车距大于等于0
                t3 = (v_max_limit - s[5]) / 3  # 计算减速时间
                if t3 > 0:  # 减速时间大于0
                    s2 = (s[5] + v_max_limit) * t3 / 2  # 计算减速距离
                    ss2 = s[7] + (t1 + t3) * s[5] - s1 - s2  # 计算完整加减速过程后车距
                    if ss2 >= 0:  # 如果车距大于等于0
                        t2 = ss2 / (v_max_limit - s[5] + 0.01)  # 计算持续高速时间
                        reward_speed = ((t1 + t2 + t3) * s[5] + s[7]) / (t1 + t2 + t3)
                        potential_reward = min(reward_speed, potential_reward_max)
                    if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                        t2_1 = round((((s[7] / 3) + round(((round((s[5] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                        t1_1 = (s[5] - s[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s[5] + s[7]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                        # print(potential_reward,'line607')
                if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                    potential_reward = potential_reward_max
                    # print(potential_reward, 'line610')
            if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                t2_1 = round((((s[7] / 3) + round(((round((s[5] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                t1_1 = (s[5] - s[0]) / 3 + t2_1
                reward_speed = ((t1_1 + t2_1) * s[5] + s[7]) / (t1_1 + t2_1)
                potential_reward = min(reward_speed, potential_reward_max)
        if s[0] > s[5]:  # 主车车速大于前车
            t_brake = (s[0] - s[5]) / 3  # 计算紧急刹车时间
            s_brake = (s[0] + s[5]) / 2 * t_brake  # 紧急刹车距离
            ss_brake = s[5] * t_brake + s[7] - s_brake  # 刹车结束车距
            if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                if s[0] == v_max_limit:  # 此时只能匀速再刹车
                    t_keep = ss_brake / (s[0] - s[5])
                    reward_speed = ((t_brake + t_keep) * s[5] + s[7]) / (t_brake + t_keep)
                    potential_reward = min(reward_speed, potential_reward_max)
                if s[0] < v_max_limit:
                    ss_last = s[5] * (t_brake + t1) + s[7] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                    if ss_last > 0:  # 主车可以先加速再匀速再减速
                        t_keep = ss_last / (v_max_limit - s[5])  # 计算持续高速时间
                        reward_speed = ((t1 + t_keep + t_brake) * s[5] + s[7]) / (t1 + t_keep + t_brake)
                        potential_reward = min(reward_speed, potential_reward_max)
                    if ss_last <= 0:
                        t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                        reward_speed = ((t_brake + t_add) * s[5] + s[7]) / (t_brake + t_add)
                        potential_reward = min(reward_speed, potential_reward_max)
            if ss_brake < 0:
                potential_reward = 0
            if ss_brake == 0:
                reward_speed = s_brake / t_brake
                potential_reward = min(reward_speed, potential_reward_max)
        potential_reward_0 = potential_reward
        #########车道1的优势函数计算
        ss1 = s[10] + t1 * s[8] - s1  # 加速阶段结束车间距
        if s[0] <= s[8]:  # 主车车速小于前车
            if ss1 >= 0:  # 加速阶段车距大于等于0
                t3 = (v_max_limit - s[8]) / 3  # 计算减速时间
                if t3 > 0:  # 减速时间大于0
                    s2 = (s[8] + v_max_limit) * t3 / 2  # 计算减速距离
                    ss2 = s[10] + (t1 + t3) * s[8] - s1 - s2  # 计算完整加减速过程后车距
                    if ss2 >= 0:  # 如果车距大于等于0
                        t2 = ss2 / (v_max_limit - s[8] + 0.01)  # 计算持续高速时间
                        reward_speed = ((t1 + t2 + t3) * s[8] + s[10]) / (t1 + t2 + t3)
                        potential_reward = min(reward_speed, potential_reward_max)
                    if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                        t2_1 = round((((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                        # t2_1 = round(((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18), 2) ** 0.5
                        t1_1 = (s[8] - s[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s[8] + s[10]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                        # print(potential_reward,'line607')
                if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                    potential_reward = potential_reward_max
                    # print(potential_reward, 'line610')
            if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                t2_1 = round((((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                # t2_1 = ((s[10] / 3) + (((s[8] - s[0]) ** 2) / 18)) ** 0.5
                t1_1 = (s[8] - s[0]) / 3 + t2_1
                reward_speed = ((t1_1 + t2_1) * s[8] + s[10]) / (t1_1 + t2_1)
                potential_reward = min(reward_speed, potential_reward_max)
        if s[0] > s[8]:  # 主车车速大于前车
            t_brake = (s[0] - s[8]) / 3  # 计算紧急刹车时间
            s_brake = (s[0] + s[8]) / 2 * t_brake  # 紧急刹车距离
            ss_brake = s[8] * t_brake + s[10] - s_brake  # 刹车结束车距
            if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                if s[0] == v_max_limit:  # 此时只能匀速再刹车
                    t_keep = ss_brake / (s[0] - s[8])
                    reward_speed = ((t_brake + t_keep) * s[8] + s[10]) / (t_brake + t_keep)
                    potential_reward = min(reward_speed, potential_reward_max)
                if s[0] < v_max_limit:
                    ss_last = s[8] * (t_brake + t1) + s[10] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                    if ss_last > 0:  # 主车可以先加速再匀速再减速
                        t_keep = ss_last / (v_max_limit - s[8])  # 计算持续高速时间
                        reward_speed = ((t1 + t_keep + t_brake) * s[8] + s[10]) / (t1 + t_keep + t_brake)
                        potential_reward = min(reward_speed, potential_reward_max)
                    if ss_last <= 0:
                        t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                        reward_speed = ((t_brake + t_add) * s[8] + s[10]) / (t_brake + t_add)
                        potential_reward = min(reward_speed, potential_reward_max)
            if ss_brake < 0:
                potential_reward = 0
            if ss_brake == 0:
                reward_speed = s_brake / t_brake
                potential_reward = min(reward_speed, potential_reward_max)
        potential_reward_1 = potential_reward
        ##################
        ######车道2优势函数计算
        ss1 = s[13] + t1 * s[11] - s1  # 加速阶段结束车间距
        if s[0] <= s[11]:  # 主车车速小于前车
            if ss1 >= 0:  # 加速阶段车距大于等于0
                t3 = (v_max_limit - s[11]) / 3  # 计算减速时间
                if t3 > 0:  # 减速时间大于0
                    s2 = (s[11] + v_max_limit) * t3 / 2  # 计算减速距离
                    ss2 = s[13] + (t1 + t3) * s[11] - s1 - s2  # 计算完整加减速过程后车距
                    if ss2 >= 0:  # 如果车距大于等于0
                        t2 = ss2 / (v_max_limit - s[11] + 0.01)  # 计算持续高速时间
                        reward_speed = ((t1 + t2 + t3) * s[11] + s[13]) / (t1 + t2 + t3)
                        potential_reward = min(reward_speed, potential_reward_max)
                    if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                        t2_1 = round((((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5),
                                     2)
                        # t2_1 = round(((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18), 2) ** 0.5
                        t1_1 = (s[11] - s[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s[11] + s[13]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                        # print(potential_reward,'line607')
                if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                    potential_reward = potential_reward_max
                    # print(potential_reward, 'line610')
            if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                t2_1 = round((((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                # t2_1 = ((s[13] / 3) + (((s[11] - s[0]) ** 2) / 18)) ** 0.5
                t1_1 = (s[13] - s[0]) / 3 + t2_1
                reward_speed = ((t1_1 + t2_1) * s[11] + s[13]) / (t1_1 + t2_1)
                potential_reward = min(reward_speed, potential_reward_max)
        if s[0] > s[11]:  # 主车车速大于前车
            t_brake = (s[0] - s[13]) / 3  # 计算紧急刹车时间
            s_brake = (s[0] + s[13]) / 2 * t_brake  # 紧急刹车距离
            ss_brake = s[11] * t_brake + s[13] - s_brake  # 刹车结束车距
            if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                if s[0] == v_max_limit:  # 此时只能匀速再刹车
                    t_keep = ss_brake / (s[0] - s[11])
                    reward_speed = ((t_brake + t_keep) * s[11] + s[13]) / (t_brake + t_keep)
                    potential_reward = min(reward_speed, potential_reward_max)
                if s[0] < v_max_limit:
                    ss_last = s[11] * (t_brake + t1) + s[13] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                    if ss_last > 0:  # 主车可以先加速再匀速再减速
                        t_keep = ss_last / (v_max_limit - s[11])  # 计算持续高速时间
                        reward_speed = ((t1 + t_keep + t_brake) * s[11] + s[13]) / (t1 + t_keep + t_brake)
                        potential_reward = min(reward_speed, potential_reward_max)
                    if ss_last <= 0:
                        t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                        reward_speed = ((t_brake + t_add) * s[11] + s[13]) / (t_brake + t_add)
                        potential_reward = min(reward_speed, potential_reward_max)
            if ss_brake < 0:
                potential_reward = 0
            if ss_brake == 0:
                reward_speed = s_brake / t_brake
                potential_reward = min(reward_speed, potential_reward_max)
        potential_reward_2 = potential_reward
        #########下部分代码为换道状态赋值
        DQN_s[0] = s[0]
        DQN_s[1] = traci.vehicle.getAcceleration('%d' % ego_id)
        DQN_s[2] = s[1]
        DQN_s[3] = s[5]
        DQN_s[4] = s[6]
        DQN_s[5] = s[7]
        DQN_s[6] = s[8]
        DQN_s[7] = s[9]
        DQN_s[8] = s[10]
        DQN_s[9] = s[11]
        DQN_s[10] = s[12]
        DQN_s[11] = s[13]
        if lane_index == 0:
            DQN_s[12] = potential_reward_0 * 1.1+0.001
            # DQN_s[13] = potential_reward_1 * s[14]
            DQN_s[13] = potential_reward_1 * (1-left_lanechange_dangerious)
            DQN_s[14] = 0
        if lane_index == 1:
            # DQN_s[12] = potential_reward_0 * s[15]
            DQN_s[12] = potential_reward_0 * (1-right_lanechange_dangerious)
            DQN_s[13] = potential_reward_1 * 1.1+0.001
            # DQN_s[14] = potential_reward_2 * s[14]
            DQN_s[14] = potential_reward_2 * (1-left_lanechange_dangerious)
        if lane_index == 2:
            DQN_s[12] = 0
            # DQN_s[13] = potential_reward_1 * s[15]
            DQN_s[13] = potential_reward_1 * (1-right_lanechange_dangerious)
            DQN_s[14] = potential_reward_2 * 1.1+0.001






        ###选择车道
        a = RL.choose_action(DQN_s)
        changelane = 0  # 换道状态初始化
        dangerious_lc = 0
        r_danger = 0
        r0 = 0  # 换道惩罚初始化
        r_potential = 0
#########开始执行换道动作
        if a == 0:  ##去车道0
            ############对于车道0来说
            cf_s[0] = DQN_s[0]
            cf_s[1] = DQN_s[3]
            cf_s[2] = DQN_s[5]
            cf_s[3] = DQN_s[4]
            # print(cf_s[1])
            a_lane0 = actor.choose_action(cf_s)
            # a_lane0 = np.clip(np.random.normal(a_lane0, var), *ACTION_BOUND)
            if cf_s[2] <= 150:
                a_lane0 = np.clip(np.random.normal(a_lane0, var), *ACTION_BOUND)
            else:
                a_lane0 = 1
            DDPG_action = a_lane0
            traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane0),v_max_limit))
            # traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)
            # traci.vehicle.setSpeed('%d' % ID_list_order[w], -1)
            if carfollowing_learn_start==1:
                traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane0), v_max_limit))
            if DQN_s[12] == 0:
                r0 = 0
                r_danger = -100
            elif lane_index == 1:  # 主车由车道1换道0
                r_potential = DQN_s[12] - DQN_s[13]
                r0 = -1
                if lanechange_learn_start==1:
                    traci.vehicle.changeLane('%d' % ego_id, lane_index - 1, 1)  # 右换道
                    changelane = 1
        if a == 1:  ###去车道1
            ############对于车道1来说
            cf_s[0] = DQN_s[0]
            cf_s[1] = DQN_s[6]
            cf_s[2] = DQN_s[8]
            cf_s[3] = DQN_s[7]
            # print(cf_s[1])
            a_lane1 = actor.choose_action(cf_s)
            # a_lane1 = np.clip(np.random.normal(a_lane1, var), *ACTION_BOUND)
            if cf_s[2] <= 150:
                a_lane1 = np.clip(np.random.normal(a_lane1, var), *ACTION_BOUND)
            else:
                a_lane1 = 1
            DDPG_action = a_lane1
            traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane1),v_max_limit))
            # traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)
            # traci.vehicle.setSpeed('%d' % ID_list_order[w], -1)
            if carfollowing_learn_start == 1:
                traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane1), v_max_limit))
            if DQN_s[13] == 0:
                r0 = 0
                r_danger = -100
            elif lane_index == 0:  # 证明主车在车道0
                if lanechange_learn_start == 1:
                    traci.vehicle.changeLane('%d' % ego_id, lane_index + 1, 1)  # 左换道
                    changelane = 1
            elif lane_index == 2:  # 主车由车道2
                if lanechange_learn_start == 1:
                    traci.vehicle.changeLane('%d' % ego_id, lane_index - 1, 1)  # 右换道
                    changelane = 1
        if a == 2:  ###去车道2
            ############对于车道2来说
            cf_s[0] = DQN_s[0]
            cf_s[1] = DQN_s[9]
            cf_s[2] = DQN_s[11]
            cf_s[3] = DQN_s[10]
            # print(cf_s[1])
            a_lane2 = actor.choose_action(cf_s)
            # a_lane2 = np.clip(np.random.normal(a_lane2, var), *ACTION_BOUND)
            if cf_s[2] <= 150:
                a_lane2 = np.clip(np.random.normal(a_lane2, var), *ACTION_BOUND)
            else:
                a_lane2 = 1
            DDPG_action = a_lane2
            traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane2),v_max_limit))
            # traci.vehicle.setSpeedMode('%d' % ID_list_order[w], 31)
            # traci.vehicle.setSpeed('%d' % ID_list_order[w], -1)
            if carfollowing_learn_start == 1:
                traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane2), v_max_limit))
            if DQN_s[14] == 0:
                r0 = 0
                r_danger = -100
            elif lane_index == 1:  # 主车由车道1换道2
                if lanechange_learn_start == 1:
                    traci.vehicle.changeLane('%d' % ego_id, lane_index + 1, 1)  # 左换道
                    changelane = 1
        if r_danger == -100:
            dangerious_lc = 1

        ######################仿真到下一步
        lc_action_list.append(a)
        cf_action_list.append(DDPG_action)
        state_space.append(s.tolist())
        if len(lc_action_list)==3:
            del(lc_action_list[0])
        if len(cf_action_list)==3:
            del(cf_action_list[0])
        if len(state_space)<2:
            continue
        if len(state_space)==3:
            del(state_space[0])

        s = np.array(state_space[-2])
        s_ = np.array(state_space[-1])
        a = lc_action_list[-2]
        DDPG_action = cf_action_list[-2]
        # print(s_-s,a,DDPG_action)


        DDPG_s[0] = s[0]
        DDPG_s[1] = s[2]
        DDPG_s[2] = s[4]
        DDPG_s[3] = s[3]
        DDPG_s_[0] = s_[0]
        DDPG_s_[1] = s_[2]
        DDPG_s_[2] = s_[4]
        DDPG_s_[3] = s_[3]

        collision = 0
        distance_headway = s_[4]
        if distance_headway > 0:
            # r_dis = 1.1*(150-distance_headway)/75
            r_dis = 1
            if distance_headway >= 150:
                r_dis = -30
        if distance_headway <= 0:
            r_dis = min(10 * distance_headway, -100)
            collision = 1
            # print(j,s[4],s[7],s[10],s[13])

        cf_r_efficient = s_[0] / 33
        cf_r = 33 * cf_r_efficient * 0.6 + r_dis - 2.25 * (DDPG_s_[0] - DDPG_s[0])*(DDPG_s_[0] - DDPG_s[0])
        all_changelanetimes.append(changelane)
        r_efficient = s_[0]-11.6
        # r = -30 * collision + 0.8*33 * r_efficient + (-1) * changelane + r_dis*collision -2.25 * (DDPG_s_[0] - DDPG_s[0])*(DDPG_s_[0] - DDPG_s[0])
        r = -30 * collision + 1 * r_efficient + (-1) * changelane
        RL.store_transition(DQN_s, a, r, DQN_s_)
        if total_step > MEMORY_CAPACITY:
            RL.learn()

        # M.store_transition(DDPG_s, DDPG_action, cf_r, DDPG_s_)
        # if M.pointer > MEMORY_CAPACITY:
        #     var = max([var * 0.9995, VAR_MIN])  # decay the action randomness
        #     # var = var * 0.99995
        #     b_M = M.sample(BATCH_SIZE)
        #     b_s = b_M[:, :STATE_DIM]
        #     b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
        #     b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
        #     b_s_ = b_M[:, -STATE_DIM:]
        #     critic.learn(b_s, b_a, b_r, b_s_)
        #     actor.learn(b_s)

        all_cf_r.append(cf_r)
        cf_collision_list.append(collision)
        DQN_s = DQN_s_

        total_step += 1
        ep_step += 1
        all_r.append(r)
        rear_v_list.append(s_[0])
        l_v_list.append(s_[2])

        danger_lc.append(dangerious_lc)
        # lane_index0_acc_list.append(a_lane0)
        # lane_index1_acc_list.append(a_lane1)
        # lane_index2_acc_list.append(a_lane2)
        ep_step_list.append(j)
        distance_headway_list.append(s_[4])

        fleet_mean_speed.append(q2)
        horizontal_position_list.append(traci.vehicle.getPosition('%d'%ego_id)[1])
        lengthwise_position_list.append(traci.vehicle.getPosition('%d'%ego_id)[0])
        order_ego_id_list.append(order_ego_id)

    mean_speed = sum(rear_v_list) / j
    mean_speed_list.append(mean_speed)
    total_ep_list.append(i)
    total_lc_r_list.append(sum(all_r))
    total_mean_speed =sum(fleet_mean_speed)/j
    total_fleet_mean_speed.append(sum(fleet_mean_speed)/j)
    origin_speed = sum(origin_speed_ego_car_list)/len(origin_speed_ego_car_list)
    end_time=time.time()
    time_running = end_time-start_time
    print('episode=%s' % i, 'steps=%s' % j, 'reward=%s' % (sum(all_r)/len(all_r)), 'lanechange-times=%s' % sum(all_changelanetimes),
          'dangerious-times=%s' % sum(danger_lc), 'epsilon=%s' % RL.epsilon,'time_spend=%s' % time_running)
    # if i >= stable_episodes:
    print("car_following:", 'reward=%s' % (sum(all_cf_r)/len(all_cf_r)),'collision-times=%s' % sum(cf_collision_list),'explore=%s'% var,'mean_speed=%s' % mean_speed,'total_fleet_mean_speed=%s' % total_mean_speed,'origin_mean_speed=%s' %  origin_speed)


    # if mmm==1 and jjj==1 and sum(cf_collision_list) == 0 and sum(all_changelanetimes) <= last_save_lc and mean_speed >= last_save_speed:
    #     path4 = './coupling train 106'
    #     if os.path.isdir(path4): shutil.rmtree(path4)
    #     os.mkdir(path4)
    #     ckpt_path = os.path.join(path4, 'DuelingDQNPrioritizedReplay.ckpt')
    #     save_path = saver2.save(sess2, ckpt_path, write_meta_graph=False)
    #     print("\nSave Model %s\n" % save_path,'ep=%s'%i,'speed=%s'%mean_speed,'lc_time=%s'%sum(all_changelanetimes))


RL.epsilon=1
var=0.1

import pandas as pd
# from construct import set_trajectory,set_lane,set_speed,read_acc
def output_useful_id(vehicle_list=None):
    useless_car_list = []
    for vehicle_id in vehicle_list:
        total_vehicle_id = df[df['Vehicle_ID'] == vehicle_id]
        hangshu = total_vehicle_id.shape[0]
        if hangshu <= 29:
            useless_car_list.append(vehicle_id)
            useless_car = list(set(useless_car_list))
    use_list = list(set(vehicle_list) - set(useless_car))
    return use_list

def set_speed(time=None , vehicle_id=None):
    df_vehicle_id = df[df['Vehicle_ID']== vehicle_id]
    hangshu = df_vehicle_id.shape[0]
    if (time + 1) <= hangshu:
    # if time <= hangshu:
        # print(df_vehicle_id.iloc[time])
        vehicle_speed = df_vehicle_id.iloc[time]   #读取第time行
        traci.vehicle.setSpeed('%d' % vehicle_id, vehicle_speed['v_Vel'])
    if time == hangshu:
    # if time > hangshu:
        # traci.vehicle.remove('%d' % vehicle_id)
        traci.vehicle.setSpeed('%d' % vehicle_id, 33)
        # vehicle_speed = df_vehicle_id.iloc[hangshu-1]
        # traci.vehicle.setSpeed('%d' % vehicle_id, vehicle_speed['v_Vel'])
        # print(hangshu,time,vehicle_id)

def read_acc(time=None,vehicle_id=None):
    df_vehicle_id = df[df['Vehicle_ID'] == vehicle_id]
    hangshu = df_vehicle_id.shape[0]
    if (time + 1) <= hangshu:
    # if time <= hangshu:
        vehicle_speed = df_vehicle_id.iloc[time]  # 读取第time+1行
        vehicle_acc = vehicle_speed['v_Acc']
    if time >= hangshu:
    # if time > hangshu:
        vehicle_acc = max(min(33-traci.vehicle.getSpeed('%d' % vehicle_id),3),-3)
    return vehicle_acc
#
#
def set_trajectory(vehicle_id=None):
    # df_vehicle_id = df[df['Vehicle_ID'].isin(vehicle_id)]
    df_vehicle_id = df[df['Vehicle_ID'] == vehicle_id]
    df_vehicle_id = df_vehicle_id.iloc[0]
    vehicle_y_distance = 1.6-(abs(df_vehicle_id['Lane_ID']-2))*3.2
    vehicle_x_distance = (df_vehicle_id['Local_Y']-min_y) + last_y[0]
    traci.vehicle.moveToXY('%d' % vehicle_id,"gneE0", abs(df_vehicle_id['Lane_ID']-5), vehicle_x_distance+5, int(vehicle_y_distance),keepRoute=1)

def set_lane(time=None,vehicle_id=None):

    df_vehicle_id = df[df['Vehicle_ID'] == vehicle_id]
    hangshu2 = df_vehicle_id.shape[0]
    if (time + 1) <= hangshu2:
    # if time <= hangshu2:
        df_vehicle_id = df_vehicle_id.iloc[time]
        vehicle_lane = abs((df_vehicle_id['Lane_ID'])-5)
        vehicle_lane_before = traci.vehicle.getLaneIndex('%d' % vehicle_id)
        if vehicle_lane!=vehicle_lane_before:
            traci.vehicle.changeLane('%d' % vehicle_id,vehicle_lane,1)
            # print(vehicle_id)
    if time == hangshu2:
    # if time > hangshu2:
    #     # traci.vehicle.remove('%d' % vehicle_id)
    #
        traci.vehicle.setSpeed('%d' % vehicle_id, 33)
        # print(hangshu2,time,vehicle_id)
MAX_EPISODES = 2
MAX_EP_STEPS = 30

for sceen in range(15):
    # ego_id = 213
    sceen_id = sceen +1
    # df = pd.read_csv('./data/sumo-'+'%d'% sceen_id + 'us-101-'+'%d'% sceen_id + '.csv',usecols=["Vehicle_ID","Global_Time","Global_Y","Local_Y","v_Class","Location","v_Vel","Lane_ID","Preceding","Following","Space_Headway","Time_Headway","Location"])
    df = pd.read_csv('D:\\Project_codes of pycharm\\2021.8.12\\data\\'+'sumo-'+'%d'%sceen_id+'\\'+'us-101-'+'%d'% sceen_id + '.csv',
                     usecols=["Vehicle_ID", "Global_Time", "Global_Y", "Local_Y", "v_Class", "Location", "v_Vel",
                              "Lane_ID", "Preceding", "Following", "Space_Headway", "Time_Headway", "Location"])
    vehicle_list = df['Vehicle_ID'].unique()  #返回一个无重复元素（车辆ID）的列表
    min_y = df['Local_Y'].min()
    for vehicle in vehicle_list:
        y1 = df[df['Vehicle_ID'] == vehicle]
        mean_speed = sum(y1["v_Vel"]) / y1.shape[0]
        # print(vehicle,mean_speed)
        y2 = y1.iloc[0]
        y3 = y2['Local_Y']
        if y3 == min_y:
            last_vehicle = vehicle

    for ego_id in output_useful_id(vehicle_list):
        # if ego_id in wrong_collision_list:
        #     break
        y1 = df[df['Vehicle_ID'] == ego_id]
        ego_mean_speed = sum(y1["v_Vel"]) / y1.shape[0]
        start_IDM_follow = 0

        for i in range(MAX_EPISODES):
            ep_reward = 0
            ep_reward_all = 0
            ep_step = 0
            car_15_speed = []
            v = []
            all_r = []
            rear_v_list = []
            l_v_list = []
            all_changelanetimes = []
            all_changelanetimes.append(0)
            lane_index0_position_x = []
            lane_index0_id_list = []
            lane_index1_position_x = []
            lane_index1_id_list = []
            lane_index2_position_x = []
            lane_index2_id_list = []
            lane_index0_front = []
            lane_index1_front = []
            lane_index2_front = []
            # r_potential = []
            danger_lc = []
            all_cf_r = []
            cf_collision_list = []
            lane_index0_acc_list=[]
            lane_index1_acc_list=[]
            lane_index2_acc_list=[]
            ep_step_list=[]
            distance_headway_list = []
            dis_safe_list = []
            total_r = 0
            r0 = 0  # 换道惩罚初始化
            t = 0
            k = 1
            s = np.zeros(s_dim)
            s_ = np.zeros(s_dim)
            DQN_s = np.zeros(DQN_s_dim)
            DQN_s_ = np.zeros(DQN_s_dim)

            DDPG_s = np.zeros(4)
            DDPG_s_ = np.zeros(4)
            r_danger_lc = 0
            r_high_frequency_lc = 1
            potential_reward_2_list = []
            horizontal_position_list = []
            lengthwise_position_list = []
            order_ego_id_list = []

            # traci.load(["-c", "./data//car-lc.sumocfg"])
            traci.load(["-c", 'D:\\Project_codes of pycharm\\2021.8.12\data\\'+'sumo-'+'%d'%sceen_id+'\\'+'car-lc.sumocfg'])
            # traci.simulationStep(33)
            for step in range(39):
                ID_list_all = traci.edge.getLastStepVehicleIDs("gneE0")
                for vehicle in vehicle_list:
                    if vehicle in ID_list_all:
                        traci.vehicle.setSpeedMode('%d' % vehicle, 12)
                        traci.vehicle.setLaneChangeMode('%d' % vehicle, 512)  # 换道机制设定
                    # traci.vehicle.setSpeed('%d' % vehicle, 15)  # 初始速度设定
                        set_speed(time=1, vehicle_id=int(vehicle))
                traci.simulationStep(1+step)
            # print(traci.vehicle.getSpeed('%d'%4))
            for vehicle in vehicle_list:
                # traci.vehicle.setSpeedMode('%d' % vehicle, 12)
                # traci.vehicle.setLaneChangeMode('%d' % vehicle, 512)  # 换道机制设定
                # # traci.vehicle.setSpeed('%d' % vehicle, 15)  # 初始速度设定
                if vehicle == last_vehicle:
                    last_y = traci.vehicle.getPosition('%d' % vehicle)
            for vehicle in vehicle_list:
                # traci.vehicle.setSpeed('%d' % vehicle, 10)  # 初始速度设定
                # traci.vehicle.setSpeedMode('%d' % vehicle, 12)
                # traci.vehicle.setLaneChangeMode('%d' % vehicle, 512)  # 换道机制设定
                set_speed(time=1, vehicle_id=int(vehicle))
                set_trajectory(vehicle_id=int(vehicle))
            # traci.vehicle.setSpeedMode('%d' % ego_id, 12)
            traci.simulationStep(39)
            # for vehicle in vehicle_list:
            #     set_speed(time=1,vehicle_id=int(vehicle))
            # traci.simulationStep(40)
            # print(vehicle_list)
            # print(traci.vehicle.getSpeed('%d'%4))
            for j in range(MAX_EP_STEPS):

                all_car_position_x = []
                all_car_position_y = []
                lane_index0_position_x = []
                lane_index0_id_list = []
                lane_index1_position_x = []
                lane_index1_id_list = []
                lane_index2_position_x = []
                lane_index2_id_list = []
                lane_index0_front = []
                lane_index1_front = []
                lane_index2_front = []
                Distance_list = []
                all_speed_list = []
                cf_s = np.zeros(4)

                changelane = 0
                m = j
                r0 = 0
                ####这部分的代码作用是获取路网所有车辆，并按照行驶的位置进行前后排序
                ID_list_all = traci.edge.getLastStepVehicleIDs("gneE0")  # 获取主路所有车辆的ID编号
                for x in range(len(ID_list_all)):  # 长度
                    Distance_list.append(traci.vehicle.getDistance(ID_list_all[x]))  # 元素跑的里程
                Index = sorted(range(len(Distance_list)), key=lambda k: Distance_list[k], reverse=True)  # 距离表降序排列的索引
                Index = np.array(Index)
                ID_list_all = list(map(int, ID_list_all))  # 字符串转化为数值型
                ID_list_order = np.array(ID_list_all)[Index]  # 按照前后顺序进行车序排列

                ###排序完成之后对主车进行判断（是否主车已经超越所有的车辆成为第一辆车）和操作
                # if ego_id == ID_list_order[0]:  # 如果主车成为了头车
                #     ego_id = ID_list_order[-1]  # 主车的控制对象变成了最后一辆车
                for vehicle in (vehicle_list):
                    if vehicle != ego_id:

                        set_speed(time=j, vehicle_id=int(vehicle))
                        set_lane(time=j, vehicle_id=int(vehicle))
                        if start_IDM_follow ==1:
                            if traci.vehicle.getLaneIndex('%d' % vehicle) != traci.vehicle.getLaneIndex('%d' % ego_id):
                                set_speed(time=j, vehicle_id=int(vehicle))
                                set_lane(time=j, vehicle_id=int(vehicle))
                        # if traci.vehicle.getPosition('%d' % vehicle)[0] - traci.vehicle.getPosition('%d' % ego_id)[
                        #     0] >= 5:
                        #     traci.vehicle.setSpeedMode('%d' % vehicle, 12)
                        #     set_speed(time=j, vehicle_id=int(vehicle))
                        #     set_lane(time=j, vehicle_id=int(vehicle))
                            if 0<traci.vehicle.getPosition('%d' % ego_id)[0] -traci.vehicle.getPosition('%d' % vehicle)[0] < 2*traci.vehicle.getSpeed('%d' % vehicle):
                                traci.vehicle.setSpeedMode('%d' % vehicle, 31)
                                traci.vehicle.setSpeed('%d' % vehicle, -1)
                        #     if 0 < abs(traci.vehicle.getPosition('%d' % ego_id)[0] - traci.vehicle.getPosition('%d' % vehicle)[
                        #         0]) < 2*traci.vehicle.getSpeed('%d' % vehicle):
                        #         traci.vehicle.setSpeedMode('%d' % vehicle, 31)
                        #         traci.vehicle.setSpeed('%d' % vehicle,traci.vehicle.getSpeed('%d' % ego_id)-3)

                # traci.simulationStep(m + 36)

                # traci.vehicle.setLaneChangeMode('%d' % ego_id, 1621)  # 让主车可以自由进行换道
                traci.vehicle.setLaneChangeMode('%d' % ego_id, 512)  # 让主车可以自由进行换道
                traci.vehicle.setSpeedMode('%d' % ego_id, 31)
                ###获取状态值
                r_v = traci.vehicle.getSpeed('%d' % ego_id)  # 获取主车车速，状态0
                s[0] = r_v
                s[1] = traci.vehicle.getLaneIndex('%d' % ego_id)  # 获取车道数
                s[14] = traci.vehicle.couldChangeLane('%d' % ego_id, 1)  # 左换道可行性
                s[15] = traci.vehicle.couldChangeLane('%d' % ego_id, -1)  # 右换道可行性
                r_distance_value = max(traci.vehicle.getDistance('%d' % ego_id), 0)  # 获得主车行驶里程

                ######此部分代码用于每车道前后车排序
                for p in range(len(ID_list_order)):
                    if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -8:
                        lane_index0_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                        lane_index0_id_list.append(ID_list_order[p])
                    if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -4.8:
                        lane_index1_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                        lane_index1_id_list.append(ID_list_order[p])
                    if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -1.6:
                        lane_index2_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                        lane_index2_id_list.append(ID_list_order[p])
                lane_index = traci.vehicle.getLaneIndex('%d' % ego_id)  # 获得主车所在车道index
                position = traci.vehicle.getPosition('%d' % ego_id)  # 获得主车坐标
                position = [max(position[0], 0), max(position[1], -8)]

                for p in range(len(lane_index0_position_x)):

                    if lane_index0_position_x[p] <= position[0] + 5:
                        break
                    lane_index0_front.append(lane_index0_id_list[p])  # int
                for p in range(len(lane_index1_position_x)):

                    if lane_index1_position_x[p] <= position[0] + 5:
                        break
                    lane_index1_front.append(lane_index1_id_list[p])
                for p in range(len(lane_index2_position_x)):

                    if lane_index2_position_x[p] <= position[0] + 5:
                        break
                    lane_index2_front.append(lane_index2_id_list[p])
                s[5] = 33
                s[6] = 0.01
                s[7] = 150
                s[8] = 33
                s[9] = 0.01
                s[10] = 150
                s[11] = 33
                s[12] = 0.01
                s[13] = 150
                for w in range(len(ID_list_all)):
                    if lane_index0_front:
                        if ID_list_order[w] == int(lane_index0_front[-1]):

                            lane_index0_v = traci.vehicle.getSpeed('%d' % lane_index0_front[-1])
                            # print(lane_index0_v)
                            if df[df['Vehicle_ID'] == int(lane_index0_front[-1])].shape[0] <=j+1:
                                lane_index0_acc = traci.vehicle.getAcceleration('%d' % lane_index0_front[-1])
                            else:
                                lane_index0_acc = (df[df['Vehicle_ID'] == int(lane_index0_front[-1])].iloc[j+1])['v_Vel'] - lane_index0_v
                                # print(lane_index0_acc,lane_index0_front[-1],j,lane_index0_v)
                            # print(lane_index0_acc)
                            # lane_index0_acc = y[0, m + 36 - w] - lane_index0_v
                            lane_index0_s = traci.vehicle.getPosition('%d' % lane_index0_front[-1])[0] - \
                                            traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                            # lane_index0_s = traci.vehicle.getDistance('%d' % lane_index0_front[-1]) - r_distance_value
                            if lane_index0_s <= 150:
                                s[5] = lane_index0_v
                                s[6] = lane_index0_acc
                                s[7] = lane_index0_s

                    if lane_index1_front:
                        if ID_list_order[w] == int(lane_index1_front[-1]):
                            lane_index1_v = traci.vehicle.getSpeed('%d' % lane_index1_front[-1])
                            # print(df[df['Vehicle_ID'] == int(lane_index1_front[-1])].shape[0],j)
                            if (df[df['Vehicle_ID'] == int(lane_index1_front[-1])].shape[0]) <= j+1:
                                lane_index1_acc = traci.vehicle.getAcceleration('%d' % lane_index1_front[-1])

                            else:
                                # print(int(lane_index1_front[-1]))
                                # print(df[df['Vehicle_ID'] == int(lane_index1_front[-1])].shape[0], j)
                                # print((df[df['Vehicle_ID'] == int(lane_index1_front[-1])]['v_Vel'].iloc[j]))


                                lane_index1_acc = (df[df['Vehicle_ID'] == int(lane_index1_front[-1])]['v_Vel'].iloc[j+1]) - lane_index1_v
                            # lane_index1_acc = y[0, m + 36 - w] - lane_index1_v
                            # lane_index1_s = traci.vehicle.getDistance(lane_index1_front[-1]) - r_distance_value
                            lane_index1_s = traci.vehicle.getPosition('%d' % lane_index1_front[-1])[0] - \
                                            traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                            if lane_index1_s <= 150:
                                s[8] = lane_index1_v
                                s[9] = lane_index1_acc
                                s[10] = lane_index1_s
                    if lane_index2_front:
                        if ID_list_order[w] == int(lane_index2_front[-1]):
                            lane_index2_v = traci.vehicle.getSpeed('%d' % lane_index2_front[-1])
                            # print(lane_index2_v)
                            if df[df['Vehicle_ID'] == int(lane_index2_front[-1])].shape[0] <=j+1:
                                lane_index2_acc = traci.vehicle.getAcceleration('%d' % lane_index2_front[-1])
                            else:
                                lane_index2_acc = (df[df['Vehicle_ID'] == int(lane_index2_front[-1])].iloc[j+1])['v_Vel'] - lane_index2_v
                            # lane_index2_acc = y[0, m + 36 - w] - lane_index2_v
                            lane_index2_s = traci.vehicle.getPosition('%d' % lane_index2_front[-1])[0] - \
                                            traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                            # lane_index2_s = traci.vehicle.getDistance(lane_index2_front[-1]) - r_distance_value
                            if lane_index2_s <= 150:
                                s[11] = lane_index2_v
                                s[12] = lane_index2_acc
                                s[13] = lane_index2_s

                if traci.vehicle.getLeader('%d' % ego_id):  # 如果前车存在
                    or_gap_1 = traci.vehicle.getLeader('%d' % ego_id)  # 获得前车的ID和车距
                    ID = or_gap_1[0]  # 获得前车的ID

                    l_v = traci.vehicle.getSpeed(ID)  # 获得前车车速
                    if df[df['Vehicle_ID'] == int(ID)].shape[0]<=j+1:
                        l_acc = traci.vehicle.getAcceleration(ID)
                    else:
                        l_acc = (df[df['Vehicle_ID'] == int(ID)].iloc[j+1])['v_Vel'] - l_v

                    # for w in range(len(ID_list_all)):
                    #     if ID_list_order[w] == int(ID):
                    #         l_acc = y[0, m + 36 - w] - l_v
                    #         break
                    s[3] = l_acc
                    s[4] = traci.vehicle.getPosition(ID)[0] - traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                    # s[4] = or_gap_1[-1]  # 前车车距赋值给换道状态3
                    s[2] = l_v  # 前车车速赋值给换道状态2


                    if s[4] > 150:
                        s[2] = 33
                        s[3] = 0.01
                        s[4] = 150
                else:  # 如果前车不存在
                    or_gap = 150  # 前车距离150
                    l_v = 33  # 前车车速30
                    l_acc = 0.01
                    s[3] = l_acc
                    s[4] = or_gap  # 前车车距赋值给换道状态3
                    s[2] = l_v
                # print(s[6], s[9], s[12], s[3],ego_id,j)
                # print(s[7], s[10], s[13],s[4], or_gap_1[-1])
                # 此部分用于计算奖励时间
                for vehicle in vehicle_list:
                    # if p != ego_id:
                    #     all_speed_list.append(traci.vehicle.getSpeed('%d' % p))
                    all_speed_list.append(traci.vehicle.getSpeed('%d' % vehicle))
                # print(all_speed_list)
                sorted(all_speed_list, reverse=False)  ##列表升序

                alist = numpy.array(all_speed_list)
                # print(alist)
                q1 = numpy.percentile(alist, 25)
                q2 = numpy.percentile(alist, 50)
                q3 = numpy.percentile(alist, 75)
                iqr = q3 - q1
                q_low = q1 - (1.5 * iqr)
                q_85 = numpy.percentile(alist, 85)  # 85位车速

                t0 = (q_85 - s[0]) / 3  # 计算加速到85位车速时间
                v_max_limit = 33
                potential_reward_max = 33
                # if t0 > 0 and s[2]>q_85:  #如果主车车速小于85位车速且前车大于85位车速
                t1 = (v_max_limit - s[0]) / 3  # 加速到最高限速时间
                s1 = (s[0] + v_max_limit) * t1 * 0.5  # 加速距离
                ##############车道0的优势函数计算
                ss1 = s[7] + t1 * s[5] - s1  # 加速阶段结束车间距
                if s[0] <= s[5]:  # 主车车速小于前车
                    if ss1 >= 0:  # 加速阶段车距大于等于0
                        t3 = (v_max_limit - s[5]) / 3  # 计算减速时间
                        if t3 > 0:  # 减速时间大于0
                            s2 = (s[5] + v_max_limit) * t3 / 2  # 计算减速距离
                            ss2 = s[7] + (t1 + t3) * s[5] - s1 - s2  # 计算完整加减速过程后车距
                            if ss2 >= 0:  # 如果车距大于等于0
                                t2 = ss2 / (v_max_limit - s[5] + 0.01)  # 计算持续高速时间
                                reward_speed = ((t1 + t2 + t3) * s[5] + s[7]) / (t1 + t2 + t3)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                                t2_1 = round(
                                    (((s[7] / 3) + round(((round((s[5] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                                t1_1 = (s[5] - s[0]) / 3 + t2_1
                                reward_speed = ((t1_1 + t2_1) * s[5] + s[7]) / (t1_1 + t2_1)
                                potential_reward = min(reward_speed, potential_reward_max)
                                # print(potential_reward,'line607')
                        if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                            potential_reward = potential_reward_max
                            # print(potential_reward, 'line610')
                    if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                        t2_1 = round((((s[7] / 3) + round(((round((s[5] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                        # t2_1 = ((s[7] / 3) + (((s[5] - s[0]) ** 2) / 18)) ** 0.5
                        t1_1 = (s[5] - s[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s[5] + s[7]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                if s[0] > s[5]:  # 主车车速大于前车
                    t_brake = (s[0] - s[5]) / 3  # 计算紧急刹车时间
                    s_brake = (s[0] + s[5]) / 2 * t_brake  # 紧急刹车距离
                    ss_brake = s[5] * t_brake + s[7] - s_brake  # 刹车结束车距
                    if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                        if s[0] == v_max_limit:  # 此时只能匀速再刹车
                            t_keep = ss_brake / (s[0] - s[5])
                            reward_speed = ((t_brake + t_keep) * s[5] + s[7]) / (t_brake + t_keep)
                            potential_reward = min(reward_speed, potential_reward_max)
                        if s[0] < v_max_limit:
                            ss_last = s[5] * (t_brake + t1) + s[7] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                            if ss_last > 0:  # 主车可以先加速再匀速再减速
                                t_keep = ss_last / (v_max_limit - s[5])  # 计算持续高速时间
                                reward_speed = ((t1 + t_keep + t_brake) * s[5] + s[7]) / (t1 + t_keep + t_brake)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss_last <= 0:
                                t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                                reward_speed = ((t_brake + t_add) * s[5] + s[7]) / (t_brake + t_add)
                                potential_reward = min(reward_speed, potential_reward_max)
                    if ss_brake < 0:
                        potential_reward = 0
                    if ss_brake == 0:
                        reward_speed = s_brake / t_brake
                        potential_reward = min(reward_speed, potential_reward_max)
                potential_reward_0 = potential_reward
                #########车道1的优势函数计算
                ss1 = s[10] + t1 * s[8] - s1  # 加速阶段结束车间距
                if s[0] <= s[8]:  # 主车车速小于前车
                    if ss1 >= 0:  # 加速阶段车距大于等于0
                        t3 = (v_max_limit - s[8]) / 3  # 计算减速时间
                        if t3 > 0:  # 减速时间大于0
                            s2 = (s[8] + v_max_limit) * t3 / 2  # 计算减速距离
                            ss2 = s[10] + (t1 + t3) * s[8] - s1 - s2  # 计算完整加减速过程后车距
                            if ss2 >= 0:  # 如果车距大于等于0
                                t2 = ss2 / (v_max_limit - s[8] + 0.01)  # 计算持续高速时间
                                reward_speed = ((t1 + t2 + t3) * s[8] + s[10]) / (t1 + t2 + t3)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                                t2_1 = round(
                                    (((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                                # t2_1 = round(((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18), 2) ** 0.5
                                t1_1 = (s[8] - s[0]) / 3 + t2_1
                                reward_speed = ((t1_1 + t2_1) * s[8] + s[10]) / (t1_1 + t2_1)
                                potential_reward = min(reward_speed, potential_reward_max)
                                # print(potential_reward,'line607')
                        if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                            potential_reward = potential_reward_max
                            # print(potential_reward, 'line610')
                    if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                        t2_1 = round((((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                        # t2_1 = ((s[10] / 3) + (((s[8] - s[0]) ** 2) / 18)) ** 0.5
                        t1_1 = (s[8] - s[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s[8] + s[10]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                if s[0] > s[8]:  # 主车车速大于前车
                    t_brake = (s[0] - s[8]) / 3  # 计算紧急刹车时间
                    s_brake = (s[0] + s[8]) / 2 * t_brake  # 紧急刹车距离
                    ss_brake = s[8] * t_brake + s[10] - s_brake  # 刹车结束车距
                    if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                        if s[0] == v_max_limit:  # 此时只能匀速再刹车
                            t_keep = ss_brake / (s[0] - s[8])
                            reward_speed = ((t_brake + t_keep) * s[8] + s[10]) / (t_brake + t_keep)
                            potential_reward = min(reward_speed, potential_reward_max)
                        if s[0] < v_max_limit:
                            ss_last = s[8] * (t_brake + t1) + s[10] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                            if ss_last > 0:  # 主车可以先加速再匀速再减速
                                t_keep = ss_last / (v_max_limit - s[8])  # 计算持续高速时间
                                reward_speed = ((t1 + t_keep + t_brake) * s[8] + s[10]) / (t1 + t_keep + t_brake)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss_last <= 0:
                                t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                                reward_speed = ((t_brake + t_add) * s[8] + s[10]) / (t_brake + t_add)
                                potential_reward = min(reward_speed, potential_reward_max)
                    if ss_brake < 0:
                        potential_reward = 0
                    if ss_brake == 0:
                        reward_speed = s_brake / t_brake
                        potential_reward = min(reward_speed, potential_reward_max)
                potential_reward_1 = potential_reward
                ##################

                ######车道2优势函数计算
                ss1 = s[13] + t1 * s[11] - s1  # 加速阶段结束车间距
                if s[0] <= s[11]:  # 主车车速小于前车
                    if ss1 >= 0:  # 加速阶段车距大于等于0
                        t3 = (v_max_limit - s[11]) / 3  # 计算减速时间
                        if t3 > 0:  # 减速时间大于0
                            s2 = (s[11] + v_max_limit) * t3 / 2  # 计算减速距离
                            ss2 = s[13] + (t1 + t3) * s[11] - s1 - s2  # 计算完整加减速过程后车距
                            if ss2 >= 0:  # 如果车距大于等于0
                                t2 = ss2 / (v_max_limit - s[11] + 0.01)  # 计算持续高速时间
                                reward_speed = ((t1 + t2 + t3) * s[11] + s[13]) / (t1 + t2 + t3)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                                t2_1 = round(
                                    (((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                                # t2_1 = round(((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18), 2) ** 0.5
                                t1_1 = (s[11] - s[0]) / 3 + t2_1
                                reward_speed = ((t1_1 + t2_1) * s[11] + s[13]) / (t1_1 + t2_1)
                                potential_reward = min(reward_speed, potential_reward_max)
                                # print(potential_reward,'line607')
                        if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                            potential_reward = potential_reward_max
                            # print(potential_reward, 'line610')
                    if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                        t2_1 = round((((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5),
                                     2)
                        # t2_1 = ((s[13] / 3) + (((s[11] - s[0]) ** 2) / 18)) ** 0.5
                        t1_1 = (s[13] - s[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s[11] + s[13]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                if s[0] > s[11]:  # 主车车速大于前车
                    t_brake = (s[0] - s[13]) / 3  # 计算紧急刹车时间
                    s_brake = (s[0] + s[13]) / 2 * t_brake  # 紧急刹车距离
                    ss_brake = s[11] * t_brake + s[13] - s_brake  # 刹车结束车距
                    if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                        if s[0] == v_max_limit:  # 此时只能匀速再刹车
                            t_keep = ss_brake / (s[0] - s[11])
                            reward_speed = ((t_brake + t_keep) * s[11] + s[13]) / (t_brake + t_keep)
                            potential_reward = min(reward_speed, potential_reward_max)
                        if s[0] < v_max_limit:
                            ss_last = s[11] * (t_brake + t1) + s[13] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                            if ss_last > 0:  # 主车可以先加速再匀速再减速
                                t_keep = ss_last / (v_max_limit - s[11])  # 计算持续高速时间
                                reward_speed = ((t1 + t_keep + t_brake) * s[11] + s[13]) / (t1 + t_keep + t_brake)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss_last <= 0:
                                t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                                reward_speed = ((t_brake + t_add) * s[11] + s[13]) / (t_brake + t_add)
                                potential_reward = min(reward_speed, potential_reward_max)
                    if ss_brake < 0:
                        potential_reward = 0
                    if ss_brake == 0:
                        reward_speed = s_brake / t_brake
                        potential_reward = min(reward_speed, potential_reward_max)
                potential_reward_2 = potential_reward
                # potential_reward_2_list.append(potential_reward_2)
                #########下部分代码为换道状态赋值
                DQN_s[0] = s[0]
                DQN_s[1] = traci.vehicle.getAcceleration('%d' % ego_id)
                DQN_s[2] = s[1]
                DQN_s[3] = s[5]
                DQN_s[4] = s[6]
                DQN_s[5] = s[7]
                DQN_s[6] = s[8]
                DQN_s[7] = s[9]
                DQN_s[8] = s[10]
                DQN_s[9] = s[11]
                DQN_s[10] = s[12]
                DQN_s[11] = s[13]
                if lane_index == 0:
                    DQN_s[12] = potential_reward_0 * 1.1 + 0.001
                    DQN_s[13] = potential_reward_1 * s[14]
                    DQN_s[14] = 0

                if lane_index == 1:
                    DQN_s[12] = potential_reward_0 * s[15]
                    DQN_s[13] = potential_reward_1 * 1.1 + 0.001
                    DQN_s[14] = potential_reward_2 * s[14]

                if lane_index == 2:
                    DQN_s[12] = 0
                    DQN_s[13] = potential_reward_1 * s[15]
                    DQN_s[14] = potential_reward_2 * 1.1 + 0.001

                r0 = 0  # 换道惩罚初始化
                changelane = 0  # 换道状态初始化
                r_danger = 0
                r_potential = 0

        ############对于车道0来说
                cf_s[0] = DQN_s[0]
                cf_s[1] = DQN_s[3]
                cf_s[2] = DQN_s[5]
                cf_s[3] = DQN_s[4]

                a_lane0 = actor.choose_action(cf_s)

                a_lane0 = np.clip(np.random.normal(a_lane0, var), *ACTION_BOUND)
        ############对于车道1来说
                cf_s[0] = DQN_s[0]
                cf_s[1] = DQN_s[6]
                cf_s[2] = DQN_s[8]
                cf_s[3] = DQN_s[7]

                a_lane1 = actor.choose_action(cf_s)

                a_lane1 = np.clip(np.random.normal(a_lane1, var), *ACTION_BOUND)
        ############对于车道2来说
                cf_s[0] = DQN_s[0]
                cf_s[1] = DQN_s[9]
                cf_s[2] = DQN_s[11]
                cf_s[3] = DQN_s[10]

                a_lane2 = actor.choose_action(cf_s)

                a_lane2 = np.clip(np.random.normal(a_lane2, var), *ACTION_BOUND)
                # if i >= stable_episodes:
                #     DQN_s[1] = max(a_lane0,a_lane1,a_lane2)
                # DQN_s[15] = all_changelanetimes[-1]
                a = RL.choose_action(DQN_s)
                dangerious_lc = 0
                if a == 0:  ##去车道0
                    # if i >= stable_episodes:
                    DDPG_action = a_lane0
                    traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane0),v_max_limit))
                    # traci.vehicle.setSpeed('%d' % ego_id,-1)
                    if DQN_s[12] == 0:
                        r0 = 0
                        r_danger = -100
                    elif lane_index == 0:  # 证明主车就在车道0
                        r_potential = DQN_s[12] - DQN_s[13]
                    elif lane_index == 1:  # 主车由车道1换道0
                        r_potential = DQN_s[12] - DQN_s[13]
                        r0 = -1
                        traci.vehicle.changeLane('%d' % ego_id, lane_index - 1, 1)  # 右换道
                        changelane = 1
                if a == 1:  ###去车道1
                    # if i >= stable_episodes:
                    DDPG_action = a_lane1
                    traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane1),v_max_limit))
                    # traci.vehicle.setSpeed('%d' % ego_id, -1)
                    if DQN_s[13] == 0:
                        r0 = 0
                        r_danger = -100
                    elif lane_index == 0:  # 证明主车在车道0
                        r_potential = DQN_s[13] - DQN_s[12]
                        r0 = -1
                        traci.vehicle.changeLane('%d' % ego_id, lane_index + 1, 1)  # 左换道
                        changelane = 1
                    elif lane_index == 1:  # 主车由车道1保持
                        r_potential = min(DQN_s[13] - DQN_s[12], DQN_s[13] - DQN_s[14])
                    elif lane_index == 2:  # 主车由车道2
                        r_potential = DQN_s[13] - DQN_s[14]
                        r0 = -1
                        traci.vehicle.changeLane('%d' % ego_id, lane_index - 1, 1)  # 右换道
                        changelane = 1
                if a == 2:  ###去车道2
                    # if i >= stable_episodes:
                    DDPG_action = a_lane2
                    traci.vehicle.setSpeed('%d' % ego_id, min((DQN_s[0] + 3 * a_lane2),v_max_limit))
                    # traci.vehicle.setSpeed('%d' % ego_id, -1)
                    if DQN_s[14] == 0:
                        r0 = 0
                        r_danger = -100
                    elif lane_index == 2:  # 证明主车就在车道2
                        r_potential = DQN_s[14] - DQN_s[13]
                    elif lane_index == 1:  # 主车由车道1换道2
                        r_potential = DQN_s[14] - DQN_s[13]
                        r0 = -1
                        traci.vehicle.changeLane('%d' % ego_id, lane_index + 1, 1)  # 左换道
                        changelane = 1
                if r_danger == -100:
                    dangerious_lc = 1
                # if dangerious_lc==1 and i>=MAX_EPISODES-200:
                # if dangerious_lc == 1 :
                #     break
                traci.simulationStep(j + 40)

                s_ = s
                # s_[0] = update_speed
                s_[0] = traci.vehicle.getSpeed('%d' % ego_id)  # 主车车速更新
                # print(s_[0])
                s_[0] = max(s_[0], 1)
                s_[1] = traci.vehicle.getLaneIndex('%d' % ego_id)
                ID_list_all = []
                ID_list_order = []
                Distance_list = []
                or_gap_1 = []
                lane_index0_position_x = []
                lane_index0_id_list = []
                lane_index1_position_x = []
                lane_index1_id_list = []
                lane_index2_position_x = []
                lane_index2_id_list = []
                lane_index0_front = []
                lane_index1_front = []
                lane_index2_front = []
                ID_list_all = traci.edge.getLastStepVehicleIDs("gneE0")  # 获取主路所有车辆的ID编号
                for x in range(len(ID_list_all)):  # 长度应该为36
                    Distance_list.append(traci.vehicle.getDistance(ID_list_all[x]))  # 36个元素跑的里程
                Index = sorted(range(len(Distance_list)), key=lambda k: Distance_list[k], reverse=True)  # 距离表降序排列的索引
                Index = np.array(Index)
                ID_list_all = list(map(int, ID_list_all))  # 字符串转化为数值型
                ID_list_order = np.array(ID_list_all)[Index]  # 按照前后顺序进行车序排列
                ID_list_order_list = ID_list_order.tolist()
                order_ego_id = ID_list_order_list.index(ego_id)
                for p in range(len(ID_list_order)):
                    if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -8:
                        lane_index0_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                        lane_index0_id_list.append(ID_list_order[p])
                    if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -4.8:
                        lane_index1_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                        lane_index1_id_list.append(ID_list_order[p])
                    if traci.vehicle.getPosition('%d' % ID_list_order[p])[1] == -1.6:
                        lane_index2_position_x.append(traci.vehicle.getPosition('%d' % ID_list_order[p])[0])
                        lane_index2_id_list.append(ID_list_order[p])
                lane_index = traci.vehicle.getLaneIndex('%d' % ego_id)  # 获得主车所在车道index
                position = traci.vehicle.getPosition('%d' % ego_id)  # 获得主车坐标
                # s[16] = (8-abs(position[1]))/3.2  # 获得主车所在车道的状态
                position = [max(position[0], 0), max(position[1], -8)]
                for p in range(len(lane_index0_position_x)):
                    if lane_index0_position_x[p] <= position[0]+5:
                        break
                    lane_index0_front.append(lane_index0_id_list[p])
                for p in range(len(lane_index1_position_x)):

                    if lane_index1_position_x[p] <= position[0]+5:
                        break
                    lane_index1_front.append(lane_index1_id_list[p])
                for p in range(len(lane_index2_position_x)):

                    if lane_index2_position_x[p] <= position[0]+5:
                        break
                    lane_index2_front.append(lane_index2_id_list[p])
                s_[5] = 33
                s_[6] = 0.01
                s_[7] = 150
                s_[8] = 33
                s_[9] = 0.01
                s_[10] = 150
                s_[11] = 33
                s_[12] = 0.01
                s_[13] = 150
                for w in range(len(ID_list_all)):
                    if lane_index0_front:
                        if ID_list_order[w] == int(lane_index0_front[-1]):
                            lane_index0_v = traci.vehicle.getSpeed('%d' % lane_index0_front[-1])
                            if df[df['Vehicle_ID'] == int(lane_index0_front[-1])].shape[0] <j+1:
                                lane_index0_acc = traci.vehicle.getAcceleration('%d' % lane_index0_front[-1])
                            else:
                                lane_index0_acc = (df[df['Vehicle_ID'] == int(lane_index0_front[-1])].iloc[j])['v_Vel'] - lane_index0_v
                            # lane_index0_acc = y[0, m + 37 - w] - lane_index0_v
                            lane_index0_s = traci.vehicle.getPosition('%d' % lane_index0_front[-1])[0] - \
                                            traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                            # lane_index0_s = traci.vehicle.getDistance(lane_index0_front[-1]) - r_distance_value
                            if lane_index0_s <= 150:
                                s_[5] = lane_index0_v
                                s_[6] = lane_index0_acc
                                s_[7] = lane_index0_s
                    if lane_index1_front:
                        if ID_list_order[w] == int(lane_index1_front[-1]):
                            lane_index1_v = traci.vehicle.getSpeed('%d' % lane_index1_front[-1])
                            if df[df['Vehicle_ID'] == int(lane_index1_front[-1])].shape[0] <j+1:
                                lane_index1_acc = traci.vehicle.getAcceleration('%d' % lane_index1_front[-1])
                            else:
                                lane_index1_acc = (df[df['Vehicle_ID'] == int(lane_index1_front[-1])].iloc[j])['v_Vel'] - lane_index1_v
                            # lane_index1_acc = y[0, m + 37 - w] - lane_index1_v
                            lane_index1_s = traci.vehicle.getPosition('%d' % lane_index1_front[-1])[0] - \
                                            traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                            # lane_index1_s = traci.vehicle.getDistance(lane_index1_front[-1]) - r_distance_value
                            if lane_index1_s <= 150:
                                s_[8] = lane_index1_v
                                s_[9] = lane_index1_acc
                                s_[10] = lane_index1_s
                    if lane_index2_front:
                        if ID_list_order[w] == int(lane_index2_front[-1]):
                            lane_index2_v = traci.vehicle.getSpeed('%d' % lane_index2_front[-1])
                            if df[df['Vehicle_ID'] == int(lane_index2_front[-1])].shape[0] <j+1:
                                lane_index2_acc = traci.vehicle.getAcceleration('%d' % lane_index2_front[-1])
                            else:
                                lane_index2_acc = (df[df['Vehicle_ID'] == int(lane_index2_front[-1])].iloc[j])['v_Vel'] - lane_index2_v
                            # lane_index2_acc = y[0, m + 37 - w] - lane_index2_v
                            # lane_index2_s = traci.vehicle.getDistance(lane_index2_front[-1]) - r_distance_value
                            lane_index2_s = traci.vehicle.getPosition('%d' % lane_index2_front[-1])[0] - \
                                            traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                            if lane_index2_s <= 150:
                                s_[11] = lane_index2_v
                                s_[12] = lane_index2_acc
                                s_[13] = lane_index2_s

                if traci.vehicle.getLeader('%d' % ego_id):  # 如果前车存在
                    or_gap_1 = traci.vehicle.getLeader('%d' % ego_id)  # 获得前车的ID和车距
                    ID = or_gap_1[0]  # 获得前车的ID

                    l_v = traci.vehicle.getSpeed(ID)  # 获得前车车速
                    if df[df['Vehicle_ID'] == int(ID)].shape[0]<j+1:
                        l_acc = traci.vehicle.getAcceleration(ID)
                    else:
                        l_acc = (df[df['Vehicle_ID'] == int(ID)].iloc[j])['v_Vel'] - l_v
                    # # l_acc = exec('acc_%d' % (int(ID)))
                    # # s[3] = l_acc
                    # for w in range(len(ID_list_all)):
                    #     if ID_list_order[w] == int(ID):
                    #         l_acc = y[0, m + 37 - w] - l_v
                    #         break
                    s_[3] = l_acc
                    s_[4] = traci.vehicle.getPosition(ID)[0] - traci.vehicle.getPosition('%d' % ego_id)[0] - 5
                    # s_[4] = or_gap_1[-1]  # 前车车距赋值给换道状态3
                    s_[2] = l_v  # 前车车速赋值给换道状态2
                    if s_[4] > 150:
                        s_[2] = 33
                        s_[3] = 0.01
                        s_[4] = 150
                else:  # 如果前车不存在
                    or_gap = 150  # 前车距离150
                    l_v = 33  # 前车车速30
                    l_acc = 0.01
                    s_[3] = l_acc
                    s_[4] = or_gap  # 前车车距赋值给换道状态3
                    s_[2] = l_v
                s_[14] = traci.vehicle.couldChangeLane('%d' % ego_id, 1)  # 左换道可行性
                s_[15] = traci.vehicle.couldChangeLane('%d' % ego_id, -1)  # 右换道可行性
                # s_[16] = a[1]  # 横向平均速度
                ######状态更新部分
                for vehicle in vehicle_list:
                    # if p != ego_id:
                    #     all_speed_list.append(traci.vehicle.getSpeed('%d' % p))
                    all_speed_list.append(traci.vehicle.getSpeed('%d' % vehicle))
                # print(all_speed_list)
                sorted(all_speed_list, reverse=False)  ##列表升序

                alist = numpy.array(all_speed_list)
                # print(alist)
                q1 = numpy.percentile(alist, 25)
                q2 = numpy.percentile(alist, 50)
                q3 = numpy.percentile(alist, 75)
                iqr = q3 - q1
                q_low = q1 - (1.5 * iqr)
                q_85 = numpy.percentile(alist, 85)  # 85位车速

                t0 = (q_85 - s_[0]) / 3  # 计算加速到85位车速时间
                # if t0 > 0 and s_[2]>q_85:  #如果主车车速小于85位车速且前车大于85位车速
                t1 = (33 - s_[0]) / 3
                s1 = (s_[0] + 33) * t1 * 0.5
                ##############车道0的优势函数计算
                ss1 = s_[7] + t1 * s_[5] - s1  # 加速阶段结束车间距
                if s_[0] <= s_[5]:  # 主车车速小于前车
                    if ss1 >= 0:  # 加速阶段车距大于等于0
                        t3 = (v_max_limit - s[5]) / 3  # 计算减速时间
                        if t3 > 0:  # 减速时间大于0
                            s2 = (s_[5] + v_max_limit) * t3 / 2  # 计算减速距离
                            ss2 = s_[7] + (t1 + t3) * s_[5] - s1 - s2  # 计算完整加减速过程后车距
                            if ss2 >= 0:  # 如果车距大于等于0
                                t2 = ss2 / (v_max_limit - s_[5] + 0.01)  # 计算持续高速时间
                                reward_speed = ((t1 + t2 + t3) * s_[5] + s_[7]) / (t1 + t2 + t3)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                                # print(s_[0],s_[5],s_[7])
                                t2_1 = (math.ceil(((math.ceil((s_[5] - s_[0]) + 0.01) ** 2) / 18 + (s_[7] / 3)))) ** 0.5
                                # t2_1 = round((((s[7] / 3) + round(((round((s[5] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5),2)
                                # t2_1 = round(((s[7] / 3) + round(((round((s[5] - s[0]), 2) + 0.01) ** 2), 2) / 18), 2) ** 0.5
                                t1_1 = (s_[5] - s_[0]) / 3 + t2_1
                                reward_speed = ((t1_1 + t2_1) * s_[5] + s_[7]) / (t1_1 + t2_1)
                                potential_reward = min(reward_speed, potential_reward_max)
                                # print(potential_reward,'line607')
                        if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                            potential_reward = potential_reward_max
                            # print(potential_reward, 'line610')
                    if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                        # print(s_[0], s_[5], s_[7])
                        t2_1 = (math.ceil(((math.ceil((s_[5] - s_[0]) + 0.01) ** 2) / 18 + (s_[7] / 3)))) ** 0.5
                        # t2_1 = round((((s[7] / 3) + round(((round((s[5] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5),2)
                        # t2_1 = ((s[7] / 3) + (((s[5] - s[0]+0.01) ** 2) / 18)) ** 0.5
                        t1_1 = (s_[5] - s_[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s_[5] + s_[7]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                if s_[0] > s_[5]:  # 主车车速大于前车
                    t_brake = (s_[0] - s_[5]) / 3  # 计算紧急刹车时间
                    s_brake = (s_[0] + s_[5]) / 2 * t_brake  # 紧急刹车距离
                    ss_brake = s_[5] * t_brake + s_[7] - s_brake  # 刹车结束车距
                    if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                        if s_[0] == v_max_limit:  # 此时只能匀速再刹车
                            t_keep = ss_brake / (s_[0] - s_[5])
                            reward_speed = ((t_brake + t_keep) * s_[5] + s_[7]) / (t_brake + t_keep)
                            potential_reward = min(reward_speed, potential_reward_max)
                        if s_[0] < v_max_limit:
                            ss_last = s_[5] * (t_brake + t1) + s_[7] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                            if ss_last > 0:  # 主车可以先加速再匀速再减速
                                t_keep = ss_last / (v_max_limit - s_[5])  # 计算持续高速时间
                                reward_speed = ((t1 + t_keep + t_brake) * s_[5] + s_[7]) / (t1 + t_keep + t_brake)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss_last <= 0:
                                t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                                reward_speed = ((t_brake + t_add) * s_[5] + s_[7]) / (t_brake + t_add)
                                potential_reward = min(reward_speed, potential_reward_max)
                    if ss_brake < 0:
                        potential_reward = 0
                    if ss_brake == 0:
                        reward_speed = s_brake / t_brake
                        potential_reward = min(reward_speed, potential_reward_max)
                potential_reward_0 = potential_reward
                #########车道1的优势函数计算
                ss1 = s_[10] + t1 * s_[8] - s1  # 加速阶段结束车间距
                if s_[0] <= s_[8]:  # 主车车速小于前车
                    if ss1 >= 0:  # 加速阶段车距大于等于0
                        t3 = (v_max_limit - s_[8]) / 3  # 计算减速时间
                        if t3 > 0:  # 减速时间大于0
                            s2 = (s_[8] + v_max_limit) * t3 / 2  # 计算减速距离
                            ss2 = s_[10] + (t1 + t3) * s_[8] - s1 - s2  # 计算完整加减速过程后车距
                            if ss2 >= 0:  # 如果车距大于等于0
                                t2 = ss2 / (v_max_limit - s_[8] + 0.01)  # 计算持续高速时间
                                reward_speed = ((t1 + t2 + t3) * s_[8] + s_[10]) / (t1 + t2 + t3)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                                t2_1 = (math.ceil(
                                    ((math.ceil((s_[8] - s_[0]) + 0.01) ** 2) / 18 + (s_[10] / 3)))) ** 0.5
                                # t2_1 = round((((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                                # t2_1 = round(((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18), 2) ** 0.5
                                t1_1 = (s_[8] - s_[0]) / 3 + t2_1
                                reward_speed = ((t1_1 + t2_1) * s_[8] + s_[10]) / (t1_1 + t2_1)
                                potential_reward = min(reward_speed, potential_reward_max)
                                # print(potential_reward,'line607')
                        if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                            potential_reward = potential_reward_max
                            # print(potential_reward, 'line610')
                    if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                        t2_1 = (math.ceil(((math.ceil((s_[8] - s_[0]) + 0.01) ** 2) / 18 + (s_[10] / 3)))) ** 0.5
                        # t2_1 = round((((s[10] / 3) + round(((round((s[8] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                        # t2_1 = ((s[10] / 3) + (((s[8] - s[0]+0.01) ** 2) / 18)) ** 0.5
                        t1_1 = (s_[8] - s_[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s_[8] + s_[10]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                if s_[0] > s_[8]:  # 主车车速大于前车
                    t_brake = (s_[0] - s_[8]) / 3  # 计算紧急刹车时间
                    s_brake = (s_[0] + s_[8]) / 2 * t_brake  # 紧急刹车距离
                    ss_brake = s_[8] * t_brake + s_[10] - s_brake  # 刹车结束车距
                    if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                        if s_[0] == v_max_limit:  # 此时只能匀速再刹车
                            t_keep = ss_brake / (s_[0] - s_[8])
                            reward_speed = ((t_brake + t_keep) * s_[8] + s_[10]) / (t_brake + t_keep)
                            potential_reward = min(reward_speed, potential_reward_max)
                        if s_[0] < v_max_limit:
                            ss_last = s_[8] * (t_brake + t1) + s_[10] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                            if ss_last > 0:  # 主车可以先加速再匀速再减速
                                t_keep = ss_last / (v_max_limit - s_[8])  # 计算持续高速时间
                                reward_speed = ((t1 + t_keep + t_brake) * s_[8] + s_[10]) / (t1 + t_keep + t_brake)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss_last <= 0:
                                t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                                reward_speed = ((t_brake + t_add) * s_[8] + s_[10]) / (t_brake + t_add)
                                potential_reward = min(reward_speed, potential_reward_max)
                    if ss_brake < 0:
                        potential_reward = 0
                    if ss_brake == 0:
                        reward_speed = s_brake / t_brake
                        potential_reward = min(reward_speed, potential_reward_max)
                potential_reward_1 = potential_reward
                ##################

                ######车道2优势函数计算
                ss1 = s_[13] + t1 * s_[11] - s1  # 加速阶段结束车间距
                if s_[0] <= s_[11]:  # 主车车速小于前车
                    if ss1 >= 0:  # 加速阶段车距大于等于0
                        t3 = (v_max_limit - s_[11]) / 3  # 计算减速时间
                        if t3 > 0:  # 减速时间大于0
                            s2 = (s_[11] + v_max_limit) * t3 / 2  # 计算减速距离
                            ss2 = s_[13] + (t1 + t3) * s_[11] - s1 - s2  # 计算完整加减速过程后车距
                            if ss2 >= 0:  # 如果车距大于等于0
                                t2 = ss2 / (v_max_limit - s_[11] + 0.01)  # 计算持续高速时间
                                reward_speed = ((t1 + t2 + t3) * s_[11] + s_[13]) / (t1 + t2 + t3)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss2 < 0:  # 如果车距小于0，证明没有完整的减速过程
                                t2_1 = (math.ceil(
                                    ((math.ceil((s_[11] - s_[0]) + 0.01) ** 2) / 18 + (s_[13] / 3)))) ** 0.5
                                # t2_1 = round((((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                                # t2_1 = round(((s[13] / 3) + round(((round((s[11] - s[0]),2)+0.01) ** 2),2) / 18),2) ** 0.5
                                t1_1 = (s_[11] - s_[0]) / 3 + t2_1
                                reward_speed = ((t1_1 + t2_1) * s_[11] + s_[13]) / (t1_1 + t2_1)
                                potential_reward = min(reward_speed, potential_reward_max)
                                # print(potential_reward,'line607')
                        if t3 == 0:  # 减速时间等于0，证明前车是最高车速，此时永远追不上
                            potential_reward = potential_reward_max
                            # print(potential_reward, 'line610')
                    if ss1 < 0:  # 主车还没加速到最高速度就追上前车
                        t2_1 = (math.ceil(((math.ceil((s_[11] - s_[0]) + 0.01) ** 2) / 18 + (s_[13] / 3)))) ** 0.5
                        # t2_1 = round((((s[13] / 3) + round(((round((s[11] - s[0]), 2) + 0.01) ** 2), 2) / 18) ** 0.5), 2)
                        t1_1 = (s_[13] - s_[0]) / 3 + t2_1
                        reward_speed = ((t1_1 + t2_1) * s_[11] + s_[13]) / (t1_1 + t2_1)
                        potential_reward = min(reward_speed, potential_reward_max)
                if s_[0] > s_[11]:  # 主车车速大于前车
                    t_brake = (s_[0] - s_[13]) / 3  # 计算紧急刹车时间
                    s_brake = (s_[0] + s_[13]) / 2 * t_brake  # 紧急刹车距离
                    ss_brake = s_[11] * t_brake + s_[13] - s_brake  # 刹车结束车距
                    if ss_brake > 0:  # 如果紧急刹车车距大于0，说明主车此刻不需要紧急刹车，可以先加速再紧急刹车或匀速再刹车
                        if s_[0] == v_max_limit:  # 此时只能匀速再刹车
                            t_keep = ss_brake / (s_[0] - s_[11])
                            reward_speed = ((t_brake + t_keep) * s_[11] + s_[13]) / (t_brake + t_keep)
                            potential_reward = min(reward_speed, potential_reward_max)
                        if s_[0] < v_max_limit:
                            ss_last = s_[11] * (t_brake + t1) + s_[13] - s1 - s_brake  # 计算车间间距能否完成加速过程和刹车过程
                            if ss_last > 0:  # 主车可以先加速再匀速再减速
                                t_keep = ss_last / (v_max_limit - s_[11])  # 计算持续高速时间
                                reward_speed = ((t1 + t_keep + t_brake) * s_[11] + s_[13]) / (t1 + t_keep + t_brake)
                                potential_reward = min(reward_speed, potential_reward_max)
                            if ss_last <= 0:
                                t_add = 2 * ((ss_brake / 3 + t_brake * t_brake) ** 0.5 - t_brake)
                                reward_speed = ((t_brake + t_add) * s_[11] + s_[13]) / (t_brake + t_add)
                                potential_reward = min(reward_speed, potential_reward_max)
                    if ss_brake < 0:
                        potential_reward = 0
                    if ss_brake == 0:
                        reward_speed = s_brake / t_brake
                        potential_reward = min(reward_speed, potential_reward_max)
                potential_reward_2 = potential_reward
                # potential_reward_2_list.append(potential_reward_2)
                #########下部分代码为换道状态赋值
                DQN_s_[0] = s_[0]
                DQN_s_[1] = traci.vehicle.getAcceleration('%d' % ego_id)
                DQN_s_[2] = s_[1]
                DQN_s_[3] = s_[5]
                DQN_s_[4] = s_[6]
                DQN_s_[5] = s_[7]
                DQN_s_[6] = s_[8]
                DQN_s_[7] = s_[9]
                DQN_s_[8] = s_[10]
                DQN_s_[9] = s_[11]
                DQN_s_[10] = s_[12]
                DQN_s_[11] = s_[13]
                if lane_index == 0:
                    DQN_s_[12] = potential_reward_0 * 1.1+0.001
                    DQN_s_[13] = potential_reward_1 * s_[14]
                    DQN_s_[14] = 0

                if lane_index == 1:
                    DQN_s_[12] = potential_reward_0 * s_[15]
                    DQN_s_[13] = potential_reward_1 * 1.1+0.001
                    DQN_s_[14] = potential_reward_2 * s_[14]

                if lane_index == 2:
                    DQN_s_[12] = 0
                    DQN_s_[13] = potential_reward_1 * s_[15]
                    DQN_s_[14] = potential_reward_2 * 1.1+0.001

                    ####以下为奖励部分

                # r_efficient = math.log((s_[0] + 0.01), max((q_85, 1)))
                # r_efficient = math.log((s_[0] + 0.01), 30)
                # r = (r_potential+5*r_danger )     #可以使得安全换道
                # if i >=stable_episodes:

                cf_danger = 0
                collision = 0
                distance_headway = s_[4]
                if distance_headway > 0:
                    r_dis = 1
                    if distance_headway > 150:
                        r_dis = -3
                if distance_headway <= 0:
                    r_dis = min(10 * distance_headway, -20)
                    collision = 1
                cf_r_efficient = s_[0] / 25
                # r = 3*r_efficient + r_dis
                cf_r = 10 * cf_r_efficient + r_dis
                DDPG_s[0] = s[0]
                DDPG_s[1] = s[2]
                DDPG_s[2] = s[4]
                DDPG_s[3] = s[3]
                DDPG_s_[0] = s_[0]
                DDPG_s_[1] = s_[2]
                DDPG_s_[2] = s_[4]
                DDPG_s_[3] = s_[3]
                all_cf_r.append(cf_r)
                cf_collision_list.append(collision)
                # all_changelanetimes.append(changelane)
                # DQN_s_[15] = all_changelanetimes[-1]
                r_efficient = s_[0] / 25
                r = 5 * r_danger + 10*r_efficient + (-1)*changelane
                # if i <500:
                #     r = r_danger + 10*r_potential
                # if i >=500:
                #     r = r_danger + 10*r_efficient
                # r = 5*r_danger + 10 * r_efficient - 2
                # r = r_danger + 10 * r_efficient
                # r = r_danger + r_efficient + 1

                # RL.store_transition(DQN_s, a, r, DQN_s_)
                # if total_step > MEMORY_CAPACITY:
                #     RL.learn()
                # M.store_transition(DDPG_s, DDPG_action, cf_r, DDPG_s_)
                # if M.pointer > MEMORY_CAPACITY:
                #     var = max([var * 0.9995, VAR_MIN])  # decay the action randomness
                #     # var = var * 0.99995
                #     b_M = M.sample(BATCH_SIZE)
                #     b_s = b_M[:, :STATE_DIM]
                #     b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                #     b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                #     b_s_ = b_M[:, -STATE_DIM:]
                #
                #     critic.learn(b_s, b_a, b_r, b_s_)
                #     actor.learn(b_s)

                s = s_
                DQN_s = DQN_s_

                total_step += 1
                ep_step += 1
                all_r.append(r)

                rear_v_list.append(s_[0])
                l_v_list.append(s_[2])
                all_changelanetimes.append(changelane)
                danger_lc.append(dangerious_lc)
                lane_index0_acc_list.append(a_lane0)
                lane_index1_acc_list.append(a_lane1)
                lane_index2_acc_list.append(a_lane2)
                # ep_step_list.append(j)
                distance_headway_list.append(s_[4])
                horizontal_position_list.append(traci.vehicle.getPosition('%d' % ego_id)[1])
                lengthwise_position_list.append(traci.vehicle.getPosition('%d' % ego_id)[0])
                order_ego_id_list.append(order_ego_id)
                # car_15_speed.append(traci.vehicle.getSpeed('15'))
            mean_speed = sum(rear_v_list)/MAX_EP_STEPS
            # car_15_mean_speed = sum(car_15_speed)/MAX_EP_STEPS
            # print(car_15_mean_speed)
            mean_speed_list.append(mean_speed)
            total_ep_list.append(i)
            total_lc_r_list.append(sum(all_r))



            # print("car_following:", 'reward=%s' % sum(all_cf_r),'collision-times=%s' % sum(cf_collision_list),'explore=%s'% var,'mean_speed=%s' % mean_speed)

            # print('sceen=%s' % sceen_id, 'ego_id=%s' % ego_id, 'episode=%s' % i, 'reward=%s' % sum(all_r),
            #       'lanechange-times=%s' % sum(all_changelanetimes),
            #       'cf-dangerious-times=%s' % sum(cf_collision_list), 'mean_speed=%s' % mean_speed)
            #
            # if sum(cf_collision_list) == 0:
            #
            #     test1 = pd.DataFrame({'ego_speed': rear_v_list, 'horizontal position_list': horizontal_position_list,
            #                           'lengthwise_position_list': lengthwise_position_list,
            #                           'order_ego_id_list': order_ego_id_list})
            #     # test1.to_csv('./figure_result.csv')
            #     test1.to_csv('data\\data_sceen_new\\' +'%d' % sceen_id +'\\'+'%d' % sceen_id+'-ego_id' +'-'+'%d' % ego_id + '-' + '%d' % i + '.csv', mode='a', header=True)  # 第一行用作列名称
            if sum(cf_collision_list) != 0:
                print('sceen=%s' % sceen_id, 'ego_id=%s' % ego_id,'cf-dangerious-times=%s' % sum(cf_collision_list),'Dangerious!!!Not save!!!')
                start_IDM_follow = 1
            # if sum(cf_collision_list) == 0 and mean_speed >= ego_mean_speed:
            if sum(cf_collision_list) == 0:
                print('sceen=%s' % sceen_id, 'ego_id=%s' % ego_id,  'ego_mean_speed=%s' % ego_mean_speed,'episode=%s' % i, 'reward=%s' % sum(all_r),
                      'lanechange-times=%s' % sum(all_changelanetimes),
                      'cf-dangerious-times=%s' % sum(cf_collision_list), 'mean_speed=%s' % mean_speed)