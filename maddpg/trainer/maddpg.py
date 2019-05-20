import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

np.random.seed(0)
def get_subpolicy_index(K): #返回0到K-1的一个随机整数
    return  np.random.randint(0,K,1)[0]

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, index, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, ensemble_num=5):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        #不懂
        # set up placeholders
        obs_ph_n = make_obs_ph_n #输入：观测
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(index)+str(i)) for i in range(len(act_space_n))]
        #输出：行动
        p_input = obs_ph_n[p_index] #这个智能体得到的观测

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func"+str(index), num_units=num_units)
        #得到映射函数，是个全连接网络
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"+str(index)))
        #得到这个网络的所有参数
        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        #不懂
        act_sample = act_pd.sample() #采样，得到一个动作输出（一个实数）

        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))#将参数展为一维，计算模方

        act_input_n = act_ph_n + [] #动作输入，是placeholder
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func: #如果是局部Q函数
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        #不懂，一个全连接网络
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        #训练网络
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        #输出的动作
        p_values = U.function([obs_ph_n[p_index]], p)
        #不懂
        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func"+str(index), num_units=num_units)
        #现实网络的函数。一个全连接网络
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"+str(index)))
        #得到全连接网络的参数
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        #更新现实网络的参数，以soft的形式，即每次更新一点点（动量更新）
        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        #不懂
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)
        #得到现实网络的动作
        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}
        #act： 更新网络得到的动作  train：训练更新网络  update_target_p：用更新网络更新现实网络
        #p_values:不懂 #target_act:现实网络得到的动作

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        #q_func是一个函数 其输出为全连接网络的输出，即q
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func")) #得到函数中的参数（全连接的参数）

        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        #定义平方损失，这是critic中的DQN的损失函数
        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg #类似参数衰减，防止过拟合

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        #将输入到输出打包为一个函数
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        #目标Q网络，用于计算Q现实，不必训练参数，每隔一段时间从q网络复制参数
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        #得到目标Q网络的参数
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        #将这个网络打包为一个函数，调用这个函数就可以方便地计算Q现实
        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}
        #train为更新网络从输入映到输出的函数，update_target_q将更新网络的参数全部复制到现实网络（动量更新，每次只更新很小的一部分）
        #q_values,target_q_values分别计算更新网络和现实网络的输出，即DQN的输出
class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args,ensemble_num = 3, num_adversaries = 1, local_q_func=False):
        self.name = name #名字
        self.n = len(obs_shape_n) #观测的个数
        self.agent_index = agent_index #智能体的编号
        self.K = ensemble_num
        self.num_adversaries = num_adversaries
        self.current_subpolicy_index = get_subpolicy_index(self.K)
        self.args = args #其余参数
        obs_ph_n = [] #placeholder 用于接收环境给予的观测矩阵，一共有self.n个观测，每一个是一个矩阵
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())#加入placeholder

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
        #q_train用于训练，q_update用于更新Q现实网络,q_debug用于调试
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,  #mlp_model 是个函数
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )


        #所有p参数都是一个长度为K的list，包括K个子策略的内容
        self.act = []
        self.p_train = []
        self.p_update = []
        self.p_debug = []
        for i in range(self.K):
            tem_act, tem_train, tem_update, tem_debug = p_train(
                scope=self.name,
                make_obs_ph_n=obs_ph_n,
                act_space_n=act_space_n,
                p_index=agent_index,
                p_func=model,
                q_func=model,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
                index = i,
                grad_norm_clipping=0.5,
                local_q_func=local_q_func,
                num_units=args.num_units,
                ensemble_num = self.K
            )
            self.act.append(tem_act)
            self.p_train.append(tem_train)
            self.p_update.append(tem_update)
            self.p_debug.append(tem_debug)
        # Create experience buffer
        self.replay_buffer = [ReplayBuffer(1e6) for i in range(self.K)] #定义K个记忆池
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len 
        self.replay_sample_index = None #抽取记记池中记忆的下标

    def change_subpolicy(self): #改变使用的子策略
        self.current_subpolicy_index = get_subpolicy_index(self.K)

    def action(self, obs):#返回动作
        return self.act[self.current_subpolicy_index](obs[None])[0] 

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # 将这次经验存到记忆池中
        # Store transition in the replay buffer.
        self.replay_buffer[self.current_subpolicy_index].add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        #不懂
        self.replay_sample_index = None

    def update(self, agents, t,arglist):
        #将更新网络的参数更新到现实网络中，同时…不懂
        if len(self.replay_buffer[self.current_subpolicy_index]) < self.max_replay_buffer_len: # replay buffer is not large enough
            self.change_subpolicy()
            return
        if not t % 100 == 0:  # only update every 100 steps
            self.change_subpolicy()
            return

        self.replay_sample_index = self.replay_buffer[self.current_subpolicy_index].make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index

        for i in range(self.n):
            #if i<self.num_adversaries:  #get ensemble_num first
            #    tem_k = arglist.adv_num;
            #else:
            #    tem_k = arglist.good_num;
            #for j in range(tem_k): #收集所有子策略的经验
            
            j = agents[i].current_subpolicy_index
            obs, act, rew, obs_next, done = agents[i].replay_buffer[j].sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer[self.current_subpolicy_index].sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):

            target_act_next_n = [agents[j].p_debug[agents[j].current_subpolicy_index]['target_act'](obs_next_n[j]) for j in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train[self.current_subpolicy_index](*(obs_n + act_n))

        self.p_update[self.current_subpolicy_index]()  #更新actor参数
        self.q_update()  #更新critic参数
        tem_policy_index = self.current_subpolicy_index
        self.change_subpolicy()
        
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
