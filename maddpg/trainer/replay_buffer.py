import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = [] #记忆池
        self._maxsize = int(size) #最大的容量
        self._next_idx = 0  #将要装入的位置

    def __len__(self): #返回目前的记记数量
        return len(self._storage)

    def clear(self): #清空记忆池
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):  #add an execution to the replay buffer
        #向记忆池中添加记忆
        data = (obs_t, action, reward, obs_tp1, done) #将一个记忆打包为元组

        if self._next_idx >= len(self._storage):#如果记忆池还没有满，则向其中直接添加一条记忆
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data    #_next_idx acts as a point indicating the current position
            #否则覆盖之前的记忆。记忆池是一个有限长度的队列，后边的把前边的推出去
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):  #get a collection of execution
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            t = min(i,len(self._storage)-1)
            if(t<0):
                print(i,len(self._storage)-1)
            data = self._storage[t]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):  #generate a collection of random index
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size): #generate a collection of latest index
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))   #get index first, then get samples
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1) #get all the executions
