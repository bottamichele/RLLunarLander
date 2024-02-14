import csv
import numpy as np
import torch as tc
from collections import deque
from torch.nn import Module, Linear, MSELoss
from torch.nn.functional import relu
from torch.optim import Adam

class DQN(Module):
    """A Deep Q-Networks MPL"""
    
    #def __init__(self, input_dim: int, output_dim: int, fcs_dim, lr: float):
    def __init__(self, input_dim: int, output_dim: int, fc1_dim: int, fc2_dim: int, lr: float):
        """
        Create DQN.

        Parameters
        --------------------
        input_dim: int
            input layer dimensions

        output_dim: int
            output layer dimensions

        fc1_dim: int
            first fully connected dimensions

        fc2_dim: int
            second fully connected dimensions

        lr: float
            learning rate
        """
        
        super(DQN, self).__init__()

        self.fc1 = Linear(input_dim, fc1_dim)
        self.fc2 = Linear(fc1_dim, fc2_dim)
        self.fc3 = Linear(fc2_dim, output_dim)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss = MSELoss()
        self.device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        x is evalueted by neural network.

        Parameter
        --------------------
        x: Tensor
            a tensor

        Return
        --------------------
        y: Tensor
            result of x evalueted
        """
    
        v = relu(self.fc1(x))
        v = relu(self.fc2(v))
        return self.fc3(v)




class AgentDQN:
    """An agent that implements Deep Q-Networks (MLP)."""

    def __init__(self, obs_dim, n_actions, mem_size, batch_size, lr=0.001, gamma=0.99, eps_init=1.0, eps_min=0.1, eps_dec=10**-4, is_trained=True):
        """
        Create agent.
        
        Parameters
        --------------------
        obs_dim: int
            dimensions of observation

        n_actions: int
            number of actions

        mem_size: int
            memory capacity

        batch_size: int
            batch size

        lr: float, optional
            learning rate

        gamma: float, optional
            discount factor

        eps_init: float, optional
            initial epsilon value

        eps_min: float, optional
            minimum value epsilon

        eps_dec: float, optional
            epsilon value of decay

        is_trained: bool, optional
            agent is trained or not
        """

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.memory = deque(maxlen=mem_size)
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon_init = eps_init
        self.epsilon = eps_init
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_dec
        self.is_trained = is_trained
        self.model = DQN(self.obs_dim, self.n_actions, 256, 256, self.lr)
        self.rng = np.random.default_rng()
        
    def choose_action(self, obs):
        """
        Choose a action from a observation.

        Parameter
        --------------------
        obs: ndarray
            an observation

        Return
        --------------------
        action: int
            an action choosen
        """

        #Agent chooses a random action.
        if self.rng.uniform() <= self.epsilon and self.is_trained:
            return self.rng.integers(0, self.n_actions)
        
        #Agent chooses best action from obs.
        x = tc.Tensor(np.array([obs])).to(self.model.device)
        q_values = self.model.forward(x)
        return tc.argmax(q_values).item()
    
    def store(self, obs, action, reward, next_obs, is_next_obs_terminal):
        """
        Store into memory.

        Parameters:
        obs: ndarray
            current observation

        action: int
            action choosen for obs

        reward: float
            reward obtained performing current action

        next_obs: ndarray
            observation obtained by performing action

        is_next_obs_terminal: bool
            nex_obs is a terminal state or not
        """

        self.memory.append((obs, action, reward, next_obs, is_next_obs_terminal))

    def train(self):
        """Perform a training step."""

        #It is not performed a training step if memory buffer doesn't have at least batch_size elements.
        if len(self.memory) < self.batch_size:
            return

        #Build a batch for DQN from memory buffer.
        obs_batch = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
        action_batch = np.zeros(self.batch_size, dtype=np.int8)
        reward_batch = np.zeros(self.batch_size, dtype=np.float32)
        next_obs_batch = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
        next_obs_terminal_batch = np.zeros(self.batch_size, dtype=bool)
        indices_batch = self.rng.permutation(np.arange(0, len(self.memory), 1, dtype=np.int32))[:self.batch_size]

        for i in range(self.batch_size):
            obs_batch[i] = self.memory[indices_batch[i]][0]
            action_batch[i] = self.memory[indices_batch[i]][1]
            reward_batch[i] = self.memory[indices_batch[i]][2]
            next_obs_batch[i] = self.memory[indices_batch[i]][3]
            next_obs_terminal_batch[i] = self.memory[indices_batch[i]][4]

        #Convert to tensors.
        obs_batch = tc.Tensor(obs_batch).to(self.model.device)
        #action_batch = tc.Tensor(action_batch).to(self.model.device)
        reward_batch = tc.Tensor(reward_batch).to(self.model.device)
        next_obs_batch = tc.Tensor(next_obs_batch).to(self.model.device)
        #next_obs_terminal_batch = tc.Tensor(next_obs_terminal_batch).to(self.model.device)
        
        self.model.optimizer.zero_grad()
        idxs = np.arange(0, self.batch_size, 1, dtype=np.int32)
        
        q_values = self.model.forward(obs_batch)[idxs, action_batch]
        q_values_next = self.model.forward(next_obs_batch)
        q_values_next[next_obs_terminal_batch] = 0.0

        q_evals = reward_batch + self.gamma * tc.max(q_values_next, dim=1)[0]

        #Do backpropagation
        loss = self.model.loss(q_evals, q_values).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        #Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min


    def save_model(self, file_name):
        """
        Save current model to disk.

        Parameter
        --------------------
        file_name: string
            file name where to save current model.
        """

        tc.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        """
        Load a model from disk.

        Parameter
        --------------------
        file_name: string
            file name where to load a model.
        """

        self.model.load_state_dict(tc.load(file_name))

    def save_report_hyperparameters(self, file_name):
        """
        Save hyperparameters used into a csv file.
        
        Parameter
        --------------------
        file_name: str
            file name to save.
        """
        
        csv_hprm_file = open(file_name, 'w')
        csv_hprm_writer = csv.writer(csv_hprm_file, delimiter=';')

        csv_hprm_writer.writerow(['learning rate', self.lr])
        csv_hprm_writer.writerow(['discount factor', self.gamma])
        csv_hprm_writer.writerow(['memory size', self.memory.maxlen])
        csv_hprm_writer.writerow(['batch size', self.batch_size])
        csv_hprm_writer.writerow(['epsilon initial', self.epsilon_init])
        csv_hprm_writer.writerow(['epsilon end', self.epsilon_min])
        csv_hprm_writer.writerow(['epsilon decay', self.epsilon_dec])

        csv_hprm_file.close()