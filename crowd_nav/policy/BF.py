
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


class UpsdBehavior(nn.Module):
    '''
    UDRL behaviour function that produces actions based on a state and command.

    Params:
        state_size (int)
        action_size (int)
        hidden_size (list of ints)
        desires_scalings (List of float)
    '''

    def __init__(self, state_size, desires_size,
                 action_size, hidden_sizes,
                 desires_scalings):
        super().__init__()
        self.desires_scalings = torch.FloatTensor(desires_scalings)

        l = nn.Linear(state_size, hidden_sizes[0])
        torch.nn.init.orthogonal_(l.weight, gain=1)
        self.state_fc = nn.Sequential(l, nn.Tanh())

        l = nn.Linear(desires_size, hidden_sizes[0])
        torch.nn.init.orthogonal_(l.weight, gain=1)
        self.command_fc = nn.Sequential(l, nn.Sigmoid())

        layers = nn.ModuleList()
        activation = nn.ReLU
        output_activation = nn.Identity
        for j in range(len(hidden_sizes) - 1):
            l = nn.Linear(hidden_sizes[j], hidden_sizes[j + 1])
            torch.nn.init.orthogonal_(l.weight, gain=np.sqrt(2))
            layers.append(l)
            layers.append(activation())

        # output layer:
        # uses default Pytorch init.
        layers.append(nn.Linear(hidden_sizes[-1], action_size))
        layers.append(output_activation())
        self.output_fc = nn.Sequential(*layers)

    def forward(self, state, command):
        '''Forward pass

        Params:
            state (List of float)
            command (List of float)

        Returns:
            FloatTensor -- action logits
        '''
        if len(command) == 1:
            command = command[0]
        else:
            command = torch.cat(command, dim=1)
        # print('entering the model', state.shape, command.shape)
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.desires_scalings)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)


class BF(Policy):

    def __init__(self):
        super().__init__()
        self.name = 'BF'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def configure(self, config):
        self.set_common_parameters(config)
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.multiagent_training = config.getboolean('cadrl', 'multiagent_training')
        logging.info('Policy: CADRL without occupancy map')

