from typing import Dict, List

import torch as th
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelConfigDict, ModelWeights
from ray.rllib.models.modelv2 import restore_original_dimensions
from torch import nn


class CustomFeedForwardModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=64,
                 fc_size=32,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape # observation space is flattened
        #self.pre_fc_size = (7 - 2) * (11 - 2) * conv_filters
        self.pre_fc_size = 1600
        self.fc_size = fc_size

        #layer_1 = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
        #layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
        #layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
        #layer_3 = conv_to_fc(layer_3)
        #return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

        # TODO: use more convolutions here? Change and check that we can still overfit on binary problem.
        #self.conv_1 = nn.Conv2d(1, out_channels=conv_filters, kernel_size=3, stride=1, padding=0)
        stride = 2
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=stride, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=stride, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=stride, padding=0)
        

        #self.fc_1 = SlimFC(1024, 512)
        self.fc_1 = SlimFC(256, self.fc_size)
        #self.fc_1 = SlimFC(self.pre_fc_size, self.fc_size)
        self.action_branch = SlimFC(self.fc_size, num_outputs)
        self.value_branch = SlimFC(self.fc_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        #raise ValueError(f'dims: {input_dict["obs"].shape}')
        input_dict['obs'] = input_dict['obs'].reshape(
                input_dict['obs'].size(0), 
                28, 
                28, 
                1
            )
        input = input_dict["obs"].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        x = nn.functional.relu(self.conv_1(input.float()))
        x = nn.functional.relu(self.conv_2(x))
        x = nn.functional.relu(self.conv_3(x))
        x = x.reshape(x.size(0), -1)
        #raise ValueError(f'{x.shape}, {self.fc_size}, {self.pre_fc_size}')
        x = nn.functional.relu(self.fc_1(x))
        self._features = x
        action_out = self.action_branch(self._features)

        return action_out, []

class CustomFeedForwardModel3D(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                #  conv_filters=64,
                fc_size=2048,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape

        # Determine size of activation after convolutional layers so that we can initialize the fully-connected layer
        # with the correct number of weights.
        # TODO: figure this out properly, independent of map size. Here we just assume width/height/length of
        # (padded) observation is 14
        #self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * 32
        #self.pre_fc_size = 128 * 2 * 2 * 2
        #self.pre_fc_size = 7*7*8*8
        self.pre_fc_size = 128

        # Convolutinal layers.
        self.conv_1 = nn.Conv3d(1, out_channels=64, kernel_size=3, stride=2, padding=1)  # 7 * 7 * 7
        self.conv_2 = nn.Conv3d(64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 4 * 4 * 4
        self.conv_3 = nn.Conv3d(128, out_channels=128, kernel_size=3, stride=2, padding=1)  # 2 * 2 * 2

        # Fully connected layer.
        #self.fc_1 = SlimFC(self.pre_fc_size, fc_size)
        self.fc_1 = SlimFC(2048, 128)

        # Fully connected action and value heads.
        #self.action_branch = SlimFC(fc_size, num_outputs)
        self.action_branch = SlimFC(128, num_outputs)
        #self.value_branch = SlimFC(fc_size, 1)
        self.value_branch = SlimFC(128, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        #raise ValueError(type(input_dict['obs']), len(input_dict['obs']))
        #raise ValueError(type(input_dict['obs']), input_dict['obs'][0].shape, len(input_dict['obs']))
        #raise ValueError(input_dict['obs'].keys())
        key = list(input_dict['obs'].keys())[0]
        input = input_dict["obs"][key].permute(0, 4, 1, 2, 3)  # Because rllib order tensors the tensorflow way (channel last)
        x = nn.functional.relu(self.conv_1(input.float()))
        x = nn.functional.relu(self.conv_2(x.float()))
        x = nn.functional.relu(self.conv_3(x.float()))
        #x = x.reshape(x.size(0), -1) # had to transpose
        x = x.view(x.size(0), -1) # had to transpose
        x = nn.functional.relu(self.fc_1(x))
        self._features = x
        action_out = self.action_branch(self._features)

        return action_out, []



