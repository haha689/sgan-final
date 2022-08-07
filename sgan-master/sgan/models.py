import torch
import torch.nn as nn


#MLP: multi layer perceptron, has a series of relu layers and activation layers.
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

#Uses built in pytorch LSTM modules as the encoder
class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim) #NEED FIX?

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )
    #forward pass of LSTM training 

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory with sptial embeddings of peoples location
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.reshape(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        #the encoder is the LSTM that takes those arguments
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos): #FIX 1
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: List of shape (len(seq_start_end), )
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            #start = start.item()
            #end = end.item()
            num_ped = end - start
            #(num_ped, self.h_dim)
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            #(num_ped, 2)
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            #(num_ped ** 2, self.h_dim)
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            #(num_ped ** 2, 2)
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            #(num_ped ** 2, 2)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            #(num_ped ** 2, 2)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            #(num_ped ** 2, embedding_dim)
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            
            #(num_ped ** 2, self.h_dim + embedding_dim)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            #(num_ped ** 2, bottleneck_dim)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            
            #(num_ped, num_ped, bottleneck_dim) -> (num_ped, 1, bottleneck_dim)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            #(num_ped ** 2, bottleneck_dim)
            curr_pool_h = curr_pool_h.repeat(num_ped,1,1)
           
            #(num_ped, num_ped, bottleneck_dim)
            curr_final_h = curr_pool_h.view(num_ped, num_ped, -1)
            #batch x n x n x hidden dimension
            #convolution matters on size of hidden dimensions
            pool_h.append(curr_pool_h + curr_final_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.conv_hidden_dim = 1024
        self.conv_output_dim = 5
        self.conv1 = torch.nn.Conv2d(self.bottleneck_dim,self.conv_hidden_dim,1)
        self.conv1_bn = torch.nn.BatchNorm2d(self.conv_hidden_dim)
        self.conv2 = torch.nn.Conv2d(self.conv_hidden_dim,self.conv_hidden_dim,1)
        self.conv2_bn = torch.nn.BatchNorm2d(self.conv_hidden_dim)
        self.conv3 = torch.nn.Conv2d(self.conv_hidden_dim,self.conv_output_dim,1)
        self.conv3_bn = torch.nn.BatchNorm2d(self.conv_output_dim)
        self.conv4 = torch.nn.Conv2d(self.conv_hidden_dim,2,1)
        self.conv4_bn = torch.nn.BatchNorm2d(2)
        self.relu_activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
        self.pool_net = PoolHiddenNet(
            embedding_dim=self.embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm
        )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

       
        input_dim = encoder_h_dim + bottleneck_dim
        
        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        
        return False
    
    def conv_pool(self, conv_input, num_ped):

        max_output = conv_input.max(1)[0]
        max_output = max_output.repeat(num_ped,1,1)

        return max_output + conv_input

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None): #MODIFY
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - outputs: List of shape (len(seq_start_end), )
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel) 
        # Pool States
       
        end_pos = obs_traj[-1, :, :]
        pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
        #CONVOLUTION ADDED
        outputs = []
        times = []
        
        for i,tensor_h in enumerate(pool_h):
            (start,end) = seq_start_end[i]
            num_ped = end - start
           # tensor_h = torch.unsqueeze(tensor_h,0)
            tensor_conv1 = torch.permute(tensor_h, (2, 0, 1))
            tensor_conv1 = torch.unsqueeze(tensor_conv1, 0)
            tensor_conv1 = self.conv1(tensor_conv1)
            tensor_conv1 = self.conv1_bn(tensor_conv1)
            tensor_conv1 = torch.squeeze(tensor_conv1, 0)
            tensor_conv1 = self.relu_activation(tensor_conv1)
            #tensor_conv1 = self.dropout(tensor_conv1)
            tensor_conv1 = torch.permute(tensor_conv1, (1, 2, 0))
            tensor_conv1 = self.conv_pool(tensor_conv1,num_ped)

            tensor_conv2 = torch.permute(tensor_conv1, (2, 0, 1))
            tensor_conv2 = torch.unsqueeze(tensor_conv2, 0)
            tensor_conv2 = self.conv2(tensor_conv2)
            tensor_conv2 = self.conv2_bn(tensor_conv2)
            tensor_conv2 = torch.squeeze(tensor_conv2, 0)
            tensor_conv2 = self.relu_activation(tensor_conv2)
            #tensor_conv2 = self.dropout(tensor_conv2)
            tensor_conv2 = torch.permute(tensor_conv2, (1, 2, 0))
            tensor_conv2 = self.conv_pool(tensor_conv2,num_ped)

            tensor_conv3 = torch.permute(tensor_conv2, (2, 0, 1))
            tensor_conv3 = torch.unsqueeze(tensor_conv3, 0)
            tensor_conv3 = self.conv3(tensor_conv3)
            tensor_conv3 = self.conv3_bn(tensor_conv3)
            tensor_conv3 = torch.squeeze(tensor_conv3, 0)
            #tensor_conv3 = self.dropout(tensor_conv3)
            tensor_conv3 = torch.permute(tensor_conv3, (1, 2, 0))

            tensor_conv4 = torch.permute(tensor_conv2, (2, 0, 1))
            tensor_conv4 = torch.unsqueeze(tensor_conv4, 0)
            tensor_conv4 = self.conv4(tensor_conv4)
            tensor_conv4 = self.conv4_bn(tensor_conv4)
            tensor_conv4 = torch.squeeze(tensor_conv4, 0)
            #tensor_conv4 = self.dropout(tensor_conv4)
            tensor_conv4 = torch.permute(tensor_conv4, (1, 2, 0))

            outputs.append(tensor_conv3)
            times.append(tensor_conv4)
            #print(tensor_conv3.size())
            #print(tensor_conv4.size())

        return outputs, times