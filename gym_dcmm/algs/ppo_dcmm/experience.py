import torch
from torch.utils.data import Dataset
import configs.env.DcmmCfg as DcmmCfg


def transform_op(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    if isinstance(arr, dict):
        return {k: transform_op(v) for k, v in arr.items()}
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(Dataset):
    def __init__(
        self, num_envs, horizon_length, batch_size, minibatch_size, obs_dim, act_dim, device,
        use_gru=None, gru_hidden_size=None, gru_num_layers=None):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # GRU configuration
        self.use_gru = use_gru if use_gru is not None else DcmmCfg.gru_config.enabled
        self.gru_hidden_size = gru_hidden_size if gru_hidden_size is not None else DcmmCfg.gru_config.hidden_size
        self.gru_num_layers = gru_num_layers if gru_num_layers is not None else DcmmCfg.gru_config.num_layers
        
        self.storage_dict = {
            'rewards': torch.zeros(
                (self.transitions_per_env, self.num_envs, 1),
                dtype=torch.float32, device=self.device),
            'values': torch.zeros(
                (self.transitions_per_env, self.num_envs,  1),
                dtype=torch.float32, device=self.device),
            'neglogpacs': torch.zeros(
                (self.transitions_per_env, self.num_envs),
                dtype=torch.float32, device=self.device),
            'dones': torch.zeros(
                (self.transitions_per_env, self.num_envs),
                dtype=torch.uint8, device=self.device),
            'actions': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            'mus': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            'sigmas': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            'returns': torch.zeros(
                (self.transitions_per_env, self.num_envs,  1),
                dtype=torch.float32, device=self.device),
        }
        
        # Add hidden state storage if GRU is enabled
        if self.use_gru:
            # Store hidden state at the START of each timestep
            # Shape: (transitions, num_envs, num_layers, hidden_size)
            self.storage_dict['hidden_states'] = torch.zeros(
                (self.transitions_per_env, self.num_envs, self.gru_num_layers, self.gru_hidden_size),
                dtype=torch.float32, device=self.device)

        if isinstance(self.obs_dim, dict):
            self.storage_dict['obses'] = {}
            for k, v in self.obs_dim.items():
                # v is tuple shape e.g. (15,) or (1, 112, 112)
                # Use uint8 for depth to save 75% memory (1 byte vs 4 bytes per pixel)
                if k == 'depth':
                    self.storage_dict['obses'][k] = torch.zeros(
                        (self.transitions_per_env, self.num_envs, *v),
                        dtype=torch.uint8, device=self.device)
                else:
                    self.storage_dict['obses'][k] = torch.zeros(
                        (self.transitions_per_env, self.num_envs, *v),
                        dtype=torch.float32, device=self.device)
        else:
            self.storage_dict['obses'] = torch.zeros(
                (self.transitions_per_env, self.num_envs, self.obs_dim),
                dtype=torch.float32, device=self.device)

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]
        
        # Return hidden_states if GRU is enabled
        if self.use_gru and 'hidden_states' in input_dict:
            return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], \
                input_dict['mus'], input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
                input_dict['obses'], input_dict['hidden_states']
        else:
            return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], \
                input_dict['mus'], input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
                input_dict['obses']

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mus'][start:end] = mu
        self.data_dict['sigmas'][start:end] = sigma

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index,:] = v
        else:
            self.storage_dict[name][index,:] = val
    
    def update_hidden_states(self, index, hidden_states):
        """
        Store hidden states at a given timestep.
        
        Args:
            index: Timestep index
            hidden_states: Tensor of shape (num_layers, num_envs, hidden_size)
        """
        if self.use_gru and hidden_states is not None:
            # Validate input shape
            expected_shape = (self.gru_num_layers, self.num_envs, self.gru_hidden_size)
            if hidden_states.shape != expected_shape:
                raise ValueError(
                    f"Hidden states shape mismatch: expected {expected_shape}, "
                    f"got {tuple(hidden_states.shape)}"
                )
            # hidden_states comes as (num_layers, num_envs, hidden_size)
            # We need to store as (num_envs, num_layers, hidden_size)
            self.storage_dict['hidden_states'][index] = hidden_states.transpose(0, 1)

    def compute_return(self, last_values, gamma, tau):
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage_dict['rewards'])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict['values'][t + 1]
            next_nonterminal = 1.0 - self.storage_dict['dones'].float()[t]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            delta = self.storage_dict['rewards'][t] + \
                gamma * next_values * next_nonterminal - self.storage_dict['values'][t]
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam
            self.storage_dict['returns'][t, :] = mb_advs[t] + self.storage_dict['values'][t]

    def prepare_training(self):
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            transformed = transform_op(v)
            # Convert uint8 depth back to float32 for neural network
            if isinstance(transformed, dict) and 'depth' in transformed:
                transformed['depth'] = transformed['depth'].float() / 255.0
            self.data_dict[k] = transformed
        advantages = self.data_dict['returns'] - self.data_dict['values']
        self.data_dict['advantages'] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)).squeeze(1)
        return self.data_dict
