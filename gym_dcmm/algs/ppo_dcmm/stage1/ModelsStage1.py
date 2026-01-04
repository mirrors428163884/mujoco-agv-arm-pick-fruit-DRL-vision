import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import configs.env.DcmmCfg as DcmmCfg


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class CNNBase(nn.Module):
    """
    CNN feature extractor for depth images.
    
    [UPDATED 2025-01-04] Added AdaptiveAvgPool2d to handle variable image dimensions.
    This ensures that changing img_dim (84/128/224) won't cause dimension explosions.
    """
    def __init__(self, input_shape, hidden_size=256):
        super(CNNBase, self).__init__()
        # Input shape is (C, H, W)
        
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # [NEW 2025-01-04] AdaptiveAvgPool ensures fixed output size regardless of input resolution
        # Output will always be (64, 4, 4) = 1024 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fixed CNN output size: 64 channels * 4 * 4 = 1024
        self.cnn_output_size = 64 * 4 * 4
        
        self.linear = nn.Linear(self.cnn_output_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # Ensure fixed size
        x = x.flatten(1)  # Flatten to (batch, features)
        x = self.linear(x)
        x = self.relu(x)
        return x

class ActorCritic(nn.Module):
    """
    Actor-Critic network for Stage 1 (tracking/approach).
    
    [UPDATED 2025-01-04] Architecture improvements:
    1. Separate output heads for base (2D: vx, vyaw) and arm (6D: joint deltas)
    2. Distance-based gating: arm_action *= sigmoid(a*(1.0 - d_base))
       - Far: rely more on base
       - Near: rely more on arm
    3. Actor also gets visual embedding (small) for obstacle awareness
    """
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        
        self.units = kwargs.pop('actor_units')
        
        self.use_vision = kwargs.get('use_vision', True)
        
        # [NEW 2025-01-04] Separate heads configuration
        self.use_separate_heads = kwargs.get('use_separate_heads', True)
        self.base_action_dim = 2  # vx, vyaw
        self.arm_action_dim = 6   # joint deltas
        
        # GRU configuration
        self.use_gru = kwargs.get('use_gru', DcmmCfg.gru_config.enabled)
        self.gru_hidden_size = kwargs.get('gru_hidden_size', DcmmCfg.gru_config.hidden_size)
        self.gru_num_layers = kwargs.get('gru_num_layers', DcmmCfg.gru_config.num_layers)
        
        # Vector input size
        if isinstance(input_shape, dict):
             mlp_input_shape = input_shape['vector'][0]
             img_input_shape = input_shape['image']
        else:
             # Fallback for legacy/tuple input
             mlp_input_shape = input_shape[0]
             img_input_shape = None
             self.use_vision = False

        # CNN Feature Extractor
        if self.use_vision:
            self.cnn = CNNBase(img_input_shape, hidden_size=256)
            combined_input_size = mlp_input_shape + 256
            # Actor currently does not consume visual features; keep its input
            # size equal to the vector MLP input size to avoid dead code and
            # inconsistent dimension accounting.
            actor_input_size = mlp_input_shape
        else:
            self.cnn = None
            combined_input_size = mlp_input_shape
            actor_input_size = mlp_input_shape
        
        # GRU Layer (before MLP)
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=combined_input_size,
                hidden_size=self.gru_hidden_size,
                num_layers=self.gru_num_layers,
                batch_first=True,
                bidirectional=False
            )
            # Orthogonal initialization for GRU weights (only for 2D weight matrices)
            for name, param in self.gru.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            
            mlp_input_size = self.gru_hidden_size
        else:
            self.gru = None
            mlp_input_size = combined_input_size
        
        out_size = self.units[-1]

        # Actor MLP (shared features)
        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_size)
        
        # [NEW 2025-01-04] Separate output heads for base and arm
        if self.use_separate_heads:
            self.base_head = nn.Linear(out_size, self.base_action_dim)
            self.arm_head = nn.Linear(out_size, self.arm_action_dim)
            # Initialize with small weights
            nn.init.orthogonal_(self.base_head.weight, gain=0.01)
            nn.init.orthogonal_(self.arm_head.weight, gain=0.01)
            # Separate log_std for base and arm
            self.sigma_base = nn.Parameter(
                torch.full((self.base_action_dim,), -1.0, dtype=torch.float32), requires_grad=True)
            self.sigma_arm = nn.Parameter(
                torch.full((self.arm_action_dim,), -1.0, dtype=torch.float32), requires_grad=True)
        else:
            # Legacy single output
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.sigma = nn.Parameter(
                torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            nn.init.constant_(self.sigma, -1.0)
            torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)
        
        # Value network
        if self.separate_value_mlp:
            self.value_mlp = MLP(units=self.units, input_size=mlp_input_size)
        self.value = torch.nn.Linear(out_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
    
    def get_initial_hidden(self, batch_size, device):
        """
        Get initial hidden state for GRU.
        
        Args:
            batch_size: Number of parallel environments
            device: Torch device
            
        Returns:
                    Hidden state tensor for the shared GRU of shape
            (num_layers, batch_size, hidden_size), or None if GRU is disabled.
        """
        if not self.use_gru:
            return None
        
        # Shape: (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(
            self.gru_num_layers, batch_size, self.gru_hidden_size,
            device=device, dtype=torch.float32
        )
        return hidden
    
    def save_actor(self, actor_mlp_path, actor_head_path):
        """
        Save actor and critic model parameters to files.

        actor_path: Path to save actor model parameters.
        """
        torch.save(self.actor_mlp.state_dict(), actor_mlp_path)
        if self.use_separate_heads:
            # Save both heads
            torch.save({
                'base_head': self.base_head.state_dict(),
                'arm_head': self.arm_head.state_dict(),
            }, actor_head_path)
        else:
            torch.save(self.mu.state_dict(), actor_head_path)

    def load_actor(self, actor_mlp_path, actor_head_path, map_location=None):
        """
        Load actor model parameters from files.

        Args:
            actor_mlp_path: Path to load actor MLP parameters from.
            actor_head_path: Path to load actor head parameters from.
            map_location: Optional device mapping for torch.load.
        """
        # Load shared MLP parameters
        mlp_state = torch.load(actor_mlp_path, map_location=map_location)
        self.actor_mlp.load_state_dict(mlp_state)

        # Load head parameters according to configuration
        head_state = torch.load(actor_head_path, map_location=map_location)
        if self.use_separate_heads:
            # Expected format: {'base_head': ..., 'arm_head': ...}
            if isinstance(head_state, dict) and 'base_head' in head_state and 'arm_head' in head_state:
                self.base_head.load_state_dict(head_state['base_head'])
                self.arm_head.load_state_dict(head_state['arm_head'])
            else:
                # Backward-compatible fallback: treat as a single head for base_head
                self.base_head.load_state_dict(head_state)
        else:
            # Single-head configuration
            self.mu.load_state_dict(head_state)
    @torch.no_grad()
    def act(self, obs_dict, hidden_state=None):
        """
        Sample action for collection during training.
        
        Args:
            obs_dict: Dictionary containing observations
            hidden_state: GRU hidden state from previous step
            
        Returns:
            result: Dictionary with actions, values, and new hidden state
        """
        mu, logstd, value, new_hidden = self._actor_critic(obs_dict, hidden_state)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            'hidden_states': new_hidden,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict, hidden_state=None):
        """
        Get deterministic action for testing.
        
        Args:
            obs_dict: Dictionary containing observations
            hidden_state: GRU hidden state from previous step
            
        Returns:
            mu: Deterministic action
            new_hidden: Updated hidden state
        """
        mu, logstd, value, new_hidden = self._actor_critic(obs_dict, hidden_state)
        return mu, new_hidden

    def _actor_critic(self, obs_dict, hidden_state=None):
        """
        Core actor-critic forward pass.
        
        Args:
            obs_dict: Dictionary with 'obs' key containing observations
                     For inference: obs shape is (batch, features)
                     For training with GRU: obs shape is (batch, seq_len, features)
            hidden_state: GRU hidden state (num_layers, batch, hidden_size)
            
        Returns:
            mu: Action mean
            logstd: Log standard deviation
            value: State value
            new_hidden: Updated hidden state (or None if GRU not used)
        """
        if isinstance(obs_dict['obs'], dict):
            vector_obs = obs_dict['obs']['vector']
            if self.use_vision:
                image_obs = obs_dict['obs']['image']
        else:
            # Fallback if obs is just tensor (legacy)
            vector_obs = obs_dict['obs']
            image_obs = None
        
        # Check if we have sequence data (for GRU training)
        # Shape: (batch, features) for inference or (batch, seq_len, features) for training
        has_sequence = vector_obs.dim() == 3
        
        if has_sequence:
            batch_size, seq_len = vector_obs.shape[0], vector_obs.shape[1]
            # Flatten for CNN processing: (batch * seq_len, features)
            vector_obs_flat = vector_obs.reshape(batch_size * seq_len, -1)
            if self.use_vision and image_obs is not None:
                # image_obs shape: (batch, seq_len, C, H, W)
                img_shape = image_obs.shape[2:]
                image_obs_flat = image_obs.reshape(batch_size * seq_len, *img_shape)
        else:
            vector_obs_flat = vector_obs
            image_obs_flat = image_obs if image_obs is not None else None

        if self.use_vision and image_obs_flat is not None:
            # Process Image
            img_features = self.cnn(image_obs_flat)
            # Concatenate
            x = torch.cat([vector_obs_flat, img_features], dim=-1)
        else:
            x = vector_obs_flat
        
        # Process through GRU if enabled
        new_hidden = None
        if self.use_gru and self.gru is not None:
            if has_sequence:
                # Reshape back to sequence: (batch * seq_len, hidden) -> (batch, seq_len, hidden)
                x = x.reshape(batch_size, seq_len, -1)
            else:
                # Add sequence dimension for GRU: (batch, features) -> (batch, 1, features)
                x = x.unsqueeze(1)
            
            # Pass through GRU
            if hidden_state is not None:
                gru_out, new_hidden = self.gru(x, hidden_state)
            else:
                gru_out, new_hidden = self.gru(x)
            
            if has_sequence:
                # Keep sequence dimension for training: (batch, seq_len, hidden)
                x = gru_out
                # Flatten for MLP: (batch * seq_len, hidden)
                x = x.reshape(batch_size * seq_len, -1)
            else:
                # Remove sequence dimension for inference: (batch, 1, hidden) -> (batch, hidden)
                x = gru_out.squeeze(1)

        x_actor = self.actor_mlp(x)
        
        # [NEW 2025-01-04] Separate heads for base and arm
        if self.use_separate_heads:
            mu_base = self.base_head(x_actor)
            mu_arm = self.arm_head(x_actor)
            # Concatenate: [base (2D), arm (6D)]
            mu = torch.cat([mu_base, mu_arm], dim=-1)
            # Combine sigmas
            sigma = torch.cat([self.sigma_base, self.sigma_arm], dim=0)
        else:
            mu = self.mu(x_actor)
            sigma = self.sigma
        
        if self.separate_value_mlp:
            x_value = self.value_mlp(x)
        else:
            x_value = x_actor # Share features if not separate
            
        value = self.value(x_value)

        # Normalize to (-1,1)
        mu = torch.tanh(mu)
        
        # Reshape output back to sequence format if needed
        if has_sequence:
            mu = mu.reshape(batch_size, seq_len, -1)
            value = value.reshape(batch_size, seq_len, -1)
        
        return mu, mu * 0 + sigma, value, new_hidden

    def forward(self, input_dict, hidden_state=None):
        """
        Forward pass for training (computes log probs, entropy, values).
        
        Args:
            input_dict: Dictionary with 'obs' and 'prev_actions'
                       For GRU: obs shape is (batch, seq_len, features)
            hidden_state: GRU hidden state
            
        Returns:
            result: Dictionary with training outputs
        """
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, new_hidden = self._actor_critic(input_dict, hidden_state)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        
        # Handle sequence dimension for GRU training
        if mu.dim() == 3:
            # mu shape: (batch, seq_len, act_dim)
            # prev_actions shape: (batch, seq_len, act_dim)
            entropy = distr.entropy().sum(dim=-1)  # (batch, seq_len)
            prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)  # (batch, seq_len)
        else:
            # Standard case: (batch, act_dim)
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
        
        result = {
            'prev_neglogp': prev_neglogp,
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'hidden_states': new_hidden,
        }
        return result