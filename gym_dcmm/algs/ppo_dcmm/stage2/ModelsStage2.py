import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DepthCNN(nn.Module):
    """
    CNN for processing depth images.
    Input: (B, 1, H, W)
    Output: (B, 256)
    
    [UPDATED 2025-01-04] Added AdaptiveAvgPool2d to handle variable image dimensions.
    This ensures that changing img_dim (84/128/224) won't cause dimension explosions.
    """
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(),
        )
        
        # [NEW 2025-01-04] AdaptiveAvgPool ensures fixed output size regardless of input resolution
        # Output will always be (32, 4, 4) = 512 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fixed output size: 32 * 4 * 4 = 512
        self.linear = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # Ensure fixed size
        x = x.flatten(1)  # Flatten to (batch, features)
        x = self.linear(x)
        return x


class ActorCritic(nn.Module):
    """
    Actor-Critic network for Stage 2 (grasping).
    
    [UPDATED 2025-01-04] Architecture improvements:
    1. Separate output heads for arm (6D) and hand (12D)
       - arm_policy: outputs 6D joint deltas
       - hand_policy: outputs 12D finger commands
    2. Separate log_std for arm (smaller, more stable) and hand (larger, more exploration)
    3. Optional: Actor can use visual features (vis_embedding)
    """
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        
        # New args for splitting
        self.state_dim = kwargs.pop('state_dim', 35)
        self.depth_pixels = kwargs.pop('depth_pixels', 84*84)
        self.img_size = kwargs.pop('img_size', 84)
        
        # [NEW 2025-01-04] Separate heads configuration
        self.use_separate_heads = kwargs.pop('use_separate_heads', True)
        self.arm_action_dim = 6   # arm joint deltas
        self.hand_action_dim = 12 # finger commands
        
        out_size = self.units[-1]
        
        # Critic: Vision Encoder
        self.critic_cnn = DepthCNN()
        
        # [NEW 2025-01-04] Optional: Actor visual embedding (small, 64D)
        self.actor_use_vision = kwargs.pop('actor_use_vision', False)
        if self.actor_use_vision:
            self.actor_cnn = DepthCNN()
            self.vis_actor_proj = nn.Linear(256, 64)
            actor_input_dim = self.state_dim + 64
        else:
            self.actor_cnn = None
            actor_input_dim = self.state_dim
        
        # Actor MLP (Takes State, optionally + visual embedding)
        self.actor_mlp_c = MLP(units=self.units, input_size=actor_input_dim)
        
        # Tracking Actor (Legacy / Placeholder)
        self.actor_mlp_t = MLP(units=self.units, input_size=self.state_dim)

        # Critic: State Encoder (MLP)
        self.value_mlp = MLP(units=self.units, input_size=self.state_dim)
        
        # Critic: Fusion Head
        # Concatenates State Features (out_size) + Visual Features (256)
        self.value_head = nn.Linear(out_size + 256, 1)
        
        # [NEW 2025-01-04] Separate output heads for arm and hand
        if self.use_separate_heads:
            self.arm_head = nn.Linear(out_size, self.arm_action_dim)
            self.hand_head = nn.Linear(out_size, self.hand_action_dim)
            nn.init.orthogonal_(self.arm_head.weight, gain=0.01)
            nn.init.orthogonal_(self.hand_head.weight, gain=0.01)
            
            # Separate log_std: arm smaller (stable), hand larger (exploration)
            self.sigma_arm = nn.Parameter(
                torch.full((self.arm_action_dim,), -2.5, dtype=torch.float32), requires_grad=True)
            self.sigma_hand = nn.Parameter(
                torch.full((self.hand_action_dim,), -1.5, dtype=torch.float32), requires_grad=True)
        else:
            self.mu_c = torch.nn.Linear(out_size, actions_num)
            self.sigma_c = nn.Parameter(
                torch.full((actions_num,), -2.0, dtype=torch.float32), requires_grad=True)
            nn.init.orthogonal_(self.mu_c.weight, gain=0.01)
        
        # Legacy tracking head (placeholder)
        self.mu_t = torch.nn.Linear(out_size, actions_num-12)
        self.sigma_t = nn.Parameter(
            torch.full((actions_num-12,), -2.0, dtype=torch.float32), requires_grad=True)
        
        # Bounds for log_sigma to prevent entropy explosion/collapse
        self.log_sigma_min = -5.0  # exp(-5) ≈ 0.007
        self.log_sigma_max = 0.0   # exp(0) = 1.0

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        
        nn.init.orthogonal_(self.mu_t.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        """
        Core actor-critic forward pass.
        
        [UPDATED 2025-01-04] Supports separate arm/hand output heads.
        """
        # obs_dict['obs'] is the flattened tensor (B, state_dim + depth_pixels)
        obs = obs_dict['obs']
        
        # 1. Split
        state = obs[:, :self.state_dim]
        depth_flat = obs[:, self.state_dim:]
        
        # 2. Process Depth
        # Reshape to (B, 1, H, W) and Normalize (0-255 -> 0-1)
        depth_img = depth_flat.view(-1, 1, self.img_size, self.img_size)
        depth_img = depth_img.float() / 255.0
        
        # 3. Actor Path
        # [NEW 2025-01-04] Optional: Actor uses visual embedding
        if self.actor_use_vision and self.actor_cnn is not None:
            actor_vis_feat = self.actor_cnn(depth_img)
            actor_vis_embed = self.vis_actor_proj(actor_vis_feat)
            actor_input = torch.cat([state, actor_vis_embed], dim=1)
        else:
            actor_input = state
        
        x_c = self.actor_mlp_c(actor_input)
        
        # [NEW 2025-01-04] Separate heads for arm and hand
        if self.use_separate_heads:
            mu_arm = self.arm_head(x_c)
            mu_hand = self.hand_head(x_c)
            # Concatenate: [arm (6D), hand (12D)] = 18D
            # But we need 20D for compatibility (2 base + 6 arm + 12 hand)
            # Prepend zeros for base (locked in Stage 2)
            mu_base = torch.zeros(x_c.shape[0], 2, device=x_c.device, dtype=x_c.dtype)
            mu = torch.cat([mu_base, mu_arm, mu_hand], dim=-1)
            
            # Combine sigmas with clamping
            sigma_arm_clamped = torch.clamp(self.sigma_arm, self.log_sigma_min, self.log_sigma_max)
            sigma_hand_clamped = torch.clamp(self.sigma_hand, self.log_sigma_min, self.log_sigma_max)
            # Include base sigma (zeros, since base is locked)
            sigma_base = torch.zeros(2, device=x_c.device, dtype=x_c.dtype)
            log_sigma = torch.cat([sigma_base, sigma_arm_clamped, sigma_hand_clamped], dim=0)
            log_sigma = log_sigma.unsqueeze(0).expand(mu.shape[0], -1)
        else:
            mu = self.mu_c(x_c)
            log_sigma_c = torch.clamp(self.sigma_c, self.log_sigma_min, self.log_sigma_max)
            log_sigma = log_sigma_c.unsqueeze(0).expand(mu.shape[0], -1)
        
        # 4. Critic Path (State + Depth)
        # State Features
        state_feat = self.value_mlp(state)
        # Visual Features
        vis_feat = self.critic_cnn(depth_img)
        # Fusion
        combined = torch.cat([state_feat, vis_feat], dim=1)
        value = self.value_head(combined)

        # Normalize mu to (-1,1)
        mu = torch.tanh(mu)

        return mu, log_sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result