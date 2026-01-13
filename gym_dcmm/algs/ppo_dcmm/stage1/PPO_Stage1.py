import os, sys
sys.path.append(os.path.abspath('../../../gym_dcmm'))
import math
import time
import torch
import torch.distributed as dist

import wandb

import numpy as np


from gym_dcmm.algs.ppo_dcmm.experience import ExperienceBuffer
from gym_dcmm.algs.ppo_dcmm.stage1.ModelsStage1 import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import AverageScalarMeter, RunningMeanStd
import configs.env.DcmmCfg as DcmmCfg

from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

class PPO_Stage1(object):
    def __init__(self, env, output_dif, full_config):
        self.rank = -1
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = int(self.ppo_config['num_actors'])
        print("num_actors: ", self.num_actors)
        self.actions_num = self.env.get_attr("act_t_dim")[0]
        print("actions_num: ", self.actions_num)
        self.actions_low = self.env.get_attr("actions_low")[0]
        self.actions_high = self.env.get_attr("actions_high")[0]
        # self.obs_shape = self.env.observation_space.shape
        
        # Define obs_shape as dict for Vision
        # Vector dim
        vector_dim = self.env.get_attr("obs_t_dim")[0]
        # Image shape (C, H, W) - hardcoded or from config
        # self.ppo_config['img_dim'] is [112, 112]. We need C.
        # DcmmVecEnv uses depth_array, so C=1.
        img_h, img_w = self.ppo_config['img_dim']
        image_shape = (1, img_h, img_w)
        
        self.obs_shape = {
            'vector': (vector_dim,),
            'image': image_shape
        }
        
        self.full_action_dim = self.env.get_attr("act_c_dim")[0]
        self.task = self.env.get_attr("task")[0]
        
        # GRU configuration
        self.use_gru = DcmmCfg.gru_config.enabled
        self.gru_hidden_size = DcmmCfg.gru_config.hidden_size
        self.gru_num_layers = DcmmCfg.gru_config.num_layers
        
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'separate_value_mlp': self.network_config.get('separate_value_mlp', True),
            'use_vision': full_config.get('use_vision', True),
            'use_gru': self.use_gru,
            'gru_hidden_size': self.gru_hidden_size,
            'gru_num_layers': self.gru_num_layers,
        }
        print("net_config: ", net_config)
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        
        # Initialize hidden state for all environments
        if self.use_gru:
            self.hidden_state = self.model.get_initial_hidden(self.num_actors, self.device)
        else:
            self.hidden_state = None
        
        # RunningMeanStd only for vector part
        self.running_mean_std = RunningMeanStd((vector_dim,)).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config['learning_rate'])
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr, eps=1e-5)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.action_track_denorm = self.ppo_config['action_track_denorm']
        self.action_catch_denorm = self.ppo_config['action_catch_denorm']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scale_value = self.ppo_config['reward_scale_value']
        self.clip_value_loss = self.ppo_config['clip_value_loss']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(
                self.init_lr,
                self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)
        self.episode_test_rewards = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(self.ppo_config['test_num_episodes'])

        self.obs = None
        self.epoch_num = 0
        
        # GRU sequence length for truncated BPTT
        self.gru_seq_len = 32  # Chunk horizon into sequences of this length
        
        # print("self.obs_shape[0]: ", type(self.obs_shape[0]))
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_shape, self.actions_num, self.device,
            use_gru=self.use_gru, gru_hidden_size=self.gru_hidden_size, 
            gru_num_layers=self.gru_num_layers,
            seq_len=self.gru_seq_len,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.max_test_steps = self.ppo_config['max_test_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0
        self.scaler = GradScaler()

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, extra_stats=None):
        """
        Write training statistics to WandB and TensorBoard.
        
        [ENHANCED 2025-01-13] Added comprehensive metrics for training analysis:
        - Gradient statistics (norm, max)
        - Policy ratio statistics
        - Advantage statistics
        - Value function statistics
        - Action statistics
        """
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
            'losses/actor_loss': torch.mean(torch.stack(a_losses)).item(),
            'losses/bounds_loss': torch.mean(torch.stack(b_losses)).item(),
            'losses/critic_loss': torch.mean(torch.stack(c_losses)).item(),
            'losses/entropy': torch.mean(torch.stack(entropies)).item(),
            'info/last_lr': self.last_lr,
            'info/e_clip': self.e_clip,
            'info/kl': torch.mean(torch.stack(kls)).item(),
            # [NEW] Training progress info
            'info/epoch': self.epoch_num,
            'info/progress_percent': self.agent_steps / self.max_agent_steps * 100,
        }
        
        # [NEW] Add extra statistics if provided
        if extra_stats:
            log_dict.update(extra_stats)
        
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)
    
    def _compute_gradient_stats(self):
        """Compute gradient statistics for monitoring training health."""
        total_norm = 0.0
        max_norm = 0.0
        param_count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                param_count += 1
        total_norm = total_norm ** 0.5
        return {
            'gradients/total_norm': total_norm,
            'gradients/max_norm': max_norm,
            'gradients/param_count': param_count,
        }
    
    def _get_training_suggestions(self, mean_rewards, mean_success, latest_kl, latest_entropy, 
                                   latest_a_loss, latest_c_loss, reward_stats=None):
        """
        Generate training suggestions based on observed metrics.
        
        Returns a list of suggestion strings for terminal output.
        """
        suggestions = []
        
        # KL divergence analysis
        if latest_kl > 0.02:
            suggestions.append("⚠️ KL过高(>{:.4f}): 考虑降低学习率或减小e_clip".format(latest_kl))
        elif latest_kl < 0.001:
            suggestions.append("ℹ️ KL过低(<0.001): 策略更新保守，考虑提高学习率")
        
        # Entropy analysis
        if latest_entropy < 0.1:
            suggestions.append("⚠️ 熵过低(<0.1): 探索不足，考虑提高entropy_coef")
        elif latest_entropy > 2.0:
            suggestions.append("ℹ️ 熵较高(>2.0): 策略随机性大，训练仍在早期阶段")
        
        # Critic loss analysis
        if latest_c_loss > 1.0:
            suggestions.append("⚠️ Critic Loss高(>1.0): 价值估计不准，检查奖励缩放")
        
        # Success rate analysis
        if mean_success < 0.1 and self.agent_steps > 5e6:
            suggestions.append("❌ 成功率低(<10%): 任务可能过难，考虑调整课程学习参数")
        elif mean_success > 0.8:
            suggestions.append("✅ 成功率高(>80%): 训练效果良好")
        
        # Reward analysis with detailed components
        if reward_stats:
            # Check for reward hacking indicators
            ee_progress = reward_stats.get('rewards/ee_progress_mean', 0)
            base_progress = reward_stats.get('rewards/base_progress_mean', 0)
            stagnation = reward_stats.get('rewards/stagnation_penalty_mean', 0)
            
            if ee_progress < 0 and base_progress < 0:
                suggestions.append("⚠️ EE和底盘都在远离目标: 检查奖励函数或动作空间")
            
            if stagnation < -0.05:
                suggestions.append("⚠️ 频繁触发停滞惩罚: 智能体可能卡住，检查动作范围")
            
            # Distance analysis
            ee_dist = reward_stats.get('distance/ee_distance_mean', 0)
            if ee_dist > 1.5:
                suggestions.append(f"ℹ️ 平均EE距离{ee_dist:.2f}m: 智能体尚未学会接近目标")
            elif ee_dist < 0.2:
                suggestions.append(f"✅ 平均EE距离{ee_dist:.2f}m: 智能体已学会接近目标")
        
        return suggestions

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.agent_steps = self.batch_size
        
        # Initialize hidden states for GRU
        if self.use_gru:
            self.hidden_state = self.model.get_initial_hidden(self.num_actors, self.device)

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            self.storage.data_dict = None

            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)
            
            
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (
                self.batch_size ) \
                / (time.time() - _last_t)
            _last_t = time.time()
            
            # Get additional metrics
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()
            
            # Loss metrics (get latest values)
            latest_a_loss = torch.mean(torch.stack(a_losses)).item()
            latest_c_loss = torch.mean(torch.stack(c_losses)).item()
            latest_b_loss = torch.mean(torch.stack(b_losses)).item() if b_losses else 0.0
            latest_entropy = torch.mean(torch.stack(entropies)).item()
            latest_kl = torch.mean(torch.stack(kls)).item()
            
            # [NEW] Compute gradient statistics
            grad_stats = self._compute_gradient_stats()
            
            # [NEW] Collect extra statistics for wandb
            extra_stats = {
                **grad_stats,
                'metrics/mean_rewards': mean_rewards,
                'metrics/mean_lengths': mean_lengths,
                'metrics/success_rate': mean_success,
            }
            
            # Get reward decomposition statistics
            reward_stats = None
            avp_stats = None
            if hasattr(self.env, 'env_method'):
                try:
                    reward_stats = self.env.env_method("get_reward_stats")[0]
                    if reward_stats:
                        extra_stats.update(reward_stats)
                except Exception:
                    pass
                
                try:
                    avp_stats = self.env.env_method("get_avp_stats")[0]
                    if avp_stats:
                        extra_stats.update(avp_stats)
                except Exception:
                    pass
            
            # ================================================================
            # ENHANCED TERMINAL OUTPUT
            # ================================================================
            print("\n" + "="*110)
            print(f"{'🚀 STAGE 1 TRAINING PROGRESS (Tracking/Approaching)':^110}")
            print("="*110)
            
            # Primary metrics
            print(f"┌{'─'*108}┐")
            print(f"│ {'📊 BASIC INFO':<106} │")
            print(f"├{'─'*108}┤")
            print(f"│ {'Epoch':<15}: {self.epoch_num:>6d}      │ {'Steps':<15}: {int(self.agent_steps // 1e3):>6d}K / {int(self.max_agent_steps // 1e6):>2d}M ({self.agent_steps / self.max_agent_steps * 100:>5.1f}%)      │ {'Best Reward':<15}: {self.best_rewards:>8.2f}  │")
            print(f"└{'─'*108}┘")
            
            # Performance metrics
            print(f"┌{'─'*108}┐")
            print(f"│ {'⚡ PERFORMANCE':<106} │")
            print(f"├{'─'*108}┤")
            print(f"│ {'FPS (Overall)':<15}: {all_fps:>8.1f}  │ {'FPS (Batch)':<15}: {last_fps:>8.1f}  │ {'Collect Time':<15}: {self.data_collect_time / 60:>6.1f}min │ {'Train Time':<15}: {self.rl_train_time / 60:>6.1f}min │")
            print(f"└{'─'*108}┘")
            
            # Episode statistics
            print(f"┌{'─'*108}┐")
            print(f"│ {'📈 EPISODE STATS':<106} │")
            print(f"├{'─'*108}┤")
            print(f"│ {'Mean Reward':<15}: {mean_rewards:>8.2f}  │ {'Mean Length':<15}: {mean_lengths:>8.1f}  │ {'Success Rate':<15}: {mean_success * 100:>6.1f}%   │ {'Learning Rate':<15}: {self.last_lr:>.2e}   │")
            print(f"└{'─'*108}┘")
            
            # Training losses
            print(f"┌{'─'*108}┐")
            print(f"│ {'🔧 TRAINING METRICS':<106} │")
            print(f"├{'─'*108}┤")
            print(f"│ {'Actor Loss':<15}: {latest_a_loss:>8.4f}  │ {'Critic Loss':<15}: {latest_c_loss:>8.4f}  │ {'Bounds Loss':<15}: {latest_b_loss:>8.5f} │ {'Entropy':<15}: {latest_entropy:>8.4f}  │")
            print(f"│ {'KL Divergence':<15}: {latest_kl:>8.5f}  │ {'Grad Norm':<15}: {grad_stats['gradients/total_norm']:>8.4f}  │ {'Grad Max':<15}: {grad_stats['gradients/max_norm']:>8.4f} │ {'e_clip':<15}: {self.e_clip:>8.4f}  │")
            print(f"└{'─'*108}┘")
            
            # Reward decomposition (if available)
            if reward_stats:
                print(f"┌{'─'*108}┐")
                print(f"│ {'🎯 REWARD DECOMPOSITION':<106} │")
                print(f"├{'─'*108}┤")
                ee_progress = reward_stats.get('rewards/ee_progress_mean', 0)
                base_progress = reward_stats.get('rewards/base_progress_mean', 0)
                orientation = reward_stats.get('rewards/orientation_mean', 0)
                alive = reward_stats.get('rewards/alive_penalty_mean', 0)
                stagnation = reward_stats.get('rewards/stagnation_penalty_mean', 0)
                collision = reward_stats.get('rewards/collision_mean', 0)
                success = reward_stats.get('rewards/success_mean', 0)
                touch = reward_stats.get('rewards/touch_mean', 0)
                milestone = reward_stats.get('rewards/milestone_mean', 0)
                print(f"│ {'EE Progress':<15}: {ee_progress:>+8.4f} │ {'Base Progress':<15}: {base_progress:>+8.4f} │ {'Orientation':<15}: {orientation:>+8.4f} │ {'Touch':<15}: {touch:>+8.4f} │")
                print(f"│ {'Alive Penalty':<15}: {alive:>+8.4f} │ {'Stagnation':<15}: {stagnation:>+8.4f} │ {'Collision':<15}: {collision:>+8.4f} │ {'Milestone':<15}: {milestone:>+8.4f} │")
                print(f"├{'─'*108}┤")
                print(f"│ {'📏 DISTANCE METRICS':<106} │")
                print(f"├{'─'*108}┤")
                ee_dist = reward_stats.get('distance/ee_distance_mean', 0)
                base_dist = reward_stats.get('distance/base_distance_mean', 0)
                arm_dist = reward_stats.get('distance/arm_reach_distance_mean', 0)
                print(f"│ {'EE Distance':<15}: {ee_dist:>7.3f}m  │ {'Base Distance':<15}: {base_dist:>7.3f}m  │ {'Arm Reach Dist':<15}: {arm_dist:>7.3f}m  │{'':>35} │")
                # Curriculum info
                curr_diff = reward_stats.get('curriculum/difficulty', 0)
                curr_stem = reward_stats.get('curriculum/w_stem', 0)
                curr_orient = reward_stats.get('curriculum/orient_power', 1.0)
                print(f"│ {'Curriculum Diff':<15}: {curr_diff:>8.3f} │ {'Stem Weight':<15}: {curr_stem:>8.3f} │ {'Orient Power':<15}: {curr_orient:>8.3f} │{'':>35} │")
                print(f"└{'─'*108}┘")
            
            # AVP statistics (if available)
            if avp_stats:
                print(f"┌{'─'*108}┐")
                print(f"│ {'🔮 AVP (Asymmetric Value Propagation)':<106} │")
                print(f"├{'─'*108}┤")
                avp_lambda = avp_stats.get('avp/lambda', 0)
                avp_value = avp_stats.get('avp/critic_value_mean', 0)
                avp_reward = avp_stats.get('avp/reward_mean', 0)
                avp_conf = avp_stats.get('avp/confidence_mean', 0)
                dist_gate = avp_stats.get('avp/distance_gate_ratio', 0)
                active_ratio = avp_stats.get('avp/active_ratio', 0)
                print(f"│ {'Lambda':<15}: {avp_lambda:>8.4f}  │ {'Critic Value':<15}: {avp_value:>8.4f}  │ {'AVP Reward':<15}: {avp_reward:>+8.4f} │ {'Confidence':<15}: {avp_conf:>8.4f}  │")
                print(f"│ {'Dist Gate':<15}: {dist_gate*100:>7.1f}%  │ {'Active Ratio':<15}: {active_ratio*100:>7.1f}%  │{'':>35} │{'':>35} │")
                print(f"└{'─'*108}┘")
            
            # Training suggestions
            suggestions = self._get_training_suggestions(
                mean_rewards, mean_success, latest_kl, latest_entropy,
                latest_a_loss, latest_c_loss, reward_stats
            )
            if suggestions:
                print(f"┌{'─'*108}┐")
                print(f"│ {'💡 TRAINING SUGGESTIONS':<106} │")
                print(f"├{'─'*108}┤")
                for sug in suggestions[:4]:  # Limit to 4 suggestions
                    print(f"│ {sug:<106} │")
                print(f"└{'─'*108}┘")
            
            print("="*110 + "\n")

            # Write stats to wandb and tensorboard
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, extra_stats)

            # Additional tensorboard logging
            self.writer.add_scalar('metrics/episode_rewards_per_step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('metrics/episode_lengths_per_step', mean_lengths, self.agent_steps)
            self.writer.add_scalar('metrics/episode_success_per_step', mean_success, self.agent_steps)

            
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'last'))

            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                # remove previous best file
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        # Define expected tuple lengths for storage data
        STORAGE_DATA_LEN_WITHOUT_HIDDEN = 8
        STORAGE_DATA_LEN_WITH_HIDDEN = 9

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                # Get data from storage (with or without hidden states)
                storage_data = self.storage[i]
                if self.use_gru and len(storage_data) == STORAGE_DATA_LEN_WITH_HIDDEN:
                    value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                        returns, actions, obs, hidden_states = storage_data
                    # For sequence-based training:
                    # hidden_states shape: (batch, seq_len, num_layers, hidden_size)
                    # We only need the hidden state at the start of each sequence
                    # Take hidden state at t=0: (batch, num_layers, hidden_size)
                    hidden_states = hidden_states[:, 0, :, :]
                    # Transpose to (num_layers, batch, hidden_size) for GRU
                    hidden_states = hidden_states.transpose(0, 1).contiguous()
                else:
                    value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                        returns, actions, obs = storage_data[:STORAGE_DATA_LEN_WITHOUT_HIDDEN]
                    hidden_states = None

                # obs is a dict {'vector': ..., 'image': ...}
                # For GRU: obs['vector'] shape is (batch, seq_len, features)
                # For MLP: obs['vector'] shape is (batch, features)
                if self.use_gru and obs['vector'].dim() == 3:
                    # Normalize each timestep in the sequence
                    batch_size, seq_len = obs['vector'].shape[0], obs['vector'].shape[1]
                    vector_flat = obs['vector'].reshape(batch_size * seq_len, -1)
                    vector_normalized = self.running_mean_std(vector_flat)
                    vector_obs = vector_normalized.reshape(batch_size, seq_len, -1)
                else:
                    vector_obs = self.running_mean_std(obs['vector'])

                processed_obs = {
                    'vector': vector_obs,
                    'image': obs['image']
                }

                batch_dict = {
                    'prev_actions': actions,
                    'obs': processed_obs,
                }

                # =================================================================================
                # [AMP 修改开始] 使用混合精度上下文
                # =================================================================================
                with autocast():
                    res_dict = self.model(batch_dict, hidden_states)
                    action_log_probs = res_dict['prev_neglogp']
                    values = res_dict['values']
                    entropy = res_dict['entropy']
                    mu = res_dict['mus']
                    sigma = res_dict['sigmas']

                    # actor loss
                    ratio = torch.exp(old_action_log_probs - action_log_probs)
                    surr1 = advantage * ratio
                    surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                    a_loss = torch.max(-surr1, -surr2)

                    # critic loss
                    if self.clip_value_loss:
                        value_pred_clipped = value_preds + \
                                             (values - value_preds).clamp(-self.e_clip, self.e_clip)
                        value_losses = (values - returns) ** 2
                        value_losses_clipped = (value_pred_clipped - returns) ** 2
                        c_loss = torch.max(value_losses, value_losses_clipped)
                    else:
                        c_loss = (values - returns) ** 2

                    # bounded loss
                    if self.bounds_loss_coef > 0:
                        soft_bound = 1.1
                        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                        b_loss = (mu_loss_low + mu_loss_high).sum(dim=-1)
                    else:
                        b_loss = 0

                    a_loss, c_loss, entropy, b_loss = [
                        torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                    loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef \
                           + b_loss * self.bounds_loss_coef

                # [AMP 修改] 使用 Scaler 进行反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                if self.truncate_grads:
                    # [AMP 修改] Unscale 之后才能进行梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

                # [AMP 修改] Step 和 Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # =================================================================================
                # [AMP 修改结束]
                # =================================================================================

                with torch.no_grad():
                    # Flatten mu/sigma for KL computation if needed
                    mu_flat = mu.reshape(-1, mu.shape[-1]) if mu.dim() == 3 else mu
                    sigma_flat = sigma.reshape(-1, sigma.shape[-1]) if sigma.dim() == 3 else sigma
                    old_mu_flat = old_mu.reshape(-1, old_mu.shape[-1]) if old_mu.dim() == 3 else old_mu
                    old_sigma_flat = old_sigma.reshape(-1, old_sigma.shape[-1]) if old_sigma.dim() == 3 else old_sigma
                    kl_dist = policy_kl(mu_flat.detach(), sigma_flat.detach(), old_mu_flat, old_sigma_flat)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            kls.append(av_kls)

            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = self.adjust_learning_rate_cos(mini_ep)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls
    
    def obs2tensor(self, obs):
        # Map the step result to tensor
        # Vector part
        if self.task == 'Catching':
            obs_array = np.concatenate((
                        obs["base"]["v_lin_2d"], 
                        obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
                        obs["object"]["pos3d"], obs["object"]["v_lin_3d"], 
                        obs["hand"],
                        ), axis=1)
        else:
            # Build list of observation components
            obs_components = [
                obs["base"]["v_lin_2d"], 
            ]
            # [FIX 2025-01-04] Include base heading observation if present
            # This provides explicit chassis orientation info to the network
            if "heading" in obs["base"]:
                heading = obs["base"]["heading"]
                if heading.ndim == 1:
                    # Single env case: (4,) -> (1, 4)
                    heading = heading[np.newaxis, :]
                obs_components.append(heading)
            obs_components.extend([
                obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
                obs["object"]["pos3d"],
            ])
            # Add validity flag if present (helps network distinguish dropped vs valid observations)
            if "is_valid" in obs["object"]:
                # is_valid is now (N, 1) from vec env stacking or (1,) for single env
                is_valid = obs["object"]["is_valid"]
                if is_valid.ndim == 1:
                    # Single env case: (1,) -> (1, 1)
                    is_valid = is_valid[np.newaxis, :]
                obs_components.append(is_valid)
            obs_array = np.concatenate(obs_components, axis=1)
        
        vector_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        
        # Image part
        # obs["depth"] is (N, H, W) or (N, 1, H, W)?
        # render() returns (N, H, W) usually if multiple envs.
        # We need (N, C, H, W).
        depth_imgs = obs["depth"]
        if len(depth_imgs.shape) == 3:
             # (N, H, W) -> (N, 1, H, W)
             depth_imgs = np.expand_dims(depth_imgs, axis=1)
        
        image_tensor = torch.tensor(depth_imgs, dtype=torch.float32).to(self.device)
        
        return {'vector': vector_tensor, 'image': image_tensor}

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        # De-normalize the actions
        if self.task == 'Tracking':
            base_tensor = actions[:, :2] * self.action_track_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_track_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_track_denorm[2]
        else:
            base_tensor = actions[:, :2] * self.action_catch_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_catch_denorm[2]
        actions_dict = {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor
        }
        return actions_dict

    def model_act(self, obs_dict, inference=False):
        # obs_dict['obs'] is {'vector': ..., 'image': ...}
        # Normalize vector part
        vector_obs = self.running_mean_std(obs_dict['obs']['vector'])
        # Image part is already [0, 1] or similar, usually doesn't need RunningMeanStd
        # But we might want to divide by 255 if it was uint8. 
        # Here it comes from depth_2_meters which is float meters.
        # We might want to clamp it? For now pass as is.
        
        processed_obs = {
            'vector': vector_obs,
            'image': obs_dict['obs']['image']
        }
        
        input_dict = {
            'obs': processed_obs,
        }
        if not inference:
            res_dict = self.model.act(input_dict, self.hidden_state)
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
            # Update hidden state for next step
            if self.use_gru and 'hidden_states' in res_dict:
                self.hidden_state = res_dict['hidden_states']
        else:
            res_dict = {}
            action, new_hidden = self.model.act_inference(input_dict, self.hidden_state)
            res_dict['actions'] = action
            # Update hidden state for next step
            if self.use_gru and new_hidden is not None:
                self.hidden_state = new_hidden
        return res_dict

    def play_steps(self):
        # Update curriculum with current global step
        if hasattr(self.env, 'env_method'):
            self.env.env_method("set_global_step", int(self.agent_steps))
        elif hasattr(self.env, 'set_global_step'):
            self.env.set_global_step(int(self.agent_steps))

        for n in range(self.horizon_length):
            # Store hidden state BEFORE taking action (for training)
            if self.use_gru and self.hidden_state is not None:
                self.storage.update_hidden_states(n, self.hidden_state)
            
            res_dict = self.model_act(self.obs)
            # Collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # Do env step
            # Clamp the actions of the action space
            actions = res_dict['actions']
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            # print("actions_dict: ", actions_dict)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scale_value * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            # print("self.dones: ", self.dones)
            done_indices = self.dones.nonzero(as_tuple=False)
            # ... (在 done_indices = ... 之后)

            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            # [Fix 2025-12-21] 正确统计成功率逻辑
            # 初始化默认为 0 (失败)
            real_success = torch.zeros(len(done_indices), dtype=torch.float32, device=self.device)

            # Gymnasium VectorEnv 标准: 结束回合的信息保存在 "final_info" 中
            if "final_info" in infos:
                final_infos = infos["final_info"]
                for i, idx in enumerate(done_indices):
                    env_idx = idx.item()
                    # final_info[env_idx] 是一个字典，包含该环境结束时的所有 info
                    info_item = final_infos[env_idx]
                    if info_item and "is_success" in info_item:
                        if info_item["is_success"]:
                            real_success[i] = 1.0

            # 如果不是 VectorEnv (单个环境)，或者是旧版 Gym，可能直接在 info 里
            elif "is_success" in infos:
                # 注意：这里拿到的通常是 Reset 后的 info (False)，除非 Wrapper 特殊处理过
                # 但为了防止键缺失导致的 100% 错误，这里加上判断
                success_arr = infos['is_success']
                # 转换为 Tensor
                if isinstance(success_arr, np.ndarray) or isinstance(success_arr, list):
                    s_tensor = torch.tensor(success_arr, dtype=torch.float32, device=self.device)
                    real_success = s_tensor[done_indices]

            # 只有当确实从 info 中提取到了成功信号，才更新；坚决不再回退到 truncates
            self.episode_success.update(real_success)

            assert isinstance(infos, dict), 'Info Should be a Dict'
            # print("infos: ", infos)
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            # Reset hidden states for finished episodes
            if self.use_gru and self.hidden_state is not None:
                # Get done mask and reset hidden for done environments
                done_mask = self.dones.bool()
                if done_mask.any():
                    # hidden_state shape: (num_layers, num_envs, hidden_size)
                    # Reset hidden state to zeros for environments that are done
                    self.hidden_state[:, done_mask, :] = 0.0

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps = (self.agent_steps + self.batch_size)
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            # For GRU mode, values/returns have shape (num_seqs, seq_len, 1)
            # Flatten to (N, 1) for RunningMeanStd, then reshape back
            original_shape = values.shape
            if values.dim() == 3:
                values_flat = values.reshape(-1, 1)
                returns_flat = returns.reshape(-1, 1)
                values_flat = self.value_mean_std(values_flat)
                returns_flat = self.value_mean_std(returns_flat)
                values = values_flat.reshape(original_shape)
                returns = returns_flat.reshape(original_shape)
            else:
                values = self.value_mean_std(values)
                returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def play_test_steps(self):
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            # Do env step
            # Clamp the actions of the action space 
            actions = res_dict['actions']
            actions[:, :] = torch.clamp(actions[:, :], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim - actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)

            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)

            # Update current rewards/lengths
            self.current_rewards += rewards
            self.current_lengths += 1

            # Find finished envs
            done_indices = self.dones.nonzero(as_tuple=False)

            # Update basic stats
            self.episode_test_rewards.update(self.current_rewards[done_indices])
            self.episode_test_lengths.update(self.current_lengths[done_indices])

            # ====================================================
            # [Fix 2025-12-21] 正确统计测试阶段的成功率
            # ====================================================
            real_success = torch.zeros(len(done_indices), dtype=torch.float32, device=self.device)

            if "final_info" in infos:
                final_infos = infos["final_info"]
                for i, idx in enumerate(done_indices):
                    env_idx = idx.item()
                    info_item = final_infos[env_idx]
                    # 检查是否成功
                    if info_item and "is_success" in info_item:
                        if info_item["is_success"]:
                            real_success[i] = 1.0

            elif "is_success" in infos:
                success_arr = infos['is_success']
                if isinstance(success_arr, np.ndarray) or isinstance(success_arr, list):
                    s_tensor = torch.tensor(success_arr, dtype=torch.float32, device=self.device)
                    real_success = s_tensor[done_indices]

            # 更新测试成功率 (注意这里用的是 episode_test_success)
            self.episode_test_success.update(real_success)
            # ====================================================

            # Log extra info
            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            # Reset rewards/lengths for finished envs
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            # Reset hidden states for finished episodes (GRU)
            if self.use_gru and self.hidden_state is not None:
                done_mask = self.dones.bool()
                if done_mask.any():
                    self.hidden_state[:, done_mask, :] = 0.0

        # 这一步是为了让 agent_steps 计数增加，虽然 test 不训练，但保持计数一致性
        # res_dict = self.model_act(self.obs) # Test 阶段其实不需要再计算 value
        self.agent_steps = (self.agent_steps + self.batch_size)

    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.test_steps = self.batch_size
        
        # Initialize hidden states for GRU
        if self.use_gru:
            self.hidden_state = self.model.get_initial_hidden(self.num_actors, self.device)

        while self.test_steps < self.max_test_steps:
            self.play_test_steps()
            self.storage.data_dict = None
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            mean_success = self.episode_test_success.get_mean()
            print("## Sample Length %d ##" % len(self.episode_test_rewards))
            print("mean_rewards: ", mean_rewards)
            print("mean_lengths: ", mean_lengths)
            print("mean_success: ", mean_success)
            # wandb.log({
            #     'metrics/episode_test_rewards': mean_rewards,
            #     'metrics/episode_test_lengths': mean_lengths,
            # }, step=self.agent_steps)

    def adjust_learning_rate_cos(self, epoch):
        lr = self.init_lr * 0.5 * (
            1. + math.cos(
                math.pi * (self.agent_steps + epoch / self.mini_epochs_num) / self.max_agent_steps))
        return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()

class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr

class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)
