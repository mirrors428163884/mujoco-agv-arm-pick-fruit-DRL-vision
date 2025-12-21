import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import os
from omegaconf import DictConfig, OmegaConf
from gym_dcmm.envs.stage2.DcmmVecEnvStage2 import DcmmVecEnvStage2
from gym_dcmm.algs.ppo_dcmm.stage2.PPO_Stage2 import PPO_Stage2

# 解决 Hydra 注册问题
try:
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
except:
    pass


def get_full_scene_render(env, lookat, distance=4.0):
    """虚拟全局上帝视角相机：捕捉环境全貌"""
    render_w, render_h = 1000, 1000
    env.Dcmm.model.vis.global_.offwidth = render_w
    env.Dcmm.model.vis.global_.offheight = render_h

    renderer = mujoco.Renderer(env.Dcmm.model, height=render_h, width=render_w)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat = np.array(lookat)
    cam.distance = distance
    cam.elevation = -90  # 垂直向下拍摄
    cam.azimuth = 90

    renderer.update_scene(env.Dcmm.data, camera=cam)
    rgb_img = renderer.render()
    renderer.close()
    return rgb_img


def get_value_at_pos(env, agent, x, y):
    """移动物体并计算 Critic Value"""
    obj_body_id = mujoco.mj_name2id(env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, env.object_name)
    if obj_body_id != -1:
        env.Dcmm.data.body(env.object_name).xpos[0:3] = [x, y, 0.82]
        mocap_id = env.Dcmm.model.body_mocapid[obj_body_id]
        if mocap_id != -1:
            env.Dcmm.data.mocap_pos[mocap_id] = [x, y, 0.82]

    mujoco.mj_forward(env.Dcmm.model, env.Dcmm.data)

    state_obs = env.obs_manager.get_state_obs_stage2()
    depth_obs = env.render_manager.get_depth_obs(width=env.img_width, height=env.img_height, add_noise=False,
                                                 add_holes=False)

    obs_dict = {'state': state_obs[np.newaxis, :], 'depth': depth_obs[np.newaxis, :]}
    obs_tensor = agent.obs2tensor(obs_dict)

    with torch.no_grad():
        res = agent.model_act({'obs': obs_tensor})
        value = res['values']
    return value.item()


@hydra.main(config_name='config_stage2', config_path='configs')
def main(config: DictConfig):
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), "outputs/best_reward_-5.72.pth")
    config.checkpoint_catching = checkpoint_path
    config.num_envs = 1
    config.test = True
    config.rl_device = f'cuda:{config.device_id}'

    # 1. 环境初始化
    env = DcmmVecEnvStage2(task='Catching', img_size=(84, 84), device=config.rl_device)
    env.get_attr = lambda attr_name: [getattr(env, attr_name)]

    # 2. 模型加载
    agent = PPO_Stage2(env, output_dif="vis_temp", full_config=config)
    agent.restore_test(checkpoint_path)
    agent.set_eval()

    # 3. 记录物体（目标点）的初始物理位置
    env.reset()
    obj_body_id = mujoco.mj_name2id(env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, env.object_name)
    target_pos = env.Dcmm.data.body(env.object_name).xpos[0:3].copy()
    print(f"Target Point (on plant): X={target_pos[0]:.3f}, Y={target_pos[1]:.3f}")

    # 4. 定义扫描区域
    x_range = np.linspace(-1.5, 1.5, 50)
    y_range = np.linspace(-0.5, 2.0, 50)
    value_map = np.zeros((len(y_range), len(x_range)))

    print("Scanning...")
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            value_map[i, j] = get_value_at_pos(env, agent, x, y)
        if i % 10 == 0: print(f"Progress: {int(i / len(y_range) * 100)}%")

    # 5. 复位并拍摄全貌图
    env.Dcmm.data.body(env.object_name).xpos[0:3] = target_pos
    mocap_id = env.Dcmm.model.body_mocapid[obj_body_id]
    if mocap_id != -1:
        env.Dcmm.data.mocap_pos[mocap_id] = target_pos
    mujoco.mj_forward(env.Dcmm.model, env.Dcmm.data)

    center_x, center_y = (x_range[0] + x_range[-1]) / 2, (y_range[0] + y_range[-1]) / 2
    rgb_full = get_full_scene_render(env, lookat=[center_x, center_y, 0.5], distance=4.0)

    # 6. 对比绘图并标记目标点
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- 左图：全貌图标记 ---
    axes[0].imshow(rgb_full, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]])
    # 在全貌图上点出红叉
    axes[0].scatter(target_pos[0], target_pos[1], color='red', marker='x', s=200, linewidth=3, label='Target Object')
    axes[0].set_title('Bird\'s Eye View with Target Marker')
    axes[0].set_xlabel('X (meters)')
    axes[0].set_ylabel('Y (meters)')
    axes[0].legend()

    # --- 右图：热力图标记 ---
    im = axes[1].imshow(value_map, origin='lower', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
                        cmap='hot')
    # 在热力图上点出红叉（为了在暗色背景清晰，可改用青色或白色）
    axes[1].scatter(target_pos[0], target_pos[1], color='cyan', marker='x', s=200, linewidth=3, label='Target Object')
    axes[1].set_title('Value Heatmap with Target Marker')
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Y (meters)')
    plt.colorbar(im, ax=axes[1], label='Critic Value')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("value_marker_comparison.png", dpi=300)
    print("Complete! Comparison with markers saved to value_marker_comparison.png")
    plt.show()
    env.close()


if __name__ == '__main__':
    main()