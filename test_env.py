import gymnasium as gym
import numpy as np
import cv2 as cv
from gym_dcmm.envs.stage1.DcmmVecEnvStage1 import DcmmVecEnvStage1
import configs.env.DcmmCfg as DcmmCfg

def test_env():
    print("Initializing DcmmVecEnvStage1...")
    env = DcmmVecEnvStage1(
        task='Tracking', 
        object_name='object', 
        render_per_step=False, 
        print_reward=True, 
        print_info=False, 
        print_contacts=False, 
        print_ctrl=False, 
        print_obs=False, 
        camera_name=["wrist"],
        render_mode="depth_array", 
        imshow_cam=False,      # 显示机械臂cv2的窗口
        viewer=True,         # 可选：MuJoCo 3D viewer
        object_eval=False,
        env_time=1000.0,      # ←← 设为很大，避免超时终止
        steps_per_policy=20,
        img_size=(112, 112)
    )
    print("Environment initialized successfully.")

    obs, info = env.reset()
    print("Reset successful.")

    # 创建 dummy action（可替换为键盘控制）
    action = {
        'base': np.zeros(2),
        'arm': np.zeros(6),
        'hand': np.zeros(12)
    }

    print("Starting interactive loop. Close the OpenCV window or press Ctrl+C to exit.")
    
    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 如果 episode 结束，自动 reset（可选）
            if terminated or truncated:
                print(f"Episode ended (terminated={terminated}, truncated={truncated}). Resetting...")
                obs, info = env.reset()

            # 检查 OpenCV 窗口是否被关闭
            # 注意：imshow_cam=True 时，RenderManager 会调用 cv.imshow()
            # 我们通过检测是否有窗口存在并响应按键来判断
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                print("Exit requested via keypress.")
                break

            # 可选：检测窗口是否被手动关闭（部分系统有效）
            # OpenCV 本身不直接提供窗口关闭事件，但可通过以下方式间接判断：
            # 如果你只开一个窗口，可以尝试：
            # if cv.getWindowProperty("wrist_depth", cv.WND_PROP_VISIBLE) < 1:
            #     break
            # 但更可靠的方式是依赖按键退出。

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        env.close()
        cv.destroyAllWindows()  # 确保关闭所有 OpenCV 窗口
        print("Environment closed.")

if __name__ == "__main__":
    test_env()
