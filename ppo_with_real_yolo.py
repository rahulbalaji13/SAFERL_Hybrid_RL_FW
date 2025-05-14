
import gym
from gym import spaces
import numpy as np
import cv2
from ultralytics import YOLO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Replace with your YOLOv11 model path and video file path
YOLO_MODEL_PATH = "yolov11_ppe.pt"
VIDEO_FILE_PATH = "construction_video.mp4"

# Real YOLO detection class
class RealYOLO:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            detections.append({"label": label})
        return detections

# Load frames from video file
def load_video_frames(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Custom Gym environment
class SafetyPPEEnv(gym.Env):
    def __init__(self, yolo_model, video_frames):
        super(SafetyPPEEnv, self).__init__()
        self.yolo = yolo_model
        self.frames = video_frames
        self.current_frame = 0
        self.action_space = spaces.Discrete(3)  # 0: do nothing, 1: alert, 2: log
        self.observation_space = spaces.Box(low=0, high=50, shape=(2,), dtype=np.int32)

    def reset(self):
        self.current_frame = 0
        return self._get_observation()

    def step(self, action):
        obs = self._get_observation()
        reward = self._get_reward(obs, action)
        self.current_frame += 1
        done = self.current_frame >= len(self.frames)
        return obs, reward, done, {}

    def _get_observation(self):
        frame = self.frames[self.current_frame]
        detections = self.yolo.detect(frame)
        compliant = sum(1 for d in detections if d['label'].lower() == 'compliant')
        non_compliant = sum(1 for d in detections if d['label'].lower() == 'non_compliant')
        return np.array([compliant, non_compliant], dtype=np.int32)

    def _get_reward(self, obs, action):
        _, non_compliant = obs
        if action == 0:  # Do nothing
            return 0 if non_compliant == 0 else -1
        elif action == 1:  # Alert
            return 1 if non_compliant > 0 else -1
        elif action == 2:  # Log
            return 0.5 if non_compliant > 0 else 0

# Main execution
if __name__ == "__main__":
    # Load real YOLO model and video frames
    yolo = RealYOLO(YOLO_MODEL_PATH)
    frames = load_video_frames(VIDEO_FILE_PATH)

    # Create environment
    env = SafetyPPEEnv(yolo, frames)
    check_env(env)

    # Train PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000)
    model.save("ppo_ppe_real_agent")

    # Run inference
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Obs: {obs}")
