import os
import json
import numpy as np
import pickle

class BaseLogger(object):
    def __init__(self, save_path, model, name):
        # print(save_path)

        self.save_path = os.path.join(save_path, name)
        os.makedirs(self.save_path, exist_ok=True)

        self.trajectory = None
        self.ballbot = []
        self.predictions = []
        self.time_log = []
        self.episode_num = 0

    def reset(self):
        if self.trajectory is not None:
            self.dump()
        self.trajectory = None
        self.ballbot = []
        self.predictions = []
        self.episode_num += 1
    
    def add_params(self, params):
        self.add_params = params
        with open(os.path.join(self.save_path, "params.json"), "w") as fout:
            json.dump(params, fout, indent=4)

    def update_trajectory(self, traj, theta, theta_dot, timestamp):
        self.trajectory = traj.copy()
        ballbot_data = np.concatenate((traj[-1, 0], np.array([theta, theta_dot, timestamp])), axis=0)
        self.ballbot.append(np.array(ballbot_data))
    
    def add_predictions(self, state, action_set, predictions, cost, goal):
        temp = dict(
            state = state,
            action_set = action_set,
            predictions = predictions,
            cost = cost,
            goal = goal
        )
        self.predictions.append(temp)
    
    def add_action(self, action):
        self.predictions[-1]["action"] = action
    
    def add_time(self, t):
        self.time_log.append(t)
    
    def dump(self):
        np.save(os.path.join(self.save_path, str(self.episode_num)+".npy"), self.trajectory)
        np.save(os.path.join(self.save_path, str(self.episode_num)+"_ballbot.npy"), self.ballbot)
        with open(os.path.join(self.save_path, str(self.episode_num)+".pkl"), "wb") as fout:
            pickle.dump(self.predictions, fout)
        with open(os.path.join(self.save_path, "loop_time.txt"), "w") as fout:
            t = np.array(self.time_log)
            json.dump(dict(mean_time=t.mean(), std_time=t.std()), fout)
        