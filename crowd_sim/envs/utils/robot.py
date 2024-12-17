from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, ObservableState
from crowd_sim.envs import policy
from crowd_sim.envs.utils.action import ActionVW
import numpy as np


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.rotation_constraint = getattr(config, section).rotation_constraint

    def get_distance(self, r: ObservableState):
        return ((r.px - self.px)**2 + (r.py - self.py)**2)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        filtered_obs = [r if self.get_distance(r) <= 2.5 else ObservableState(100.0, 100.0, 0.0, 0.0, r.radius) for r in ob]   
        state = JointState(self.get_full_state(), filtered_obs)
        state.w = self.w
        action, action_index = self.policy.predict(state) 
        if isinstance(action, ActionVW):
            self.w = action.w
        return action, action_index

    def get_state(self, ob):
        state = JointState(self.get_full_state(), ob)
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        return self.policy.transform(state)
