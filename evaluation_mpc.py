import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from crowd_nav.configs.config_vecmpc import Config
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.info import *


def evaluate(config, env, visualize=False):
    if visualize:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        plt.ion()
        plt.show()
        env.render_axis = ax
    
    test_size = config.env.test_size

    eval_episode_rewards = []

    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []
    chc_total = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0.
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    gamma = 0.99

    for k in range(test_size):
        obs = env.reset()
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0

        global_time = 0.0
        path = 0.0
        chc = 0.0

        last_pos = env.robot.get_full_state().get_position()
        last_angle = env.robot.get_full_state().get_sim_heading()

        while not done:
            if not done:
                global_time = env.global_time
            if visualize:
                env.render()

            # Obser reward and next obs
            print(env.robot)
            action = env.robot.act(obs)
            obs, rew, done, info = env.step(action)

            path = path + np.linalg.norm(env.robot.get_full_state().get_position() - last_pos)

            cur_angle = env.robot.get_full_state().get_sim_heading()
            chc = chc +  abs(cur_angle - last_angle)

            last_pos = env.robot.get_full_state().get_position()
            last_angle = cur_angle

            rewards.append(rew)

            if isinstance(info, Danger):
                too_close = too_close + 1
                min_dist.append(info.min_dist)

            episode_rew += rew

            # for info in infos:
            #     if 'episode' in info.keys():
            #         eval_episode_rewards.append(info['episode']['r'])

        eval_episode_rewards.append(episode_rew)
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        path_lengths.append(path)
        chc_total.append(chc)

        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print('Collision')
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(env.time_limit)
            print('Time out')
        else:
            raise ValueError('Invalid end signal from environment')

        cumulative_rewards.append(sum([pow(gamma, t * env.robot.time_step * env.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]))
        
        env.robot.policy.reset()


    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else env.time_limit  # env.env.time_limit

    extra_info = ''
    phase = 'test'
    print(
        '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, total reward: {:.4f}'.
            format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, avg_nav_time,
                   np.average(cumulative_rewards)))
    if phase in ['val', 'test']:
        total_time = sum(success_times + collision_times + timeout_times)
        if min_dist:
            avg_min_dist = np.average(min_dist)
        else:
            avg_min_dist = float("nan")
        print('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                     too_close * env.robot.time_step / total_time, avg_min_dist)

    print(
        '{:<5} {}has average path length: {:.2f}, CHC: {:.2f}'.
            format(phase.upper(), extra_info, sum(path_lengths) / test_size, sum(chc_total) / test_size))
    print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

def main_config(config):
    env = CrowdSim()
    env.configure(config)

    env.thisSeed = 0
    env.nenv = 1
    env.seed(0)

    evaluate(config, env, True)

def main():
    config = Config('cv')
    env = CrowdSim()
    env.configure(config)

    env.thisSeed = 0
    env.nenv = 1
    env.seed(0)

    evaluate(config, env, True)

if __name__ == "__main__":
    main()