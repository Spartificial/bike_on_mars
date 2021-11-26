def plot_moving_average(episode_reward):
    avg_reward = []
    sum_reward = 0
    span = 100
    for i in range(len(episode_reward)):
        if i >= span: sum_reward -= episode_reward[i - span]
        sum_reward += episode_reward[i]
        if i >= span: avg_reward.append(sum_reward / span)