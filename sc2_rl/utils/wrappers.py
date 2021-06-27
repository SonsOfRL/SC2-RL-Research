import gym


class RepeatAction(gym.Wrapper):
    
    def __init__(self, env, n_repeat, no_op=None):
        super().__init__(env)
        self.n_repeat = n_repeat
        self.no_op = no_op

    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.n_repeat):
            obs, reward, done, info = self.env.step(action)
            if self.no_op is not None:
                action = self.no_op
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, {}
