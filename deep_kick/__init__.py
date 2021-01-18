from gym.envs.registration import register

from deep_kick.env import WolfgangEnv

register(
    id='WolfgangBulletEnv-v1',
    entry_point='deep_kick:WolfgangEnv',
    kwargs={'reward_function_name': 'DeepMimicReward',
            'debug': True},
)
