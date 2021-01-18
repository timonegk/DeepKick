import math

from stable_baselines3.common.callbacks import BaseCallback


class RewardLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogCallback, self).__init__(verbose)
        self.highest_episode_reward = -math.inf

    def _on_step(self) -> bool:
        # iterate through the infos of the envs
        reward_name = None
        for info in self.locals["infos"]:
            # see if episode was finished
            if "rewards" in info.keys():
                # log all rewards
                for reward_name, reward in info["rewards"].items():
                    self.logger.record(f"Rewards/{reward_name}", reward)
                    self.logger.record(f"PerStepRewards/{reward_name}",
                                       reward / info["episode"]["l"])
        # check if we have written some data
        if reward_name is not None:
            # we need to call dump explicitly otherwise it will only be written on end of epoch
            self.logger.dump(self.num_timesteps)
        return True
