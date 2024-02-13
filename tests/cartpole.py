#import ray
#
#ray.init()
#
#from ray import tune
#
#tune.run("PPO", 
#         config = {"env": "CartPole-v1",
#                   "evaluation_interval": 2,
#                   "evaluation_num_episodes": 20,
#                   "num_gpus": 1,
#                   },
#         local_dir = "./cartpole_v1"
#        )

from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
import ray
from pathlib import Path

ray.init()
config = PPOConfig()
# Print out some default values.
print("default config: {}".format(config))

# Update the model object.
config.model["fcnet_hiddens"]=[64, 64]


# Update the config object.
#config.training(
#    lr=tune.grid_search([0.001 ]), clip_param=0.2
#)

config.resources(num_gpus=1)

config.evaluation(
    evaluation_interval=2,
    evaluation_duration_unit="episodes",
    evaluation_duration=20)

# Set the config object's env.
config = config.environment(env="CartPole-v1")


print("updated config: {}".format(config))

stop_criteria = {
    "episode_reward_mean": 500
}

checkpoint_config = ray.train.CheckpointConfig(
    num_to_keep=5, 
    checkpoint_score_order="max", 
    checkpoint_score_attribute="episode_reward_mean",
    checkpoint_frequency=10
)

sotarage_path_full = Path("./results").resolve()
# Use to_dict() to get the old-style python config dict
# when running with tune.
tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        name="PPO_CartPole-v1",
        storage_path=sotarage_path_full,
        stop=stop_criteria,
        checkpoint_config=checkpoint_config),
    param_space=config.to_dict(),
)

tuner.fit()