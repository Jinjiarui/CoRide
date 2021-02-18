from gym.envs.registration import register
from Simulator.imitation_world.settings import TIME_LIMIT

# Gridworld
# =========================================

register(
    id='Red-Blue-Gridworld-v0',
    entry_point='imitation_world.envs:GridWorld',
    max_episode_steps=TIME_LIMIT
)
