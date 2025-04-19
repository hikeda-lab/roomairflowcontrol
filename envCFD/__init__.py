from gym.envs.registration import register

register(
    id='FDM-v0',
    entry_point='envCFD.env:FDMEnv',
)
