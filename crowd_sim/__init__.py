from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim'
)

register(
    id='CrowdSim_mixed-v0',
    entry_point='crowd_sim.envs:CrowdSim_mixed'
)

register(
    id='CrowdSim_eth-v0',
    entry_point='crowd_sim.envs:CrowdSim_eth'
)

register(
    id='CrowdSim_wall-v0',
    entry_point='crowd_sim.envs:CrowdSim_wall'
)

register(
    id='CrowdSim_igp-v0',
    entry_point='crowd_sim.envs:CrowdSim_IGP'
)
