from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.socialforce import SocialForce
from crowd_sim.envs.policy.ssp import SSP
from crowd_sim.envs.policy.ssp2 import SSP2
from crowd_sim.envs.policy.igp_dist import Igp_Dist


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['socialforce'] = SocialForce
policy_factory['ssp'] = SSP
policy_factory['ssp2'] = SSP2
policy_factory['igp_dist'] = Igp_Dist
policy_factory['none'] = none_policy
