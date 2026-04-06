from agents.state_trl import StateTRLAgent
from agents.state_trl import get_config as get_state_trl_config


TRLAgent = StateTRLAgent


def get_config():
    config = get_state_trl_config()
    config.agent_name = 'trl'
    return config
