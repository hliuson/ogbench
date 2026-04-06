from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gciql_nstep import NStepGCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.mc import MCAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.sharsa import SHARSAAgent
from agents.state_trl import StateTRLAgent
from agents.trl import TRLAgent
from agents.latent_trl import LatentTRLAgent
from agents.latent_sharsa import LatentSHARSAAgent
from agents.vae_trl import VaeTRLAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gciql_nstep=NStepGCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    mc=MCAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    sharsa=SHARSAAgent,
    state_trl=StateTRLAgent,
    trl=TRLAgent,
    latent_trl=LatentTRLAgent,
    latent_sharsa=LatentSHARSAAgent,
    vae_trl=VaeTRLAgent,
)
