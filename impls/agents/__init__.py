from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.sharsa import SHARSAAgent
from agents.discrete_latent_sharsa import DiscreteLatentSHARSAAgent
from agents.latent_sharsa import LatentSHARSAAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    sharsa=SHARSAAgent,
    discrete_latent_sharsa=DiscreteLatentSHARSAAgent,
    latent_sharsa=LatentSHARSAAgent,
)
