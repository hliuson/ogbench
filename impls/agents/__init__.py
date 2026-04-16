from agents.cf_trl import CFTRLAgent
from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gciql_nstep import NStepGCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.latent_sharsa import LatentSHARSAAgent
from agents.mc import MCAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.sharsa import SHARSAAgent
from agents.trl import TRLAgent
from agents.trl_original import TRLOriginalAgent

agents = dict(
    cf_trl=CFTRLAgent,
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gciql_nstep=NStepGCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    latent_sharsa=LatentSHARSAAgent,
    mc=MCAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    sharsa=SHARSAAgent,
    trl=TRLAgent,
    trl_original=TRLOriginalAgent,
)
