"""Objets échangés entre agents (le contrat du système)."""
from pydantic import BaseModel


class Paper(BaseModel):
    id: str                 # P1, P2... attribué au chargement
    title: str
    abstract: str
    url: str = ""
    year: int | None = None
    source: str = ""


class Claim(BaseModel):
    id: str                 # C1, C2...
    text: str               # l'affirmation (une seule idée)
    paper_id: str           # article cité


class Verdict(BaseModel):   # utilisé au J3
    claim_id: str
    label: str              # "etayee" | "non_etayee" | "partielle"
    evidence: str = ""      # phrase de l'abstract qui justifie
    reason: str = ""        # explication courte du vérificateur
    evidence_in_source: bool = True   # la citation existe-t-elle vraiment dans l'abstract ?