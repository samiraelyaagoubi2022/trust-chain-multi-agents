# Contournement pour Chainlit : syncer remplace asyncio
import asyncio
import functools

def run(func):
    """Lance une fonction asynchrone dans une boucle d'événements."""
    return asyncio.run(func())

def sync(func):
    """Retourne une version synchrone d'une fonction asynchrone."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper