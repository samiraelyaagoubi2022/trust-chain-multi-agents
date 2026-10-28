"""Paramètres centralisés."""
MODEL = "openai/gpt-oss-120b"   # modèle Groq ; changez-le ici uniquement
PAPERS_PATH = "data/papers.json"
CLAIMS_PATH = "data/claims.json"
MAX_PAPERS = 10          # nombre d'articles donnés au rédacteur
MIN_RELEVANCE = 2        # on ne garde que les articles notés >= 2 par le chercheur (juge 0-3)

# Vérificateur : volontairement un AUTRE modèle que le rédacteur (erreurs moins corrélées)
VERIFIER_MODEL = "openai/gpt-oss-120b"
VERDICTS_PATH = "data/verdicts.json"

# Boucle de correction (J4) : nombre maximal de tours rédacteur -> vérificateur
MAX_ROUNDS = 2
NOTE_PATH = "data/note.md"