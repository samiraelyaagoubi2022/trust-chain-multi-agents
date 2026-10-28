"""Tous les appels au LLM passent ici : retries + comptage tokens/temps."""
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError

from src.config import MODEL

load_dotenv()
_client = None
WAIT_FOR_QUOTA = False   # True : attendre le quota journalier au lieu de s'arrêter (mode --wait-quota)
STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


def get_client() -> Groq:
    """Initialise et retourne le client Groq (utilisé par chat_json et synthesize)."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _retry_after(msg: str):
    """Groq writes 'Please try again in 1m23.4s' / '850ms' : returns seconds, or None."""
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if not m:
        return None
    units = {"ms": 0.001, "h": 3600, "m": 60, "s": 1}
    total = sum(float(v) * units[u] for v, u in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)))
    return total or None


def _quota_summary(msg: str) -> str:
    """Short, readable version of Groq's long 429 message (which limit, used/requested, reset delay)."""
    m = re.search(r"on ((?:tokens|requests) per \w+.+?)\. Please try again in ([0-9hms.]+)", msg, flags=re.S)
    model = re.search(r"for model `([^`]+)`", msg)
    if not m:
        return msg[:300]
    return (f"{model.group(1) if model else 'modèle'} | {m.group(1)} | "
            f"réessayez dans {m.group(2).rstrip('.')}")


def _is_daily_quota(msg: str) -> bool:
    low = msg.lower()
    return "per day" in low or "(tpd)" in low or "(rpd)" in low


def reset_stats():
    for k in STATS:
        STATS[k] = 0.0 if k == "seconds" else 0


def chat_json(system: str, user: str, temperature: float = 0.2, retries: int = 4,
              model: str | None = None) -> dict:
    """Appelle le modèle en mode JSON et renvoie un dict.
    `model` remplace config.MODEL pour cet appel (utilisé par le vérificateur).
    Quota journalier épuisé : arrêt immédiat, ou attente automatique si WAIT_FOR_QUOTA.
    """
    client = get_client()  # Utilise la nouvelle fonction
    last = None
    attempt = 0
    while attempt < retries:
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=model or MODEL,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            STATS["calls"] += 1
            STATS["seconds"] += time.time() - t0
            STATS["prompt_tokens"] += r.usage.prompt_tokens
            STATS["completion_tokens"] += r.usage.completion_tokens
            return json.loads(r.choices[0].message.content)
        except RateLimitError as e:
            last = e
            msg = str(e)
            if _is_daily_quota(msg):
                if not WAIT_FOR_QUOTA:       # inutile de réessayer : le quota est épuisé
                    raise RuntimeError(f"Quota journalier Groq atteint : {_quota_summary(msg)}")
                wait = min((_retry_after(msg) or 300) + 1, 3600)
                print(f"[llm] quota journalier épuisé : reprise automatique dans {wait / 60:.1f} min",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue                     # une attente de quota ne consomme pas d'essai
            wait = min(_retry_after(msg) or 2 ** attempt * 3, 90)
            print(f"[llm] limite de débit, attente {wait:.0f}s (essai {attempt + 1}/{retries})",
                  file=sys.stderr, flush=True)
            time.sleep(wait + 0.5)
        except (APIError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1 + attempt)
        attempt += 1
    raise RuntimeError(f"LLM indisponible après {retries} essais : {last}")