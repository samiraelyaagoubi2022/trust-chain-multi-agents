
# 🤖 Trust Chain Multi-Agents : Veille Scientifique et Fact-Checking Automatisé

Un système multi-agents autonome conçu pour résoudre un problème majeur des grands modèles de langage (LLM) : **les hallucinations**. Le système recherche des articles scientifiques, en extrait les affirmations clés (claims), les vérifie, les corrige automatiquement, et génère un **état de l'art garanti sans hallucination**.

---

## 🏗️ Architecture : La "Trust Chain" en 3 Agents

Le système repose sur une chaîne de confiance orchestrée par `src/pipeline.py`, divisée en 3 agents spécialisés :

1. **🕵️ Agent 1 : Le Chercheur (`researcher.py`)**
   * **Rôle** : Trouver la matière première.
   * **Actions** : Interroge arXiv et OpenAlex, filtre les articles hors-sujet, utilise des embeddings (`fastembed`) et un classement hybride (BM25 + RRF).

2. **✍️ Agent 2 : Le Rédacteur (`writer.py`)**
   * **Rôle** : Extraire et Synthétiser.
   * **Actions** : Lit les abstracts et en extrait des affirmations courtes et factuelles (claims). En fin de chaîne, il rédige la synthèse finale en se basant **uniquement** sur les faits validés.

3. **🛡️ Agent 3 : L'Auditeur (`verifier.py` + `corrector.py`)**
   * **Rôle** : Garantir la véracité (Fact-Checking & Auto-Correction).
   * **Actions** : Vérifie chaque claim mot à mot par rapport à l'abstract source. S'il détecte une erreur (chiffre inversé, mauvaise attribution), il demande sa réécriture au Rédacteur, puis le re-vérifie.

---

## 📊 Métriques de Performance (End-to-End)

Le système a été rigoureusement évalué sur un corpus de 84 affirmations (sujet : *RAG evaluation*).

| Métrique Globale | Définition | Résultat |
| :--- | :--- | :--- |
| **Taux de Rétention (Yield)** | Faits conservés / Faits extraits au départ | **94,0 %** (79/84) |
| **Taux d'Auto-Guérison** | Erreurs corrigées automatiquement / Erreurs détectées | **77,3 %** (17/22) |
| **Garantie Zéro Hallucination** | Le rapport final ne contient que des faits audités | **100 %** |

**Évaluation de l'Auditeur (Agent 3) :**
*   **Accuracy** : 95 %
*   **Kappa de Cohen** : 0.88 (Accord excellent avec l'annotation humaine)
*   **Fiabilité inter-annotateur** : 0.77 (testé sur 15 claims en aveugle)

**Généralisation (Test hors-domaine) :**
Le système est agnostique du sujet. Testé sur *Quantum Machine Learning* et *Climate Modeling*, il conserve un taux d'étayage supérieur à **93 %**.

---

## 🚀 Installation & Lancement

### 1. Prérequis
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Créez un fichier `.env` à la racine et ajoutez votre clé API Groq :
```env
GROQ_API_KEY=votre_cle_groq_ici
```

### 2. Lancer l'interface web (Démo interactive)
L'interface interactive (Chainlit) permet de voir les agents travailler en temps réel :
```bash
python -m chainlit run app.py -w
```
*Ouvrez votre navigateur sur `http://localhost:8000`, entrez un sujet de recherche, et observez la magie opérer.*

### 3. Lancer le pipeline en ligne de commande (CLI)
Pour générer un rapport complet sans interface web :
```bash
python -m src.pipeline --claims eval/claims_merged.json --mode verified --prompt v3 --note eval/results/final/synthesis_final.md
```

---

## 📂 Structure du Projet

```text
veille-agents/
├── src/
│   ├── agents/
│   │   ├── researcher.py    # Agent 1
│   │   ├── writer.py        # Agent 2 (Extraction + Synthèse)
│   │   └── verifier.py      # Agent 3 (Vérification)
│   ├── pipeline.py          # Orchestrateur
│   ├── llm.py               # Gestion API et retry/quota
│   └── schemas.py           # Contrats de données (Paper, Claim, Verdict)
├── scripts/
│   ├── corrector.py         # Agent 3 (Auto-Correction)
│   ├── evaluate_verifier.py # Script d'évaluation scientifique
│   └── measure_recall.py    # Mesure du rappel
├── app.py                   # Interface de démonstration Chainlit
├── syncer.py                # Correctif de compatibilité Windows/Python 3.13
├── compute_metrics.py       # Calcul des métriques End-to-End
├── eval/                    # Données d'évaluation et résultats finaux
└── README.md
```

## ⚠️ Limites connues
*   **Abstracts uniquement** : Le système ne lit que les résumés des articles, pas les PDF complets.
*   **Quotas API** : Les modèles gratuits (Groq) imposent des limites. Le système gère cela avec une attente automatique.
*   **Variance** : Les résultats peuvent varier légèrement entre deux exécutions en raison de la nature stochastique des LLM.
