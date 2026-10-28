import chainlit as cl
import asyncio
import json
import os
import subprocess

from src.schemas import Paper, Claim
from src.agents import writer, verifier

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Bienvenue sur le systeme multi-agents de veille scientifique.\n\n"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    topic = message.content
    await cl.Message(content=f" Lancement du pipeline en direct pour : {topic}").send()

    # 1. AGENT 1 : CHERCHEUR (Appel réel au script researcher.py)
    async with cl.Step(name="Agent 1 : Chercheur (arXiv + OpenAlex)") as step1:
        step1.output = "Recherche des articles en direct sur les bases de données..."
        
        # On lance le script researcher.py en arrière-plan
        def run_researcher():
            # On utilise le tag "demo" pour ne pas écraser tes fichiers finaux
            cmd = f'python src/agents/researcher.py "{topic}" --tag demo --no-judge' 
            subprocess.run(cmd, shell=True, check=True)
            
        try:
            await cl.make_async(run_researcher)()
            # On charge les articles qui viennent d'être trouvés
            papers = writer.load_papers("data/papers_demo.json")
            step1.output = f" {len(papers)} articles scientifiques pertinents retenus."
        except Exception as e:
            step1.output = f" Echec de la recherche : {e}"
            return

    # 2. AGENT 3 : EXTRACTION
    async with cl.Step(name="Agent 3 : Redacteur (Extraction)") as step3a:
        step3a.output = f"Extraction des affirmations depuis les {len(papers)} abstracts..."
        try:
            claims = await cl.make_async(writer.write_note)(topic, papers, per_paper=2)
            step3a.output = f"{len(claims)} affirmations (claims) extraites."
        except Exception as e:
            step3a.output = f" Echec de l'extraction (Quota ?) : {e}"
            return

    # 3. AGENT 2 : AUDITEUR (Appel réel au vérificateur)
    async with cl.Step(name="Agent 2 : Auditeur (Verification & Auto-Correction)") as step2:
        step2.output = "Verification croisee des affirmations en direct..."
        
        def run_audit():
            # On sauvegarde les claims temporairement
            with open("eval/claims_demo.json", "w", encoding="utf-8") as f:
                json.dump({"topic": topic, "papers": [p.model_dump() for p in papers], 
                           "claims": [c.model_dump() for c in claims]}, f)
            
            # On lance la vérification et correction
            cmd = 'python -m src.pipeline --claims eval/claims_demo.json --mode verified --prompt v3 --out eval/results/pipeline_demo.json'
            subprocess.run(cmd, shell=True, check=True)
            
            # On récupère le résultat
            with open("eval/results/pipeline_demo.json", "r", encoding="utf-8") as f:
                return json.load(f)
                
        try:
            res = await cl.make_async(run_audit)()
            step2.output = f" {len(res['removed'])} claims supprimes, {len(res['corrected'])} corriges."
            final_claims = [Claim(**c) for c in res["claims_final"]]
        except Exception as e:
            step2.output = f" Echec de l'audit : {e}"
            return

    # 4. AGENT 3 : SYNTHESE (Génération dynamique du rapport)
    async with cl.Step(name="Agent 3 : Redacteur (Synthese Finale)") as step3b:
        step3b.output = "Redaction du rapport en direct base sur les faits verifies..."
        
        try:
            synthesis_text = await cl.make_async(writer.synthesize)(topic, final_claims)
            step3b.output = "Rapport d'etat de l'art genere en direct."
        except Exception as e:
            synthesis_text = f"Erreur lors de la synthese : {e}"
            step3b.output = "Echec de la synthese."

    # --- AFFICHAGE DU RÉSULTAT FINAL ---
    await cl.Message(
        content=f"###Synthese Scientifique : {topic}\n\n{synthesis_text}"
    ).send()