import json
import time
from src.schemas import Claim
from src.agents import writer

# 1. Charger les claims originaux
with open('eval/claims_merged.json', encoding='utf-8') as f:
    claims_data = json.load(f)
claims_by_id = {c['id']: Claim(**c) for c in claims_data['claims']}

# 2. Charger les corrections
with open('eval/corrected_claims.json', encoding='utf-8') as f:
    corrections = json.load(f)['results']
corrected_ids = {c['claim_id']: c['corrected'] for c in corrections if c['status'] == 'fixed'}
# On liste les claims qui ont échoué ou qui ont été rejetés et supprimés
failed_ids = {c['claim_id'] for c in corrections if c['status'] != 'fixed'}

# 3. Créer le panier final (uniquement les valides + les corrigés)
final_claims = []
for cid, claim in claims_by_id.items():
    if cid in corrected_ids:
        final_claims.append(Claim(id=cid, text=corrected_ids[cid], paper_id=claim.paper_id))
    elif cid in failed_ids:
        continue # [ERROR] On JETTE le claim s'il n'a pas pu être corrigé
    else:
        final_claims.append(claim)

print(f'[AGENT] Agent 3 : Rédaction de la synthèse avec {len(final_claims)} faits 100% fiables...')

try:
    synthesis = writer.synthesize('evaluation of RAG systems', final_claims)
except Exception as e:
    if "429" in str(e) or "Rate limit" in str(e):
        print("\n[WARNING] Quota atteint. Attente de 7 minutes...")
        time.sleep(420)
        synthesis = writer.synthesize('evaluation of RAG systems', final_claims)
    else:
        raise e

with open('eval/results/final/synthesis_final.md', 'w', encoding='utf-8') as f:
    f.write(synthesis)

print('\n[OK] Synthèse sécurisée sauvegardée !')