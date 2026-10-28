import json

with open('data/pool_improved.json', encoding='utf-8-sig') as f:
    pool = json.load(f)['papers']

print(f"Pool : {len(pool)} articles\n")

for name in ['ARES', 'CRAG', 'MIRAGE', 'RAGAS', 'RAGBench', 'Benchmarking Large Language Models']:
    found = [p['title'] for p in pool if name.lower() in p['title'].lower()]
    print(f"{name}: {len(found)} trouve(s)")
    for t in found:
        print(f"  -> {t[:90]}")
    print()
