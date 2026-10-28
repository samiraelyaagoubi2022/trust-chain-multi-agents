# Supprime seulement les caches et les logs. Ne touche JAMAIS aux annotations, au corpus figé ni aux résultats.
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Where-Object { $_.FullName -notmatch '\\.venv\\' } | Remove-Item -Recurse -Force
Write-Host "Caches supprimés. Données, annotations et résultats intacts."
