import os

# Dictionnaire de remplacement : on associe un emoji à une balise professionnelle
REPLACEMENTS = {
    "[OK]": "[OK]",
    "[ERROR]": "[ERROR]",
    "[AGENT]": "[AGENT]",
    "[WARNING]": "[WARNING]",
    "[RUN]": "[RUN]",
    "[METRICS]": "[METRICS]",
    "[AUDIT]": "[AUDIT]",
    "[FIX]": "[FIX]",
    "[TARGET]": "[TARGET]",
    "[INFO]": "[INFO]",
    "[REPORT]": "[REPORT]",
    "[DONE]": "[DONE]",
    "[CLEANUP]": "[CLEANUP]",
    "[WAIT]": "[WAIT]",
    "[SUCCESS]": "[SUCCESS]"
}

def replace_emojis(filepath):
    """Remplace les emojis par des balises professionnelles dans un fichier Python."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return # Ignore les fichiers non-texte

    original_content = content

    # 1. Remplacer les emojis par leurs balises
    for emoji, tag in REPLACEMENTS.items():
        content = content.replace(emoji, tag)

    # 2. Nettoyer les espaces doubles créés par le remplacement (ex: "  [OK]" -> " [OK]")
    content = content.replace("[OK] ", "[OK] ")
    content = content.replace("[ERROR] ", "[ERROR] ")
    content = content.replace("[AGENT] ", "[AGENT] ")
    # (Tu peux en ajouter si besoin, mais en général les remplacements se font bien)

    # 3. Si le fichier a changé, on le sauvegarde
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] Fichier professionnalisé : {filepath}")

def main():
    print("Demarrage du remplacement des emojis par des balises professionnelles...\n")
    
    ignored_dirs = {'.venv', '.git', '__pycache__', 'node_modules'}
    count = 0

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                replace_emojis(filepath)
                count += 1
                
    print(f"\nTermine ! {count} fichiers Python ont ete analyses.")

if __name__ == "__main__":
    main()