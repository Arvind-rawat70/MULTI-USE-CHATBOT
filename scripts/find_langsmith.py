import os
matches=[]
for root,dirs,files in os.walk('.'):
    skip = {'.venv','.git','__pycache__','.idea','.vscode'}
    parts = set(root.split(os.sep))
    if parts & skip:
        continue
    for f in files:
        path = os.path.join(root,f)
        try:
            with open(path,encoding='utf-8',errors='ignore') as fh:
                for i,line in enumerate(fh,1):
                    if 'langsmith' in line.lower() or 'lang smith' in line.lower():
                        matches.append((path.replace('\\','/'),i,line.strip()))
        except Exception:
            pass

import json
print(json.dumps(matches,indent=2))
