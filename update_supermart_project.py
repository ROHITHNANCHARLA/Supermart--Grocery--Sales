# update_supermart_project.py
import os, shutil, zipfile
FILES = {
    'app.py': 'app.py',
    'templates/layout.html': 'templates/layout.html',
    'templates/index.html': 'templates/index.html',
    'templates/predict.html': 'templates/predict.html',
    'templates/predictions.html': 'templates/predictions.html',
    'templates/sql.html': 'templates/sql.html',
    'static/css/style.css': 'static/css/style.css',
    'static/img/logo2.svg': 'static/img/logo2.svg'
}
# backup
os.makedirs('backup_before_update', exist_ok=True)
for dest in FILES.values():
    if os.path.exists(dest):
        shutil.copy2(dest, os.path.join('backup_before_update', os.path.basename(dest)+'.bak'))
# copy replacements: this script assumes you already pasted the new files in same folder as script.
# If you want the script to write content automatically, let me know and I'll give a version that writes files.
# Create zip
zipname = 'Supermart_Project_v2.zip'
with zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        # skip virtual envs and backups
        if root.startswith('./backup_before_update') or root.startswith('./.venv'):
            continue
        for f in files:
            if f.endswith('.py') or f.endswith('.html') or f.endswith('.css') or f.endswith('.svg') or f.endswith('.db') or f.endswith('.pkl') or f.endswith('.csv'):
                full = os.path.join(root, f)
                arc = os.path.relpath(full, '.')
                zf.write(full, arc)
print("Backup created in backup_before_update/ and zip created:", zipname)
