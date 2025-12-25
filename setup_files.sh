#!/bin/bash

# Create .gitignore
echo "Creating .gitignore..."
curl -o .gitignore https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore
echo "" >> .gitignore
echo "# Project specific" >> .gitignore
echo "config.py" >> .gitignore
echo "reports/" >> .gitignore
echo "py2txt/" >> .gitignore

# Create config.template.py
echo "Creating config.template.py..."
cp config.py config.template.py
# You'll need to manually edit config.template.py to replace your real keys with placeholders

echo "Done! Now edit config.template.py to replace real API keys with 'YOUR_KEY_HERE'"