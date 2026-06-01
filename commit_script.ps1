git config user.email "bot@example.com"
git config user.name "AI Assistant"

git add .gitignore docs.md
git commit -m "Initial commit: Add project documentation and gitignore"

git add chatbot-web/package.json chatbot-web/index.html chatbot-web/vite.config.js chatbot-web/tailwind.config.js chatbot-web/postcss.config.js
git commit -m "Add web application frontend skeleton"

git add chatbot-web/src/
git commit -m "Implement web application UI components"

git add chatbot-app/
git commit -m "Add mobile application frontend"

git add backend/app.py backend/requirements.txt
git commit -m "Initialize backend server setup"

git add backend/inference.py
git commit -m "Implement SLM inference logic in backend"

git add slm.ipynb
git commit -m "Add SLM notebook research"

git add prepare_dataset.py tokenize_dataset.py
git commit -m "Add data tokenization and prep scripts"

git add generate_global_suggestions.py
git commit -m "Add global dataset generation script"

git add global_suggestions_dataset.txt
git commit -m "Add massive global clinical suggestions dataset"

git add train_slm.py test_slm.py
git commit -m "Implement SLM training script"

git add start_backend.bat retrain_slm_global.bat training_results_sinhala.csv
git commit -m "Add convenience bat scripts and initial training results"

git add .
git commit -m "Final cleanup and uncommitted changes"
