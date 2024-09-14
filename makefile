PROJECT_DIR := /Users/luki/Documents/GitHub/MatrixAutoEncoder
PYTHON_ENV := /Volumes/Transcend/DataAnalysis-MCI-AD-HC/venv/bin/python

train_model:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/train_model.py

generate_rm:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/generate_rm.py

visualize_latent_space:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/visualize_latent_space.py --n_neighbors 10 --min_dist 0.25 --metric cosine

plot_html_cloud:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/plot_html_cloud.py

visualize_model_architecture:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/visualize_model_architecture.py

signal_analysis:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/evaluation_tests/signal_analysis.py