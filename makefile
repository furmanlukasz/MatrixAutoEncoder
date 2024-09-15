PROJECT_DIR := /Users/luki/Documents/GitHub/MatrixAutoEncoder
PYTHON_ENV := /Volumes/Transcend/DataAnalysis-MCI-AD-HC/venv/bin/python

train_model:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/train_model.py 

train_model_pod:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/cloud-pod/train-model-pod.py

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

generate_angular_matrix_movie:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/generate_angular_matrix_movie.py --window 2.0 --step 0.1 --dilation 1.0

clean_pyc:
	cd $(PROJECT_DIR) && find . -name "*.pyc" -delete
