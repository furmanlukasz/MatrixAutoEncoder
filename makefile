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

cluster_transform:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/cluster_transform.py --input_csv /results/latent_space_data_checkpoint_epoch_300_loss_0.0852_lr_0.001000_fl3.0_fh48.0_hs64_eps0.1963_20240916_170907_n10_d0.25_cosine.csv --eps 0.1 --min_samples 3 --output_dir /Users/luki/Documents/GitHub/MatrixAutoEncoder/results/cluster_results

classification:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/classification.py --input_csv $(PROJECT_DIR)/results/rqa_features.csv --output_dir $(PROJECT_DIR)/results/classification

classification_adv:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/classification_adv.py --input_csv $(PROJECT_DIR)/results/rqa_features.csv --output_dir $(PROJECT_DIR)/results/classification_adv

classification_auc:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/classification_auc.py --input_csv $(PROJECT_DIR)/results/rqa_features.csv --output_dir $(PROJECT_DIR)/results/classification_auc

c_stats:
	cd $(PROJECT_DIR) && $(PYTHON_ENV) scripts/c_stats.py --input_csv $(PROJECT_DIR)/results/rqa_features.csv --output_dir $(PROJECT_DIR)/results/stats
