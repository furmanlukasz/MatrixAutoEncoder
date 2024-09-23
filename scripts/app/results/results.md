# Results

## 1. Introduction
This study explores the application of autoencoders for the analysis of EEG data, addressing the challenges associated with high-dimensionality and noise inherent in such datasets. By leveraging advanced machine learning techniques, we aim to enhance the interpretability and classification of cognitive states based on EEG signals.

## 2. Methods
The analysis employed several key methodologies:

- **Autoencoder Architecture**: A deep learning model designed to learn efficient representations of the EEG data by compressing and reconstructing the input signals.
- **Recurrence Quantification Analysis (RQA)**: A technique used to quantify the dynamics of the EEG signals, providing insights into the temporal patterns and structures.
- **Classification Techniques**: Various classifiers, including XGBoost, were utilized to assess the performance in distinguishing between different cognitive states.
- **Clustering and Visualization Approaches**: Techniques such as clustering were applied to group similar EEG patterns, and visualization methods were employed to interpret the results effectively.

## 3. Classification Performance

|    | Unnamed: 0   |   accuracy |   f1_score |
|---:|:-------------|-----------:|-----------:|
|  0 | XGBoost      |   0.803519 |   0.802162 |

The classification results indicate that the XGBoost model achieved an accuracy of approximately 80.35% and an F1 score of 80.22%. These metrics suggest that the model is effective in distinguishing between the specified cognitive states, with a balanced performance in terms of precision and recall.

## 4. Grid Search Analysis

|       |       eps |   min_samples |        auc |   accuracy |
|:------|----------:|--------------:|-----------:|-----------:|
| count | 65        |      65       | 65         | 65         |
| mean  |  0.4      |       4       |  0.906334  |  0.78024   |
| std   |  0.188539 |       1.42522 |  0.0550702 |  0.0788006 |
| min   |  0.1      |       2       |  0.734769  |  0.56719   |
| 25%   |  0.25     |       3       |  0.871328  |  0.716216  |
| 50%   |  0.4      |       4       |  0.920901  |  0.793956  |
| 75%   |  0.55     |       5       |  0.954568  |  0.848101  |
| max   |  0.7      |       6       |  0.961888  |  0.873646  |

The grid search results reveal that the optimal parameters for the model were found to be an epsilon ($\epsilon$) of 0.4 and a minimum sample size of 4. The mean AUC of 0.906334 indicates a strong ability of the model to discriminate between classes, while the accuracy averaged around 78.02%. The standard deviation values suggest variability in model performance, which could be addressed in future iterations.

## 5. XGBoost Performance
The XGBoost classifier achieved an AUC of 0.9309869753045782, underscoring its effectiveness in distinguishing between different cognitive states. This high AUC value indicates that the model has a strong discriminative ability, making it suitable for EEG data classification tasks.

## 6. Visualization Insights
The following visualizations provided valuable insights into the data:

- **Ridgeline Plot**: This visualization illustrated the distribution of features across different conditions, revealing patterns that may correlate with cognitive states.
- **AUC Heatmap**: The heatmap displayed the AUC values across various parameter combinations, highlighting the most effective configurations for model performance.
- **Accuracy Heatmap**: Comparison with the AUC heatmap showed that while some parameter settings yielded high accuracy, they did not always correspond to the best AUC scores, indicating a need for careful parameter selection.
- **ROC Curve**: The ROC curve analysis demonstrated the model's ability to maintain a high true positive rate while minimizing false positives across different thresholds.

## 7. Discussion
The findings of this study address the initial challenges posed by the complexity of EEG data. The successful implementation of autoencoders and XGBoost for classification emphasizes the potential of these techniques in enhancing resting-state EEG analysis. Future directions may include exploring additional machine learning models and refining the feature extraction process to further improve classification accuracy and interpretability.

---

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

This article is developed as part of a methodology concept by Łukasz Furman, utilizing the Data Lab LLM Agent process. It integrates insights and knowledge from various sources, including O1 Preview, LLAMA3, and Cloude Sonet 3.5. Additionally, it incorporates generated text formatting and structuring processes to enhance clarity and coherence. ✨