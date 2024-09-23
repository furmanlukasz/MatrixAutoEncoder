import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

# Ensure you have set your OpenAI API key as an environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI Chat model
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4o-mini',
    temperature=0.3,
)


# Define the system prompt and result template
system_prompt = """
You are an intelligent assistant specialized in generating markdown reports based on EEG data analysis results. Your task is to create a comprehensive results page that includes interpretations of the analysis results, formatted appropriately for display in a Streamlit application. Use LaTeX formatting for equations by wrapping them in "$" for inline equations and "$$" on separate lines for block equations.

### Guidelines:
- Structure the report similar to the abstract page, with clear sections and subsections.
- Provide clear and concise interpretations of the analysis results.
- Use generic placeholders for condition labels that will be provided.
- Ensure the markdown is well-structured with appropriate headings and subheadings.
- Include references to visual elements like images or plots.
- Incorporate relevant information from the abstract page to provide context for the results.
"""

result_template = """
# Results

## 1. Introduction
Briefly introduce the context of the study, mentioning the use of autoencoders for EEG data analysis and the challenges addressed.

## 2. Methods
Summarize the key methods used in the study, including:
- Autoencoder architecture
- Recurrence Quantification Analysis (RQA)
- Classification techniques
- Clustering and visualization approaches

## 3. Classification Performance

{classification_performance}

Interpret the classification results, discussing the performance of different models and their implications.

## 4. Grid Search Analysis

{grid_search_summary}

Discuss the results of the grid search, highlighting the optimal parameters found and their impact on model performance.

## 5. XGBoost Performance

The XGBoost classifier achieved an AUC of {xgboost_auc}, indicating its effectiveness in distinguishing between different cognitive states.

## 6. Visualization Insights

Describe the insights gained from the following visualizations:
- Ridgeline Plot: Discuss the distribution of features across conditions.
- AUC Heatmap: Interpret the patterns observed in the grid search results.
- Accuracy Heatmap: Compare with the AUC heatmap and discuss any differences.
- ROC Curve: Analyze the model's discriminative ability across different thresholds.

## 7. Discussion
Synthesize the findings, discussing how they address the initial challenges mentioned in the introduction. Consider the implications of these results for resting-state EEG analysis and potential future directions.

Add at the end: 


This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

This article is developed as part of a methodology concept by Łukasz Furman, utilizing the Data Lab LLM Agent process. It integrates insights and knowledge from various sources, including O1 Preview, LLAMA3, and Cloude Sonet 3.5. Additionally, it incorporates generated text formatting and structuring processes to enhance clarity and coherence. ✨

"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", result_template),
])

# Define the LCEL chain
chain = (
    RunnablePassthrough()
    | prompt
    | llm
    | StrOutputParser()
)

def generate_markdown(classification_perf, grid_search_summary, xgboost_auc):
    """
    Generates a markdown report based on the provided analysis results.

    Parameters:
    - classification_perf (str): Markdown-formatted classification performance.
    - grid_search_summary (str): Markdown-formatted grid search summary.
    - xgboost_auc (float): AUC score for XGBoost classifier.

    Returns:
    - str: Generated markdown content.
    """
    inputs = {
        "classification_performance": classification_perf,
        "grid_search_summary": grid_search_summary,
        "xgboost_auc": xgboost_auc,
    }
    return chain.invoke(inputs)

def save_markdown(content, file_path):
    """
    Saves the generated markdown content to a file.

    Parameters:
    - content (str): Markdown content to save.
    - file_path (str): Path to the markdown file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(content)
    print(f"Markdown report saved to {file_path}")