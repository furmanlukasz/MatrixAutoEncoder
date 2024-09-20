import streamlit as st
import os
import pandas as pd
import plotly.io as pio
import streamlit.components.v1 as components

# Import the refactored functions
from cluster_transform import cluster_transform
from classification import classification
from classification_auc import classification_auc
from grid_search_heatmap import grid_search_heatmap as perform_grid_search

# State management
st.set_page_config(layout="wide") 


def main():
    st.title("Matrix AutoEncoder Analysis Dashboard")

    st.sidebar.title("Navigation")
    options = ["DataFrame Selection & Visualization", "Clustering", "Classification", "Statistics & AUC", "Grid Search Heatmap"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "DataFrame Selection & Visualization":
        dataframe_selection_and_visualization()
    elif choice == "Clustering":
        clustering()
    elif choice == "Classification":
        classification_section()
    elif choice == "Statistics & AUC":
        statistics_and_auc_computation()
    elif choice == "Grid Search Heatmap":
        grid_search_heatmap()

def dataframe_selection_and_visualization():
    st.header("Step 1: DataFrame Selection & Visualization")

    # Define paths
    dataframes_dir = "scripts/app/dataframes"  # Adjust as per your directory structure
    latents_html_dir = "scripts/app/latents_html"

    # Get list of available dataframes (CSV files)
    available_dataframes = []
    if os.path.exists(dataframes_dir):
        for f in os.listdir(dataframes_dir):
            if f.endswith(".csv"):
                available_dataframes.append(f)
    else:
        st.error(f"DataFrames directory '{dataframes_dir}' does not exist.")

    # DataFrame selection
    selected_dataframe = st.selectbox("Select a DataFrame", available_dataframes)

    if selected_dataframe:
        # Path to selected DataFrame CSV
        dataframe_csv_path = os.path.join(dataframes_dir, selected_dataframe)


        # Corresponding HTML plot
        base_filename = os.path.splitext(selected_dataframe)[0]
        html_plot_file = os.path.join(latents_html_dir, f"{base_filename}_Condition.html")
        html_plot_file = html_plot_file.replace("latent_space_data","latent_space_umap")
        
        if os.path.exists(html_plot_file):
            st.subheader("Latent Space Visualization")
            with open(html_plot_file, 'r') as f:
                html_content = f.read()
                components.html(html_content, height=600, scrolling=True)
        else:
            st.warning("Latent space HTML plot not found. Please ensure the plot is generated.")

def clustering():
    st.header("Step 2: Clustering")

    # Select DataFrame to perform clustering on
    dataframes_dir = "scripts/app/dataframes"
    available_dataframes = []
    if os.path.exists(dataframes_dir):
        for f in os.listdir(dataframes_dir):
            if f.endswith(".csv"):
                available_dataframes.append(f)
    else:
        st.error(f"DataFrames directory '{dataframes_dir}' does not exist.")

    selected_dataframe = st.selectbox("Select a DataFrame for Clustering", available_dataframes)

    if selected_dataframe:
        # Path to selected DataFrame CSV
        dataframe_csv_path = os.path.join(dataframes_dir, selected_dataframe)
        
        if not os.path.exists(dataframe_csv_path):
            st.error(f"DataFrame CSV file '{dataframe_csv_path}' does not exist.")
            return

        st.subheader("Clustering Parameters")
        eps = st.slider("DBSCAN eps", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        min_samples = st.slider("DBSCAN min_samples", min_value=1, max_value=10, value=3, step=1)

        if st.button("Apply Clustering"):
            # Define output directory based on DataFrame name
            base_filename = os.path.splitext(selected_dataframe)[0]
            output_dir = os.path.join("scripts/app/results/cluster_results", base_filename + "_clusters")
            
            # Perform clustering
            with st.spinner("Performing clustering..."):
                result = cluster_transform(
                    input_csv=dataframe_csv_path,
                    eps=eps,
                    min_samples=min_samples,
                    output_dir=output_dir
                )
            st.success("Clustering applied successfully.")

            # Display Clustered Feature Table
            clustered_csv = result.get('clustered_csv')
            if os.path.exists(clustered_csv):
                st.subheader("Clustered Feature Table")
                df_clustered = pd.read_csv(clustered_csv)
                st.dataframe(df_clustered.head())
            else:
                st.warning("Clustered CSV file not found.")

            # Display filtered latent space plot
            filtered_html = result.get('filtered_html')
            if os.path.exists(filtered_html):
                st.subheader("Filtered Latent Space Visualization")
                with open(filtered_html, 'r') as f:
                    html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
            else:
                st.warning("Filtered latent space HTML plot not found.")

def classification_section():
    st.header("Step 3: Classification")

    dataframes_dir = "scripts/app/results/cluster_results"
    available_dataframes = []
    if os.path.exists(dataframes_dir):
        for f in os.listdir(dataframes_dir):
            if os.path.isdir(os.path.join(dataframes_dir, f)):
                csv_file = os.path.join(dataframes_dir, f, f"{f}.csv")
                if os.path.exists(csv_file):
                    available_dataframes.append(f)
    else:
        st.error(f"DataFrames directory '{dataframes_dir}' does not exist.")

    selected_dataframe = st.selectbox("Select a DataFrame for Classification", available_dataframes)
    
    if selected_dataframe:
        clustered_csv_path = os.path.join(dataframes_dir, selected_dataframe, f"{selected_dataframe}.csv")
        st.write(f"Selected file: {clustered_csv_path}")

        if not os.path.exists(clustered_csv_path):
            st.error(f"Clustered CSV file '{clustered_csv_path}' does not exist. Please perform clustering first.")
            return

        # Feature selection
        all_features = ['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
        selected_features = st.multiselect("Select features for classification", all_features, default=all_features)

        # Classifier selection
        all_classifiers = ['Random Forest', 'SVM', 'XGBoost']
        selected_classifiers = st.multiselect("Select classifiers", all_classifiers, default=all_classifiers)

        output_dir = os.path.join("scripts/app/results/classification", selected_dataframe)

        if st.button("Run Classification"):
            with st.spinner("Performing classification..."):
                results = classification(
                    input_csv=clustered_csv_path,
                    output_dir=output_dir,
                    selected_features=selected_features,
                    selected_classifiers=selected_classifiers
                )
            st.success("Classification completed successfully.")

        classification_results_file = os.path.join(output_dir, "classification_results.csv")
        if os.path.exists(classification_results_file):
            st.subheader("Classification Results")
            df_class = pd.read_csv(classification_results_file)
            st.dataframe(df_class)
        else:
            st.warning("Classification results file not found. Please run classification first.")

        ridgeline_plot = os.path.join(output_dir, "ridgeline_plot.png")
        if os.path.exists(ridgeline_plot):
            st.subheader("Ridgeline Plot of Features by Condition")
            st.image(ridgeline_plot, use_column_width=True)
        else:
            st.warning("Ridgeline plot not found. Please run classification first.")

def statistics_and_auc_computation():
    st.header("Step 4: Statistics & AUC")

    dataframes_dir = "scripts/app/results/cluster_results"
    available_dataframes = []
    if os.path.exists(dataframes_dir):
        for f in os.listdir(dataframes_dir):
            if os.path.isdir(os.path.join(dataframes_dir, f)):
                csv_file = os.path.join(dataframes_dir, f, f"{f}.csv")
                if os.path.exists(csv_file):
                    available_dataframes.append(f)
    else:
        st.error(f"DataFrames directory '{dataframes_dir}' does not exist.")

    selected_dataframe = st.selectbox("Select a DataFrame for Statistics & AUC", available_dataframes)
    
    if selected_dataframe:
        clustered_csv_path = os.path.join(dataframes_dir, selected_dataframe, f"{selected_dataframe}.csv")
        st.write(f"Selected file: {clustered_csv_path}")

        if not os.path.exists(clustered_csv_path):
            st.error(f"Clustered CSV file '{clustered_csv_path}' does not exist. Please perform clustering first.")
            return

        if st.button("Run Statistics"):
            stats_output_dir = os.path.join("scripts/app/results/stats", selected_dataframe)
            with st.spinner("Computing statistics..."):
                stats_results = classification_auc(
                    input_csv=clustered_csv_path,
                    output_dir=stats_output_dir
                )
            st.success("Statistics computed successfully.")

        stats_file = os.path.join("scripts/app/results/stats", selected_dataframe, "xgboost_auc_results.csv")
        auc_img = os.path.join("scripts/app/results/stats", selected_dataframe, "roc_curve.png")
        if os.path.exists(stats_file):
            st.subheader("Statistics Results")
            df_stats = pd.read_csv(stats_file)
            st.dataframe(df_stats)
        if os.path.exists(auc_img):
            st.subheader("AUC ROC Curve")
            st.image(auc_img)
        else:
            st.warning("AUC ROC Curve file not found. Please run statistics computation first.")

def grid_search_heatmap():
    st.header("Step 5: Grid Search Heatmap")

    dataframes_dir = "scripts/app/results/cluster_results"
    available_dataframes = [f for f in os.listdir(dataframes_dir) if os.path.isdir(os.path.join(dataframes_dir, f))]

    selected_dataframe = st.selectbox("Select a DataFrame for Grid Search", available_dataframes)

    if selected_dataframe:
        input_csv = os.path.join(dataframes_dir, selected_dataframe, f"{selected_dataframe}.csv")
        output_dir = os.path.join("scripts/app/results/grid_search", selected_dataframe)

        eps_start = st.number_input("Eps start", value=0.1, step=0.05, format="%.2f")
        eps_stop = st.number_input("Eps stop", value=1.0, step=0.05, format="%.2f")
        eps_step = st.number_input("Eps step", value=0.05, step=0.01, format="%.2f")
        min_samples_start = st.number_input("Min samples start", value=2, step=1)
        min_samples_stop = st.number_input("Min samples stop", value=20, step=1)

        if st.button("Run Grid Search"):
            with st.spinner("Performing grid search... This may take a while."):
                perform_grid_search(
                    input_csv=input_csv,
                    output_dir=output_dir,
                    eps_range=(eps_start, eps_stop, eps_step),
                    min_samples_range=(min_samples_start, min_samples_stop)
                )
            st.success("Grid search completed successfully.")

        auc_heatmap = os.path.join(output_dir, "heatmap_auc.png")
        acc_heatmap = os.path.join(output_dir, "heatmap_accuracy.png")

        if os.path.exists(auc_heatmap) and os.path.exists(acc_heatmap):
            st.subheader("Grid Search Heatmaps")
            st.image(auc_heatmap, caption="AUC Heatmap", use_column_width=True)
            st.image(acc_heatmap, caption="Accuracy Heatmap", use_column_width=True)
        else:
            st.warning("Heatmap images not found. Please run the grid search first.")

if __name__ == "__main__":
    main()