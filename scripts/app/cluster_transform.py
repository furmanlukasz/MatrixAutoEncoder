# MatrixAutoEncoder/scripts/cluster_transform.py

import pandas as pd
import plotly.express as px
import os
from sklearn.cluster import DBSCAN
import streamlit as st

def cluster_transform(input_csv, eps=0.1, min_samples=3, output_dir='cluster_results'):
    """
    Perform clustering on latent space data and generate visualizations.

    Args:
        input_csv (str): Path to the input CSV file containing latent space data.
        eps (float): DBSCAN eps parameter.
        min_samples (int): DBSCAN min_samples parameter.
        output_dir (str): Directory to save results and plots.
    
    Returns:
        dict: Paths to the clustered CSV and filtered HTML plot.
    """
    print("📊 Starting cluster analysis and visualization...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 0: Load the DataFrame
    print(f"📂 Loading data from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    print(f"✅ Loaded data with {len(df)} samples.")

    # Step 1: Cluster Identification using DBSCAN
    print("🔍 Performing clustering using DBSCAN...")
    umap_embeddings = df[['UMAP1', 'UMAP2', 'UMAP3']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(umap_embeddings)
    print("✅ Clustering complete.")

    # Step 2: Cluster Filtering
    print("📝 Identifying clusters with mixed conditions...")
    cluster_condition_counts = df.groupby('Cluster')['Condition'].nunique()
    mixed_clusters = cluster_condition_counts[cluster_condition_counts > 1].index.tolist()
    print(f"❌ Clusters with mixed conditions: {mixed_clusters}")

    # Mark samples in mixed clusters for exclusion
    df['ForFurtherAnalysis'] = ~df['Cluster'].isin(mixed_clusters)
    num_excluded = len(df) - df['ForFurtherAnalysis'].sum()
    print(f"🚫 Excluding {num_excluded} samples from mixed clusters.")

    # Save the updated DataFrame
    input_filename = os.path.splitext(os.path.basename(input_csv))[0]
    df_file_updated = os.path.join(output_dir, input_filename + "_clusters.csv")
    df.to_csv(df_file_updated, index=False)
    print(f"💾 Updated DataFrame with clustering results saved to '{df_file_updated}'")

    # Step 3: Plotting

    # Plot with all data points
    print("📈 Generating plot with all data points...")
    fig_all = px.scatter_3d(
        df,
        x='UMAP1',
        y='UMAP2',
        z='UMAP3',
        color='Condition',
        hover_data=['Condition', 'BaseSubject', 'Subject', 'Index', 'Label', 'Cluster'],
        labels={'color': 'Condition'},
        title='3D UMAP Projection - All Data',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_all.update_traces(marker=dict(size=3))
    fig_all.update_layout(
        width=1000,
        height=800,
        margin=dict(r=0, b=0, l=0, t=50),
        template="plotly_white"
    )
    html_file_all = os.path.join(output_dir, input_filename + "_all_data.html")
    fig_all.write_html(html_file_all)
    print(f"💾 Plot with all data saved to '{html_file_all}'")

    # Plot with filtered data
    print("📉 Generating plot after excluding mixed clusters...")
    filtered_df = df[df['ForFurtherAnalysis']]
    fig_filtered = px.scatter_3d(
        filtered_df,
        x='UMAP1',
        y='UMAP2',
        z='UMAP3',
        color='Condition',
        hover_data=['Condition', 'BaseSubject', 'Subject', 'Index', 'Label', 'Cluster'],
        labels={'color': 'Condition'},
        title='3D UMAP Projection - Filtered Data',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_filtered.update_traces(marker=dict(size=3))
    fig_filtered.update_layout(
        width=1000,
        height=800,
        margin=dict(r=0, b=0, l=0, t=50),
        template="plotly_white"
    )
    html_file_filtered = os.path.join(output_dir, input_filename + "_filtered_data.html")
    fig_filtered.write_html(html_file_filtered)
    print(f"💾 Plot with filtered data saved to '{html_file_filtered}'")

    # Display summary information
    print("\n📊 Filtered Dataset Summary:")
    print(f"✅ Total samples after filtering: {len(filtered_df)}")
    print("\n🔢 Samples per condition:")
    print(filtered_df['Condition'].value_counts())
    print("\n👥 Unique subjects per condition:")
    print(filtered_df.groupby('Condition')['BaseSubject'].nunique())

    # Number of samples per condition in each cluster
    print("\n📁 Number of samples per condition in each filtered cluster:")
    filtered_cluster_condition_counts = filtered_df.groupby(['Cluster', 'Condition']).size().unstack(fill_value=0)
    print(filtered_cluster_condition_counts)

    print("\n🎉 Cluster analysis and visualization complete!")

    return {
        'clustered_csv': df_file_updated,
        'filtered_html': html_file_filtered
    }