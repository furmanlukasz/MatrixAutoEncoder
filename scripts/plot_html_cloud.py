import numpy as np
import plotly.graph_objects as go

def load_data(embeddings_file, labels_file):
    embeddings = np.load(embeddings_file)
    labels = np.load(labels_file)
    return embeddings, labels

def create_plot(embeddings, labels):
    # Create color mapping
    unique_labels = np.unique(labels)
    label_to_color = {label: idx / len(unique_labels) for idx, label in enumerate(unique_labels)}
    color_values = [label_to_color[label] for label in labels]

    # Create an interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers',
        marker=dict(
            size=2.75,
            color=color_values,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f"Condition: {label}" for label in labels],
        hoverinfo='text'
    )])

    # Update layout
    fig.update_layout(
        title='3D UMAP Projection of Latent Space',
        scene=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3',
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
        ),
        width=1024,
        height=1024,
        margin=dict(r=0, b=0, l=0, t=50),
        template="plotly_dark"
    )

    return fig

def main():
    # File paths
    embeddings_file = 'results/umap_embeddings.npy'
    labels_file = 'results/labels.npy'
    output_html = 'results/latent_space_umap_from_saved_data.html'

    # Load data
    embeddings, labels = load_data(embeddings_file, labels_file)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded labels shape: {labels.shape}")

    # Create plot
    fig = create_plot(embeddings, labels)

    # Save the plot as an HTML file
    fig.write_html(output_html)
    print(f"Interactive plot saved to '{output_html}'")

    # Optionally, show the plot
    # fig.show()

if __name__ == '__main__':
    main()