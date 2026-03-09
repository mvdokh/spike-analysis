"""
Whisker Clustering Script
Clusters whisker lines by their length and visualizes them in PCA space.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go


def calculate_whisker_length(x_coords, y_coords):
    """
    Calculate the total length of a whisker from its coordinate points.
    
    Parameters:
    -----------
    x_coords : str
        String of comma-separated x coordinates
    y_coords : str
        String of comma-separated y coordinates
        
    Returns:
    --------
    float : Total length of the whisker in pixels
    """
    # Parse coordinates
    x_values = [float(x.strip()) for x in x_coords.split(',')]
    y_values = [float(y.strip()) for y in y_coords.split(',')]
    
    if len(x_values) < 2:
        return 0.0
    
    # Calculate total length as sum of distances between consecutive points
    total_length = 0.0
    for i in range(len(x_values) - 1):
        dx = x_values[i+1] - x_values[i]
        dy = y_values[i+1] - y_values[i]
        segment_length = np.sqrt(dx**2 + dy**2)
        total_length += segment_length
    
    return total_length


def prepare_whisker_features(csv_path, verbose=True):
    """
    Load whisker data and calculate features including length.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    verbose : bool
        If True, print progress information
        
    Returns:
    --------
    pd.DataFrame : DataFrame with whisker features
    """
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"Loading data from: {csv_path}")
        print(f"Total whiskers: {len(df)}")
    
    # Calculate length for each whisker
    df['Length'] = df.apply(
        lambda row: calculate_whisker_length(row['X'], row['Y']), 
        axis=1
    )
    
    # Count number of points per whisker
    df['Num_Points'] = df['X'].apply(lambda x: len(x.split(',')))
    
    # Add row index for reference
    df['Row_Index'] = range(len(df))
    
    if verbose:
        print(f"\nLength Statistics:")
        print(f"  Mean: {df['Length'].mean():.2f} pixels")
        print(f"  Std: {df['Length'].std():.2f} pixels")
        print(f"  Min: {df['Length'].min():.2f} pixels")
        print(f"  Max: {df['Length'].max():.2f} pixels")
    
    return df


def cluster_whiskers_by_length(df, n_clusters=5, random_state=42, verbose=True):
    """
    Cluster whiskers based on their length using K-means.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with whisker data including 'Length' column
    n_clusters : int
        Number of clusters (default: 5)
    random_state : int
        Random state for reproducibility (default: 42)
    verbose : bool
        If True, print cluster information
        
    Returns:
    --------
    pd.DataFrame : DataFrame with added 'Cluster' column
    """
    # Prepare feature matrix (just length for now)
    X = df[['Length']].values.copy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df['Cluster'] = cluster_labels
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Means Clustering with {n_clusters} clusters")
        print(f"{'='*60}")
        
        for cluster_id in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            print(f"\nCluster {cluster_id}:")
            print(f"  Count: {len(cluster_data)} whiskers")
            print(f"  Mean Length: {cluster_data['Length'].mean():.2f} pixels")
            print(f"  Length Range: [{cluster_data['Length'].min():.2f}, {cluster_data['Length'].max():.2f}]")
    
    return df, scaler, kmeans


def perform_pca_on_whiskers(df, n_components=2, verbose=True):
    """
    Perform PCA on whisker features for visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with whisker data
    n_components : int
        Number of PCA components (default: 2)
    verbose : bool
        If True, print PCA information
        
    Returns:
    --------
    tuple : (df with PCA columns, pca object, scaler object)
    """
    # Features for PCA: Length and Num_Points
    features = ['Length', 'Num_Points']
    X = df[features].values.copy()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Add PCA components to dataframe
    df = df.copy()
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PCA Analysis")
        print(f"{'='*60}")
        print(f"Explained variance ratio:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var*100:.2f}%")
        print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return df, pca, scaler


def create_interactive_pca_plot(df, title="Whisker Clustering in PCA Space", 
                                 color_by='Cluster', size_points=6):
    """
    Create an interactive Plotly scatter plot of whiskers in PCA space.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with PCA components and cluster labels
    title : str
        Plot title
    color_by : str
        Column name to color points by (default: 'Cluster')
    size_points : int
        Size of scatter points (default: 6)
        
    Returns:
    --------
    plotly.graph_objects.Figure : Interactive plot
    """
    # Create hover text with frame, length, and cluster info
    df['Hover_Text'] = df.apply(
        lambda row: f"Frame: {row['Frame']}<br>" +
                   f"Length: {row['Length']:.2f} px<br>" +
                   f"Cluster: {row['Cluster']}<br>" +
                   f"Points: {row['Num_Points']}<br>" +
                   f"Row Index: {row['Row_Index']}", 
        axis=1
    )
    
    # Convert cluster to string for better color mapping
    df['Cluster_Label'] = df['Cluster'].apply(lambda x: f"Cluster {x}")
    
    # Create the plot
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster_Label',
        hover_data={
            'PC1': False,
            'PC2': False,
            'Cluster_Label': False,
            'Hover_Text': True
        },
        title=title,
        labels={'PC1': 'Principal Component 1', 
                'PC2': 'Principal Component 2',
                'Cluster_Label': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Update hover template
    fig.update_traces(
        marker=dict(size=size_points, line=dict(width=0.5, color='white')),
        hovertemplate='%{customdata[0]}<extra></extra>'
    )
    
    # Update layout
    fig.update_layout(
        width=900,
        height=700,
        hovermode='closest',
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', 
                     zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                     zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    
    return fig


def analyze_and_visualize_whiskers(csv_path, n_clusters=5, random_state=42, 
                                    save_html=None, verbose=True):
    """
    Complete pipeline: load data, cluster, perform PCA, and create interactive plot.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    n_clusters : int
        Number of clusters for K-means (default: 5)
    random_state : int
        Random state for reproducibility (default: 42)
    save_html : str, optional
        Path to save the interactive plot as HTML
    verbose : bool
        If True, print detailed information
        
    Returns:
    --------
    tuple : (df, fig, pca, kmeans)
    """
    # Step 1: Load and prepare data
    df = prepare_whisker_features(csv_path, verbose=verbose)
    
    # Step 2: Cluster by length
    df, length_scaler, kmeans = cluster_whiskers_by_length(
        df, n_clusters=n_clusters, random_state=random_state, verbose=verbose
    )
    
    # Step 3: Perform PCA for visualization
    df, pca, pca_scaler = perform_pca_on_whiskers(df, n_components=2, verbose=verbose)
    
    # Step 4: Create interactive plot
    fig = create_interactive_pca_plot(df)
    
    # Save if requested
    if save_html:
        fig.write_html(save_html)
        print(f"\nInteractive plot saved to: {save_html}")
    
    # Show the plot
    fig.show()
    
    return df, fig, pca, kmeans


if __name__ == "__main__":
    # Example usage
    csv_path = "1027_lines.csv"
    
    df, fig, pca, kmeans = analyze_and_visualize_whiskers(
        csv_path, 
        n_clusters=5, 
        save_html="whisker_clusters_pca.html",
        verbose=True
    )
