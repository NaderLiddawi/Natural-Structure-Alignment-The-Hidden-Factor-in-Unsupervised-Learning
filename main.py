"""
CS7641 Assignment 3 - Unsupervised Learning and Dimensionality Reduction
========================================================================

This implementation covers all required components:
1. Clustering: K-Means and Expectation Maximization (EM)
2. Dimensionality Reduction: PCA, ICA, Random Projection
3. Analysis of clusters and dimensionality reduction effects
4. Neural Network baseline and comparison with reduced dimensions and cluster features

Structure follows assignment requirements with clear sections for:
- Applying clustering to original datasets
- Applying dimensionality reduction to datasets
- Re-applying clustering to reduced datasets
- Running neural networks with reduced dimensions
- Running neural networks with cluster features

Author: Nader Liddawi
Date: Spring 2025
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             mean_squared_error, accuracy_score, f1_score, roc_auc_score)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from numpy.linalg import pinv
from kneed import KneeLocator
from tabulate import tabulate



# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==========================================================================
# PRELIMINARY
# ==========================================================================

# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Create subdirectories for better organization
for subdir in ['clustering', 'dim_reduction', 'combined', 'neural_net', 'extra']:
    if not os.path.exists(f'results/{subdir}'):
        os.makedirs(f'results/{subdir}')

# Configure plot style
# plt.style.use('seaborn-v0_8-whitegrid')
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'sans-serif',
    'axes.edgecolor': 'white',  # White borders for visibility
    'xtick.color': 'white',     # White x-axis ticks
    'ytick.color': 'white',     # White y-axis ticks
    'axes.labelcolor': 'white', # White axis labels
    'axes.titlecolor': 'white'  # White title color
})

# Neural network's best parameter configuration from Assignment 1
NN_PARAMS = {
    'hidden_layer_sizes': (100,),
    'activation': 'logistic',
    'alpha': 0.0002463768595899747,
    'learning_rate_init': 0.00582938454299474,
    'max_iter': 500,
    'early_stopping': True,
    'random_state': RANDOM_SEED
}


def create_metric_plot(x_values, y_values, title, xlabel, ylabel, best_value=None,
                       best_value_label=None, filename=None, figsize=(10, 6), color='blue'):
    """
    Generic function to create metric plots with optional best value marking.
    """
    plt.figure(figsize=figsize)
    plt.plot(x_values, y_values, 'o-', color=color)

    if best_value is not None:
        plt.axvline(x=best_value, linestyle='--', color='red', alpha=0.7)
        if best_value_label:
            # Position the text at an appropriate y-value
            y_position = min(y_values) + 0.1 * (max(y_values) - min(y_values))
            plt.text(best_value + 0.5, y_position, best_value_label, color='red')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ==========================================================================
# DATA LOADING AND PREPROCESSING
# ==========================================================================

def label_encode_categorical(df):
    """
    Label-encodes all categorical (object) columns in the DataFrame.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def load_marketing_data(filepath='marketing_campaign.csv'):
    """
    Loads and preprocesses the marketing campaign dataset.
    """
    print(f"Loading marketing data from {filepath}...")
    try:
        df = pd.read_csv(filepath, delimiter='\t', encoding='latin1')
        # Drop rows with missing values
        df = df.dropna()

        # Prepare target variable
        y = df['Response'].astype(int)

        # Prepare features (drop ID, date and target)
        X = df.drop(columns=['ID', 'Dt_Customer', 'Response'], errors='ignore')

        # Handle categorical variables
        X = label_encode_categorical(X)

        print(f"Marketing data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"Error loading marketing data: {e}")
        raise


def load_spotify_data(filepath='spotify-2023.csv'):
    """
    Loads and preprocesses the Spotify 2023 dataset.
    """
    print(f"Loading Spotify data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='latin1')

        # Convert streams to numeric
        df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

        # Drop rows with missing values
        df = df.dropna(subset=['streams'])

        # Create binary target: high/low streams relative to median
        median_streams = df['streams'].median()
        df['streams_high'] = (df['streams'] > median_streams).astype(int)

        # Prepare features (drop non-feature columns)
        X = df.drop(columns=[
            'streams', 'streams_high', 'track_name',
            'artist(s)_name'
        ], errors='ignore')

        # Handle categorical variables
        X = label_encode_categorical(X)

        # Prepare target
        y = df['streams_high']

        print(f"Spotify data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"Error loading Spotify data: {e}")
        raise


def prepare_data(X, y, test_size=0.3):
    """
    Splits data into train and test sets, and applies standardization.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


# ==========================================================================
# CLUSTERING ALGORITHMS
# ==========================================================================

def find_optimal_kmeans(X, k_range=range(2, 21), n_init_values=[5, 10, 20], verbose=True):
    """
    Determines the optimal number of clusters for K-Means using multiple metrics,
    while also testing different n_init values for stability.
    """
    # Dictionary to store results for each n_init value
    n_init_results = {}
    best_n_init = None
    best_avg_silhouette = -1

    if verbose:
        print_section_header("Finding optimal K and n_init for K-Means", level=2)

    # Test different n_init values
    for n_init in n_init_values:
        # Initialize accumulator lists
        silhouette_scores = []
        inertia_values = []
        ch_scores = []
        db_scores = []
        execution_times = []

        if verbose:
            print(f"\nTesting n_init={n_init}")
            table_data = []

        # Evaluate each K value with this n_init
        for k in k_range:
            start_time = time.time()

            # Create and fit the model
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_SEED)
            labels = kmeans.fit_predict(X)

            # Compute performance metrics
            inertia = kmeans.inertia_
            sil_score = silhouette_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            db_score = davies_bouldin_score(X, labels)

            # Record execution time
            exec_time = time.time() - start_time

            # Store values
            silhouette_scores.append(sil_score)
            inertia_values.append(inertia)
            ch_scores.append(ch_score)
            db_scores.append(db_score)
            execution_times.append(exec_time)

            if verbose:
                table_data.append([
                    k,
                    f"{sil_score:.4f}",
                    f"{inertia:.2f}",
                    f"{ch_score:.2f}",
                    f"{db_score:.4f}",
                    f"{exec_time:.2f}s"
                ])

        # Determine optimal K for each metric with this n_init
        best_k_silhouette = k_range[np.argmax(silhouette_scores)]
        best_k_ch = k_range[np.argmax(ch_scores)]
        best_k_db = k_range[np.argmin(db_scores)]

        # Use the elbow method for inertia
        kl = KneeLocator(list(k_range), inertia_values, curve='convex', direction='decreasing')
        best_k_elbow = kl.elbow if kl.elbow is not None else k_range[1]

        # Store results for this n_init
        n_init_results[n_init] = {
            'k_range': list(k_range),
            'silhouette_scores': silhouette_scores,
            'inertia_values': inertia_values,
            'ch_scores': ch_scores,
            'db_scores': db_scores,
            'execution_times': execution_times,
            'best_k_silhouette': best_k_silhouette,
            'best_k_ch': best_k_ch,
            'best_k_db': best_k_db,
            'best_k_elbow': best_k_elbow,
            'avg_silhouette': np.mean(silhouette_scores)
        }

        # Check if this n_init is better overall
        if np.mean(silhouette_scores) > best_avg_silhouette:
            best_avg_silhouette = np.mean(silhouette_scores)
            best_n_init = n_init

        if verbose:
            headers = ["K", "Silhouette", "Inertia", "CH", "DB", "Time"]
            print_table(table_data, headers, f"K-Means Evaluation (n_init={n_init})")

            # Print best K values for this n_init
            best_k_table = [
                ["Silhouette", best_k_silhouette],
                ["Calinski-Harabasz", best_k_ch],
                ["Davies-Bouldin", best_k_db],
                ["Elbow Method", best_k_elbow]
            ]
            print_table(best_k_table, ["Metric", "Best K"], f"Optimal K Values (n_init={n_init})")

    # Select results from the best n_init
    results = n_init_results[best_n_init]
    results['best_n_init'] = best_n_init
    results['n_init_comparison'] = {n: res['avg_silhouette'] for n, res in n_init_results.items()}

    if verbose:
        print(f"\nBest n_init value: {best_n_init} (avg. silhouette: {best_avg_silhouette:.4f})")

        # Print n_init comparison
        n_init_table = [[n, f"{res['avg_silhouette']:.4f}"] for n, res in n_init_results.items()]
        print_table(n_init_table, ["n_init", "Avg. Silhouette"], "n_init Parameter Comparison")




    return results


def find_optimal_em(X, k_range=range(2, 21), cov_types=['full', 'tied', 'diag'], n_init_values=[1, 2, 5], verbose=True):
    """
    Determines the optimal number of components, covariance type, and n_init for EM (GMM).

    Parameters:
    -----------
    X : array-like
        Input data to cluster
    k_range : range, default=range(2, 21)
        Range of cluster numbers to test
    cov_types : list, default=['full', 'tied', 'diag']
        Covariance types to test
    n_init_values : list, default=[1, 2, 5]
        Number of initialization values to test
    verbose : bool, default=True
        Whether to print detailed results

    Returns:
    --------
    dict
        Dictionary containing results and optimal parameters
    """
    # Dictionary to store results for each covariance type and n_init
    all_results = {}
    best_cov_type = None
    best_n_init = None
    best_bic = float('inf')

    if verbose:
        print_section_header("Finding optimal K, covariance type, and n_init for EM", level=2)

    # Test different covariance types and n_init values
    for cov_type in cov_types:
        for n_init in n_init_values:
            # Initialize accumulator lists
            bic_values = []
            aic_values = []
            ll_values = []
            execution_times = []

            if verbose:
                print(f"\nTesting covariance_type='{cov_type}', n_init={n_init}")
                table_data = []

            # Test different K values with this covariance type and n_init
            for k in k_range:
                start_time = time.time()

                # Create and fit the model
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    n_init=n_init,
                    random_state=RANDOM_SEED
                )

                try:
                    gmm.fit(X)

                    # Compute metrics
                    bic = gmm.bic(X)
                    aic = gmm.aic(X)
                    log_likelihood = gmm.score(X) * X.shape[0]  # per-sample log likelihood * n_samples

                    # Record execution time
                    exec_time = time.time() - start_time

                    # Store values
                    bic_values.append(bic)
                    aic_values.append(aic)
                    ll_values.append(log_likelihood)
                    execution_times.append(exec_time)

                    if verbose:
                        table_data.append([
                            k,
                            f"{bic:.2f}",
                            f"{aic:.2f}",
                            f"{log_likelihood:.2f}",
                            f"{exec_time:.2f}s"
                        ])
                except:
                    # Handle potential convergence failures
                    if verbose:
                        print(f"  Warning: Failed to converge for K={k}, covariance={cov_type}, n_init={n_init}")
                    bic_values.append(float('inf'))
                    aic_values.append(float('inf'))
                    ll_values.append(float('-inf'))
                    execution_times.append(float('inf'))

            # Find optimal K for this covariance type and n_init
            if any(bic < float('inf') for bic in bic_values):
                best_k_bic_idx = np.argmin(bic_values)
                best_k_bic = k_range[best_k_bic_idx]
                best_k_aic_idx = np.argmin(aic_values)
                best_k_aic = k_range[best_k_aic_idx]

                # Store results for this covariance type and n_init
                config_key = f"{cov_type}_{n_init}"
                all_results[config_key] = {
                    'k_range': list(k_range),
                    'bic_values': bic_values,
                    'aic_values': aic_values,
                    'll_values': ll_values,
                    'execution_times': execution_times,
                    'best_k_bic': best_k_bic,
                    'best_k_aic': best_k_aic,
                    'best_bic': bic_values[best_k_bic_idx],
                    'cov_type': cov_type,
                    'n_init': n_init
                }

                # Check if this is the best configuration
                if bic_values[best_k_bic_idx] < best_bic:
                    best_bic = bic_values[best_k_bic_idx]
                    best_cov_type = cov_type
                    best_n_init = n_init

                if verbose and table_data:
                    headers = ["K", "BIC", "AIC", "Log-Likelihood", "Time"]
                    print_table(table_data, headers, f"EM Evaluation (covariance='{cov_type}', n_init={n_init})")

                    # Print best K values for this configuration
                    best_k_table = [
                        ["BIC", best_k_bic],
                        ["AIC", best_k_aic]
                    ]
                    print_table(best_k_table, ["Metric", "Best K"],
                                f"Optimal K Values (covariance='{cov_type}', n_init={n_init})")

    # Select results from the best configuration
    if best_cov_type and best_n_init:
        best_config_key = f"{best_cov_type}_{best_n_init}"
        results = all_results[best_config_key]
        results['best_cov_type'] = best_cov_type
        results['best_n_init'] = best_n_init

        # Create comparison data for different configurations
        config_comparison = {
            'cov_type_comparison': {
                ct: min([res['best_bic'] for k, res in all_results.items() if res['cov_type'] == ct])
                for ct in cov_types if any(res['cov_type'] == ct for k, res in all_results.items())
            },
            'n_init_comparison': {
                ni: min([res['best_bic'] for k, res in all_results.items() if res['n_init'] == ni])
                for ni in n_init_values if any(res['n_init'] == ni for k, res in all_results.items())
            }
        }
        results.update(config_comparison)

        if verbose:
            print(f"\nBest configuration: covariance='{best_cov_type}', n_init={best_n_init} (BIC: {best_bic:.2f})")

            # Print covariance type comparison
            cov_table = [[ct, f"{results['cov_type_comparison'][ct]:.2f}"]
                         for ct in results['cov_type_comparison']]
            print_table(cov_table, ["Covariance Type", "Best BIC"], "Covariance Type Comparison")

            # Print n_init comparison
            n_init_table = [[ni, f"{results['n_init_comparison'][ni]:.2f}"]
                            for ni in results['n_init_comparison']]
            print_table(n_init_table, ["n_init", "Best BIC"], "n_init Parameter Comparison")

        return results
    else:
        # All configurations failed
        if verbose:
            print("Warning: All configurations failed to converge properly.")

        # Return dummy results
        return {
            'k_range': list(k_range),
            'bic_values': [float('inf')] * len(k_range),
            'aic_values': [float('inf')] * len(k_range),
            'll_values': [float('-inf')] * len(k_range),
            'execution_times': [0] * len(k_range),
            'best_k_bic': k_range[0],
            'best_k_aic': k_range[0],
            'best_cov_type': None,
            'best_n_init': None
        }


def visualize_kmeans_results(results, dataset_name, save_path='results/clustering'):
    """
    Visualizes K-Means clustering results with multiple metrics.
    """
    k_range = results['k_range']

    # Create a 2x2 figure for all metrics (we'll keep this part)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Silhouette Score
    create_metric_plot(
        x_values=k_range,
        y_values=results['silhouette_scores'],
        title='Silhouette Score vs K',
        xlabel='Number of Clusters (K)',
        ylabel='Silhouette Score',
        best_value=results['best_k_silhouette'],
        best_value_label=f'Best K = {results["best_k_silhouette"]}',
        color='blue',
        filename=f'{save_path}/kmeans_silhouette_{dataset_name.replace(" ", "_").lower()}.png'
    )

    # Plot 2: Inertia (Elbow Method)
    create_metric_plot(
        x_values=k_range,
        y_values=results['inertia_values'],
        title='Inertia vs K (Elbow Method)',
        xlabel='Number of Clusters (K)',
        ylabel='Inertia',
        best_value=results['best_k_elbow'],
        best_value_label=f'Elbow K = {results["best_k_elbow"]}',
        color='green',
        filename=f'{save_path}/kmeans_inertia_{dataset_name.replace(" ", "_").lower()}.png'
    )

    # Plot 3: Calinski-Harabasz Score
    create_metric_plot(
        x_values=k_range,
        y_values=results['ch_scores'],
        title='Calinski-Harabasz Score vs K',
        xlabel='Number of Clusters (K)',
        ylabel='Calinski-Harabasz Score',
        best_value=results['best_k_ch'],
        best_value_label=f'Best K = {results["best_k_ch"]}',
        color='purple',
        filename=f'{save_path}/kmeans_ch_{dataset_name.replace(" ", "_").lower()}.png'
    )

    # Plot 4: Davies-Bouldin Score
    create_metric_plot(
        x_values=k_range,
        y_values=results['db_scores'],
        title='Davies-Bouldin Score vs K',
        xlabel='Number of Clusters (K)',
        ylabel='Davies-Bouldin Score (lower is better)',
        best_value=results['best_k_db'],
        best_value_label=f'Best K = {results["best_k_db"]}',
        color='orange',
        filename=f'{save_path}/kmeans_db_{dataset_name.replace(" ", "_").lower()}.png'
    )

    # Create a combined figure with all metrics (keeping the original combined visualization)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Silhouette Score
    axs[0, 0].plot(k_range, results['silhouette_scores'], 'o-', color='blue')
    axs[0, 0].set_title('Silhouette Score vs K')
    axs[0, 0].set_xlabel('Number of Clusters (K)')
    axs[0, 0].set_ylabel('Silhouette Score')
    axs[0, 0].axvline(x=results['best_k_silhouette'], color='red', linestyle='--')
    axs[0, 0].text(results['best_k_silhouette'], max(results['silhouette_scores']),
                   f'Best K = {results["best_k_silhouette"]}', color='red')

    # Inertia
    axs[0, 1].plot(k_range, results['inertia_values'], 'o-', color='green')
    axs[0, 1].set_title('Inertia vs K (Elbow Method)')
    axs[0, 1].set_xlabel('Number of Clusters (K)')
    axs[0, 1].set_ylabel('Inertia')
    axs[0, 1].axvline(x=results['best_k_elbow'], color='red', linestyle='--')
    axs[0, 1].text(results['best_k_elbow'], results['inertia_values'][k_range.index(results['best_k_elbow'])],
                   f'Elbow K = {results["best_k_elbow"]}', color='red')

    # Calinski-Harabasz Score
    axs[1, 0].plot(k_range, results['ch_scores'], 'o-', color='purple')
    axs[1, 0].set_title('Calinski-Harabasz Score vs K')
    axs[1, 0].set_xlabel('Number of Clusters (K)')
    axs[1, 0].set_ylabel('Calinski-Harabasz Score')
    axs[1, 0].axvline(x=results['best_k_ch'], color='red', linestyle='--')
    axs[1, 0].text(results['best_k_ch'], results['ch_scores'][k_range.index(results['best_k_ch'])],
                   f'Best K = {results["best_k_ch"]}', color='red')

    # Davies-Bouldin Score
    axs[1, 1].plot(k_range, results['db_scores'], 'o-', color='orange')
    axs[1, 1].set_title('Davies-Bouldin Score vs K')
    axs[1, 1].set_xlabel('Number of Clusters (K)')
    axs[1, 1].set_ylabel('Davies-Bouldin Score (lower is better)')
    axs[1, 1].axvline(x=results['best_k_db'], color='red', linestyle='--')
    axs[1, 1].text(results['best_k_db'], results['db_scores'][k_range.index(results['best_k_db'])],
                   f'Best K = {results["best_k_db"]}', color='red')

    plt.tight_layout()
    plt.suptitle(f'K-Means Clustering Evaluation Metrics: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.93)

    # Save the figure
    plt.savefig(f'{save_path}/kmeans_metrics_{dataset_name.replace(" ", "_").lower()}.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    # Plot execution time
    create_metric_plot(
        x_values=k_range,
        y_values=results['execution_times'],
        title=f'Execution Time vs K: {dataset_name}',
        xlabel='Number of Clusters (K)',
        ylabel='Execution Time (seconds)',
        color='tomato',
        filename=f'{save_path}/kmeans_time_{dataset_name.replace(" ", "_").lower()}.png'
    )


def visualize_em_results(results, dataset_name, save_path='results/clustering'):
    """
    Visualizes EM (GMM) clustering results with BIC and AIC metrics.
    """
    k_range = results['k_range']

    # Plot BIC
    create_metric_plot(
        x_values=k_range,
        y_values=results['bic_values'],
        title='BIC vs K',
        xlabel='Number of Components (K)',
        ylabel='BIC (lower is better)',
        best_value=results['best_k_bic'],
        best_value_label=f'Best K = {results["best_k_bic"]}',
        color='blue',
        filename=f'{save_path}/em_bic_{dataset_name.replace(" ", "_").lower()}.png'
    )

    # Plot AIC
    create_metric_plot(
        x_values=k_range,
        y_values=results['aic_values'],
        title='AIC vs K',
        xlabel='Number of Components (K)',
        ylabel='AIC (lower is better)',
        best_value=results['best_k_aic'],
        best_value_label=f'Best K = {results["best_k_aic"]}',
        color='green',
        filename=f'{save_path}/em_aic_{dataset_name.replace(" ", "_").lower()}.png'
    )

    # Plot Log-Likelihood
    create_metric_plot(
        x_values=k_range,
        y_values=results['ll_values'],
        title='Log-Likelihood vs K',
        xlabel='Number of Components (K)',
        ylabel='Log-Likelihood (higher is better)',
        color='purple',
        filename=f'{save_path}/em_loglikelihood_{dataset_name.replace(" ", "_").lower()}.png'
    )

    # Create a combined figure with all metrics (keeping the original combined visualization)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot BIC
    axs[0].plot(k_range, results['bic_values'], 'o-', color='blue')
    axs[0].set_title('BIC vs K')
    axs[0].set_xlabel('Number of Components (K)')
    axs[0].set_ylabel('BIC (lower is better)')
    axs[0].axvline(x=results['best_k_bic'], color='red', linestyle='--')
    axs[0].text(results['best_k_bic'], results['bic_values'][k_range.index(results['best_k_bic'])],
                f'Best K = {results["best_k_bic"]}', color='red')

    # Plot AIC
    axs[1].plot(k_range, results['aic_values'], 'o-', color='green')
    axs[1].set_title('AIC vs K')
    axs[1].set_xlabel('Number of Components (K)')
    axs[1].set_ylabel('AIC (lower is better)')
    axs[1].axvline(x=results['best_k_aic'], color='red', linestyle='--')
    axs[1].text(results['best_k_aic'], results['aic_values'][k_range.index(results['best_k_aic'])],
                f'Best K = {results["best_k_aic"]}', color='red')

    # Plot Log-Likelihood
    axs[2].plot(k_range, results['ll_values'], 'o-', color='purple')
    axs[2].set_title('Log-Likelihood vs K')
    axs[2].set_xlabel('Number of Components (K)')
    axs[2].set_ylabel('Log-Likelihood (higher is better)')

    plt.tight_layout()
    plt.suptitle(f'EM (GMM) Clustering Evaluation Metrics: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)

    # Save the figure
    plt.savefig(f'{save_path}/em_metrics_{dataset_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot execution time
    create_metric_plot(
        x_values=k_range,
        y_values=results['execution_times'],
        title=f'EM Execution Time vs K: {dataset_name}',
        xlabel='Number of Components (K)',
        ylabel='Execution Time (seconds)',
        color='tomato',
        filename=f'{save_path}/em_time_{dataset_name.replace(" ", "_").lower()}.png'
    )


def analyze_clusters(X, labels, feature_names, dataset_name, cluster_method, save_path='results/clustering'):
    """
    Analyzes the characteristics of each cluster.
    """
    # Create a DataFrame with features and cluster labels
    cluster_df = pd.DataFrame(X, columns=feature_names)
    cluster_df['Cluster'] = labels

    # Calculate cluster statistics
    cluster_stats = cluster_df.groupby('Cluster').mean()

    # Calculate normalized values for radar chart (scale to 0-1)
    normalized_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())

    # Plot cluster profile (top 8 discriminating features)
    variance_by_feature = cluster_df.groupby('Cluster').var().mean(axis=0)
    top_features = variance_by_feature.sort_values(ascending=False).head(8).index

    # Create a radar chart for each cluster
    n_clusters = len(cluster_stats)

    # Create a figure for cluster profiles
    plt.figure(figsize=(12, 8))
    for i, cluster_id in enumerate(cluster_stats.index):
        plt.subplot(1, n_clusters, i + 1)
        profile = cluster_stats.loc[cluster_id, top_features]
        plt.bar(range(len(top_features)), profile, alpha=0.7)
        plt.xticks(range(len(top_features)), top_features, rotation=90)
        plt.title(f'Cluster {cluster_id} Profile')

    plt.tight_layout()
    plt.suptitle(f'{cluster_method} Cluster Profiles for {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_path}/{cluster_method.lower()}_profiles_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create a 2D visualization of clusters using t-SNE
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='plasma', alpha=1.0)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f't-SNE Visualization of {cluster_method} Clusters for {dataset_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(f'{save_path}/{cluster_method.lower()}_tsne_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Return cluster statistics
    return cluster_stats


def analyze_feature_distributions_by_cluster(X, labels, feature_names, n_top_features=5, verbose=True):
    """
    Analyzes how features distribute across different clusters.

    Parameters:
    -----------
    X : array-like
        Input data
    labels : array-like
        Cluster assignments
    feature_names : list
        List of feature names
    n_top_features : int, default=5
        Number of top discriminative features to analyze
    verbose : bool, default=True
        Whether to print results

    Returns:
    --------
    dict
        Dictionary containing feature analysis by cluster
    """
    # Create DataFrame with features and cluster labels
    cluster_df = pd.DataFrame(X, columns=feature_names)
    cluster_df['Cluster'] = labels

    # Get unique clusters
    clusters = np.unique(labels)
    n_clusters = len(clusters)

    if verbose:
        print(f"\nAnalyzing feature distributions across {n_clusters} clusters")

    # Calculate feature importance for differentiation
    feature_importance = {}

    for feature in feature_names:
        # Calculate F-statistic for this feature (ANOVA)
        feature_values = cluster_df[feature].values
        cluster_means = [cluster_df[cluster_df['Cluster'] == c][feature].mean() for c in clusters]
        cluster_vars = [cluster_df[cluster_df['Cluster'] == c][feature].var() for c in clusters]
        cluster_sizes = [len(cluster_df[cluster_df['Cluster'] == c]) for c in clusters]

        # Calculate between-group and within-group variances
        grand_mean = np.mean(feature_values)
        between_var = sum(size * (mean - grand_mean) ** 2 for size, mean in zip(cluster_sizes, cluster_means))
        within_var = sum((size - 1) * var for size, var in zip(cluster_sizes, cluster_vars))

        # Calculate degrees of freedom
        df_between = n_clusters - 1
        df_within = len(feature_values) - n_clusters

        # Calculate F-statistic (higher means more discriminative)
        if within_var == 0:
            f_stat = float('inf')  # Avoid division by zero
        else:
            f_stat = (between_var / df_between) / (within_var / df_within)

        feature_importance[feature] = f_stat

    # Get top discriminative features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:n_top_features]
    top_feature_names = [f[0] for f in top_features]

    # Calculate cluster statistics for top features
    cluster_stats = {}
    for cluster in clusters:
        cluster_stats[cluster] = {
            'size': len(cluster_df[cluster_df['Cluster'] == cluster]),
            'percentage': len(cluster_df[cluster_df['Cluster'] == cluster]) / len(cluster_df) * 100,
            'features': {}
        }

        for feature in top_feature_names:
            feature_values = cluster_df[cluster_df['Cluster'] == cluster][feature]
            cluster_stats[cluster]['features'][feature] = {
                'mean': feature_values.mean(),
                'std': feature_values.std(),
                'min': feature_values.min(),
                'max': feature_values.max(),
                'z_score': (feature_values.mean() - cluster_df[feature].mean()) / cluster_df[feature].std()
            }

    if verbose:
        # Print top discriminative features
        print("\nTop Discriminative Features:")
        feature_table = [[name, f"{importance:.2f}"] for name, importance in top_features]
        print_table(feature_table, ["Feature", "F-statistic"], "Feature Importance by F-statistic")

        # Print cluster statistics
        print("\nCluster Statistics:")
        for cluster in clusters:
            print(
                f"\nCluster {cluster} ({cluster_stats[cluster]['size']} samples, {cluster_stats[cluster]['percentage']:.1f}%)")

            feature_stats = []
            for feature in top_feature_names:
                stats = cluster_stats[cluster]['features'][feature]
                feature_stats.append([
                    feature,
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['z_score']:.2f}"
                ])

            print_table(feature_stats, ["Feature", "Mean", "Std Dev", "Z-score"],
                        f"Feature Statistics for Cluster {cluster}")

    # Return comprehensive results
    return {
        'feature_importance': feature_importance,
        'top_features': top_features,
        'cluster_stats': cluster_stats,
        'cluster_profiles': cluster_df.groupby('Cluster').mean()
    }


def visualize_cluster_feature_distributions(analysis_results, dataset_name,
                                            save_path='results/clustering', n_features=5):
    """
    Visualizes feature distributions across clusters.
    """
    # Get top features and cluster stats
    top_features = [f[0] for f in analysis_results['top_features'][:n_features]]
    cluster_stats = analysis_results['cluster_stats']
    clusters = sorted(cluster_stats.keys())

    # Create heatmap of z-scores for top features across clusters
    plt.figure(figsize=(12, 8))

    # Prepare z-score matrix
    z_score_matrix = np.zeros((len(top_features), len(clusters)))
    for i, feature in enumerate(top_features):
        for j, cluster in enumerate(clusters):
            z_score_matrix[i, j] = cluster_stats[cluster]['features'][feature]['z_score']

    # Create heatmap
    sns.heatmap(z_score_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=[f"Cluster {c}" for c in clusters],
                yticklabels=top_features)

    plt.title(f'Feature Z-scores by Cluster: {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_zscores_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create bar chart of cluster sizes
    plt.figure(figsize=(10, 6))
    sizes = [cluster_stats[c]['size'] for c in clusters]
    percentages = [cluster_stats[c]['percentage'] for c in clusters]

    bars = plt.bar(range(len(clusters)), sizes, alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title(f'Cluster Sizes: {dataset_name}')
    plt.xticks(range(len(clusters)), [f"Cluster {c}" for c in clusters])

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{pct:.1f}%', ha='center', va='bottom')

    plt.savefig(f'{save_path}/cluster_sizes_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create feature mean value bar charts for each cluster
    for feature in top_features:
        plt.figure(figsize=(12, 6))

        # Get feature values by cluster
        feature_by_cluster = [analysis_results['cluster_profiles'].loc[c, feature] for c in clusters]

        # Create bar chart instead of boxplot since we're dealing with means
        plt.bar(range(len(clusters)), feature_by_cluster)
        plt.xticks(range(len(clusters)), [f"Cluster {c}" for c in clusters])
        plt.ylabel(feature)
        plt.title(f'{feature} Mean Value by Cluster: {dataset_name}')
        plt.grid(True, alpha=0.3, axis='y')

        plt.savefig(f'{save_path}/{feature}_by_cluster_{dataset_name.replace(" ", "_").lower()}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()



def compare_clustering_algorithms(X, y, kmeans_k, em_k, feature_names, dataset_name, save_path='results/clustering'):
    """
    Compares K-Means and EM clustering algorithms.
    """
    # Apply K-Means
    kmeans = KMeans(n_clusters=kmeans_k, random_state=RANDOM_SEED)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_silhouette = silhouette_score(X, kmeans_labels)

    # Apply EM (GMM)
    em = GaussianMixture(n_components=em_k, random_state=RANDOM_SEED)
    em.fit(X)
    em_labels = em.predict(X)
    em_silhouette = silhouette_score(X, em_labels)

    # Create comparison table
    comparison_data = {
        'Algorithm': ['K-Means', 'EM (GMM)'],
        'Number of Clusters': [kmeans_k, em_k],
        'Silhouette Score': [kmeans_silhouette, em_silhouette]
    }

    # Check agreement with true labels if available
    if y is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        kmeans_ari = adjusted_rand_score(y, kmeans_labels)
        em_ari = adjusted_rand_score(y, em_labels)
        kmeans_nmi = normalized_mutual_info_score(y, kmeans_labels)
        em_nmi = normalized_mutual_info_score(y, em_labels)

        comparison_data['ARI with True Labels'] = [kmeans_ari, em_ari]
        comparison_data['NMI with True Labels'] = [kmeans_nmi, em_nmi]

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison to CSV
    comparison_df.to_csv(f'{save_path}/clustering_comparison_{dataset_name.replace(" ", "_").lower()}.csv', index=False)

    # Create a visualization to compare cluster assignments
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    X_tsne = tsne.fit_transform(X)

    # K-Means clusters
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    plt.title(f'K-Means Clusters (K={kmeans_k})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # EM clusters
    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=em_labels, cmap='viridis', alpha=0.7)
    plt.title(f'EM Clusters (K={em_k})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.tight_layout()
    plt.suptitle(f'K-Means vs EM Clustering for {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)

    plt.savefig(f'{save_path}/clustering_comparison_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return comparison_df, kmeans_labels, em_labels


# ==========================================================================
# DIMENSIONALITY REDUCTION ALGORITHMS
# ==========================================================================

def pca_sensitivity_analysis(X, variance_thresholds=[0.9, 0.95, 0.99], verbose=True):
    """
    Analyzes how PCA performance varies with different variance thresholds.

    Parameters:
    -----------
    X : array-like
        Input data for dimensionality reduction
    variance_thresholds : list, default=[0.9, 0.95, 0.99]
        List of variance thresholds to test
    verbose : bool, default=True
        Whether to print results

    Returns:
    --------
    dict
        Dictionary containing results for each threshold
    """
    results = {}

    if verbose:
        print_section_header("PCA Sensitivity Analysis", level=2)
        table_data = []

    for threshold in variance_thresholds:
        # Run PCA with this threshold
        pca_results = apply_pca(X, variance_threshold=threshold)

        # Store results
        results[threshold] = {
            'n_components': pca_results['n_components_threshold'],
            'dimension_reduction': 1 - (pca_results['n_components_threshold'] / X.shape[1]),
            'execution_time': pca_results['execution_time'],
            'explained_variance': pca_results['explained_variance'],
            'components': pca_results['components'],
            'X_reduced': pca_results['X_reduced'][:, :pca_results['n_components_threshold']]
        }

        if verbose:
            table_data.append([
                f"{threshold:.0%}",
                pca_results['n_components_threshold'],
                f"{100 * (1 - pca_results['n_components_threshold'] / X.shape[1]):.1f}%",
                f"{pca_results['execution_time']:.4f}s"
            ])

    if verbose:
        headers = ["Variance Threshold", "Components", "Dimension Reduction", "Execution Time"]
        print_table(table_data, headers, "PCA Sensitivity Analysis Results")

    return results


def visualize_pca_sensitivity(sensitivity_results, dataset_name, save_path='results/dim_reduction'):
    """
    Visualizes PCA sensitivity analysis results.
    """
    thresholds = list(sensitivity_results.keys())
    components = [sensitivity_results[t]['n_components'] for t in thresholds]

    # Create figure for component count vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, components, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Variance Threshold')
    plt.ylabel('Number of Components')
    plt.title(f'PCA Components vs Variance Threshold: {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.xticks([t for t in thresholds], [f"{t:.0%}" for t in thresholds])

    # Add value labels
    for i, comp in enumerate(components):
        plt.text(thresholds[i], comp + 0.3, str(comp), ha='center')

    # Save the figure
    plt.savefig(f'{save_path}/pca_sensitivity_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return



def apply_pca(X, max_components=None, variance_threshold=0.95):
    """
    Applies PCA, determines optimal components, and returns reduced data.

    Principal Component Analysis works by:
    1. Finding orthogonal directions (eigenvectors) of maximum variance in the data
    2. Ranking these directions by their variance contribution (eigenvalues)
    3. Selecting top components that capture most variance

    The variance threshold approach (e.g., 95%) ensures we retain enough
    information while reducing dimensionality. This is more theoretically
    justified than arbitrary component selection.

    Parameters:
    -----------
    X : array-like
        Input data to reduce dimensions
    max_components : int, optional
        Maximum number of components to consider
    variance_threshold : float, default=0.95
        Minimum cumulative variance to retain (0.0-1.0)

    Returns:
    --------
    dict
        Dictionary containing reduced data and metadata
    """
    print_section_header("Applying PCA", level=2)

    # If max_components not specified, use all features
    if max_components is None:
        max_components = X.shape[1]

    # Create PCA model with whitening for better component separation
    pca = PCA(n_components=max_components, random_state=RANDOM_SEED)

    # Fit and transform the data
    start_time = time.time()
    X_pca = pca.fit_transform(X)
    execution_time = time.time() - start_time

    # Calculate explained variance metrics
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find optimal components based on variance threshold - theoretically justified approach
    n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Create results dictionary
    results = {
        'X_reduced': X_pca,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'components': pca.components_,
        'execution_time': execution_time,
        'n_components_threshold': n_components_threshold,
        'model': pca
    }

    # Print results summary
    summary_table = [
        ["Execution Time", f"{execution_time:.2f} seconds"],
        ["Original Dimensions", X.shape[1]],
        [f"Optimal Components ({variance_threshold * 100}% variance)", n_components_threshold],
        ["Dimension Reduction",
         f"{X.shape[1]} → {n_components_threshold} ({n_components_threshold / X.shape[1] * 100:.1f}%)"],
        ["Largest Principal Component Variance", f"{explained_variance[0] * 100:.2f}%"],
        ["Data Rank (estimated)", f"{np.linalg.matrix_rank(X)}"]
    ]
    print_table(summary_table, ["Metric", "Value"], "PCA Results Summary")

    return results


def apply_ica(X, max_components=None):
    """
    Applies ICA and determines optimal components based on kurtosis.

    Independent Component Analysis (ICA) works by:
    1. Seeking statistically independent source signals that were mixed to create the data
    2. Maximizing non-Gaussianity of components (via kurtosis or negentropy)
    3. Unlike PCA, components are not ranked by variance but by independence

    Kurtosis measures the "tailedness" of a distribution. High absolute kurtosis
    indicates potential meaningful independent components, as truly independent
    signals tend to be non-Gaussian (Central Limit Theorem).

    Parameters:
    -----------
    X : array-like
        Input data to reduce dimensions
    max_components : int, optional
        Maximum number of components to consider

    Returns:
    --------
    dict
        Dictionary containing reduced data and metadata
    """
    print_section_header("Applying ICA", level=2)

    # If max_components not specified, use all features (bounded by sample count)
    if max_components is None:
        max_components = min(X.shape[1], X.shape[0])

    # Test different numbers of components to find optimal value
    components_range = range(2, max_components + 1)
    avg_kurtosis = []
    execution_times = []
    table_data = []

    for n_comp in components_range:
        start_time = time.time()

        # Create and fit ICA model with defined tolerance and iterations
        ica = FastICA(
            n_components=n_comp,
            random_state=RANDOM_SEED,
            max_iter=1000,  # Ensure convergence
            tol=1e-4 # Tight tolerance for better components
        )
        X_ica = ica.fit_transform(X)

        # Calculate kurtosis for each component
        # Higher absolute kurtosis means more non-Gaussian (independent) components
        kurtosis_values = np.abs([stats.kurtosis(X_ica[:, i]) for i in range(n_comp)])
        avg_kurt = np.mean(kurtosis_values)
        avg_kurtosis.append(avg_kurt)

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        table_data.append([n_comp, f"{avg_kurt:.4f}", f"{execution_time:.2f}s"])

    # Find optimal number of components (highest average kurtosis)
    # This is theoretically justified as components with high kurtosis are more likely independent
    optimal_components = components_range[np.argmax(avg_kurtosis)]

    # Refit with optimal components
    start_time = time.time()
    ica_optimal = FastICA(
        n_components=optimal_components,
        random_state=RANDOM_SEED,
        max_iter=2000  # More iterations for final model to ensure convergence
    )
    X_ica_optimal = ica_optimal.fit_transform(X)
    final_execution_time = time.time() - start_time

    # Calculate final kurtosis values for analysis
    kurtosis_values = np.abs([stats.kurtosis(X_ica_optimal[:, i]) for i in range(optimal_components)])

    # Create results dictionary
    results = {
        'X_reduced': X_ica_optimal,
        'components_range': list(components_range),
        'avg_kurtosis': avg_kurtosis,
        'kurtosis_values': kurtosis_values,
        'execution_times': execution_times,
        'optimal_components': optimal_components,
        'final_execution_time': final_execution_time,
        'mixing_matrix': ica_optimal.mixing_,
        'model': ica_optimal
    }

    # Print results in a table
    headers = ["Components", "Avg. Kurtosis", "Time"]
    print_table(table_data, headers, "ICA Component Evaluation")

    # Print summary of optimal configuration
    summary_table = [
        ["Optimal Components", optimal_components],
        ["Maximum Avg. Kurtosis", f"{max(avg_kurtosis):.4f}"],
        ["Kurtosis Range", f"{min(kurtosis_values):.2f} - {max(kurtosis_values):.2f}"],
        ["Gaussian Kurtosis Reference", "3.0 (For comparison)"],
        ["Final Execution Time", f"{final_execution_time:.2f} seconds"],
        ["Dimension Reduction", f"{X.shape[1]} → {optimal_components} ({optimal_components / X.shape[1] * 100:.1f}%)"]
    ]
    print_table(summary_table, ["Metric", "Value"], "ICA Optimal Configuration")

    return results


def ica_sensitivity_analysis(X, kurtosis_thresholds=[3.0, 4.0, 5.0, 6.0], verbose=True):
    """
    Analyzes how ICA performance varies with different kurtosis thresholds.
    """
    results = {}

    if verbose:
        print_section_header("ICA Sensitivity Analysis", level=2)
        table_data = []

    # First run ICA for all possible components to get kurtosis values
    max_components = min(X.shape[1], X.shape[0])
    components_range = range(2, max_components + 1)

    # Storage for kurtosis values by component
    component_kurtosis = []
    execution_times = []

    # Run ICA for each number of components
    for n_comp in components_range:
        start_time = time.time()

        ica = FastICA(
            n_components=n_comp,
            random_state=RANDOM_SEED,
            max_iter=1000,
            tol=1e-4
        )
        X_ica = ica.fit_transform(X)

        # Calculate kurtosis for each component
        kurtosis_values = np.abs([stats.kurtosis(X_ica[:, i]) for i in range(n_comp)])
        avg_kurt = np.mean(kurtosis_values)

        component_kurtosis.append({
            'n_components': n_comp,
            'avg_kurtosis': avg_kurt,
            'execution_time': time.time() - start_time,
            'kurtosis_values': kurtosis_values,
            'model': ica,  # Store the model for later use
            'X_transformed': X_ica  # Store the transformed data
        })

    # For each threshold, find optimal number of components
    for threshold in kurtosis_thresholds:
        # Find first component count that exceeds the threshold
        optimal_comp_data = None
        for comp_data in component_kurtosis:
            if comp_data['avg_kurtosis'] >= threshold:
                optimal_comp_data = comp_data
                break

        # If no component meets threshold, use the one with highest kurtosis
        if optimal_comp_data is None:
            optimal_comp_data = max(component_kurtosis, key=lambda x: x['avg_kurtosis'])

        n_comp = optimal_comp_data['n_components']

        # Store results including the transformed data
        results[threshold] = {
            'n_components': n_comp,
            'dimension_reduction': 1 - (n_comp / X.shape[1]),
            'avg_kurtosis': optimal_comp_data['avg_kurtosis'],
            'execution_time': optimal_comp_data['execution_time'],
            'X_reduced': optimal_comp_data['X_transformed'],  # Add the transformed data
            'model': optimal_comp_data['model']  # Add the model for future transformations
        }

        if verbose:
            table_data.append([
                f"{threshold:.1f}",
                n_comp,
                f"{100 * (1 - n_comp / X.shape[1]):.1f}%",
                f"{optimal_comp_data['avg_kurtosis']:.4f}",
                f"{optimal_comp_data['execution_time']:.4f}s"
            ])

    if verbose:
        headers = ["Kurtosis Threshold", "Components", "Dimension Reduction", "Avg. Kurtosis", "Execution Time"]
        print_table(table_data, headers, "ICA Sensitivity Analysis Results")

    return results


def visualize_ica_sensitivity(sensitivity_results, dataset_name, save_path='results/dim_reduction'):
    """
    Visualizes ICA sensitivity analysis results.
    """
    thresholds = list(sensitivity_results.keys())
    components = [sensitivity_results[t]['n_components'] for t in thresholds]
    kurtosis = [sensitivity_results[t]['avg_kurtosis'] for t in thresholds]

    # Create figure for component count vs threshold
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, components, 'o-', color='purple', linewidth=2, markersize=8)
    plt.xlabel('Kurtosis Threshold')
    plt.ylabel('Number of Components')
    plt.title('Components vs Kurtosis Threshold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, comp in enumerate(components):
        plt.text(thresholds[i], comp + 0.3, str(comp), ha='center')

    plt.subplot(1, 2, 2)
    plt.plot(components, kurtosis, 'o-', color='magenta', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Average Absolute Kurtosis')
    plt.title('Kurtosis vs Components')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, kurt in enumerate(kurtosis):
        plt.text(components[i], kurt + 0.1, f"{kurt:.2f}", ha='center')

    plt.tight_layout()
    plt.suptitle(f'ICA Sensitivity: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)

    # Save the figure
    plt.savefig(f'{save_path}/ica_sensitivity_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return





def rp_sensitivity_analysis(X, error_thresholds=[0.3, 0.4, 0.5], n_trials=5, verbose=True):
    """
    Analyzes how Random Projection performance varies with different error thresholds.

    Parameters:
    -----------
    X : array-like
        Input data for dimensionality reduction
    error_thresholds : list, default=[0.3, 0.4, 0.5]
        List of reconstruction error thresholds to test
    n_trials : int, default=5
        Number of trials for each configuration
    verbose : bool, default=True
        Whether to print results

    Returns:
    --------
    dict
        Dictionary containing results for each threshold
    """
    results = {}

    if verbose:
        print_section_header("Random Projection Sensitivity Analysis", level=2)
        table_data = []

    for threshold in error_thresholds:
        # Run RP with this threshold
        rp_results = apply_rp(X, error_threshold=threshold, n_trials=n_trials)

        # Store results
        results[threshold] = {
            'optimal_components': rp_results['optimal_components'],
            'dimension_reduction': 1 - (rp_results['optimal_components'] / X.shape[1]),
            'reconstruction_error': rp_results['avg_reconstruction_errors'][
                rp_results['components_range'].index(rp_results['optimal_components'])],
            'error_std': rp_results['std_errors'][
                rp_results['components_range'].index(rp_results['optimal_components'])],
            'execution_time': rp_results['final_execution_time'],
            'X_reduced': rp_results['X_reduced']
        }

        if verbose:
            table_data.append([
                f"{threshold:.2f}",
                rp_results['optimal_components'],
                f"{100 * (1 - rp_results['optimal_components'] / X.shape[1]):.1f}%",
                f"{results[threshold]['reconstruction_error']:.4f} ± {results[threshold]['error_std']:.4f}",
                f"{rp_results['final_execution_time']:.4f}s"
            ])

    if verbose:
        headers = ["Error Threshold", "Components", "Dimension Reduction", "Reconstruction Error", "Execution Time"]
        print_table(table_data, headers, "Random Projection Sensitivity Analysis Results")

    return results


def visualize_rp_sensitivity(sensitivity_results, dataset_name, save_path='results/dim_reduction'):
    """
    Visualizes Random Projection sensitivity analysis results.
    """
    thresholds = list(sensitivity_results.keys())
    components = [sensitivity_results[t]['optimal_components'] for t in thresholds]
    errors = [sensitivity_results[t]['reconstruction_error'] for t in thresholds]

    # Create figure for component count vs threshold
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, components, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('Error Threshold')
    plt.ylabel('Number of Components')
    plt.title('Components vs Error Threshold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, comp in enumerate(components):
        plt.text(thresholds[i], comp + 0.3, str(comp), ha='center')

    plt.subplot(1, 2, 2)
    plt.plot(components, errors, 'o-', color='red', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs Components')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, err in enumerate(errors):
        plt.text(components[i], err + 0.01, f"{err:.3f}", ha='center')

    plt.tight_layout()
    plt.suptitle(f'Random Projection Sensitivity: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)

    # Save the figure
    plt.savefig(f'{save_path}/rp_sensitivity_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return



def apply_rp(X, max_components=None, n_trials=5, error_threshold=0.4):
    """
    Applies Random Projection and determines optimal components based on reconstruction error.

    Random Projection works via the Johnson-Lindenstrauss lemma:
    1. Projects high-dimensional data to lower dimensions using random matrices
    2. Preserves pairwise distances with high probability
    3. Computationally efficient compared to PCA/ICA
    4. Trade-off: Some information loss but faster computation

    Reconstruction error is used to determine optimal components by measuring
    how well the original data can be recovered from the projection.

    Parameters:
    -----------
    X : array-like
        Input data to reduce dimensions
    max_components : int, optional
        Maximum number of components to test
    n_trials : int, default=5
        Number of trials for each component size to account for randomness
    error_threshold : float, default=0.4
        Target reconstruction error threshold for determining optimal components

    Returns:
    --------
    dict
        Dictionary containing all results including the reduced data and optimal components
    """
    print_section_header("Applying Random Projection", level=2)

    # If max_components not specified, use all features
    if max_components is None:
        max_components = X.shape[1]

    # Test different numbers of components
    components_range = range(2, max_components + 1)
    avg_reconstruction_errors = []
    execution_times = []
    std_errors = []
    table_data = []

    for n_comp in components_range:
        reconstruction_errors = []
        trial_times = []

        for trial in range(n_trials):
            start_time = time.time()

            # Create and fit RP model with Gaussian distribution
            # Johnson-Lindenstrauss lemma guarantees distance preservation
            rp = GaussianRandomProjection(
                n_components=n_comp,
                random_state=RANDOM_SEED + trial,
                eps=0.5  # Controls error tolerance in the Johnson-Lindenstrauss lemma
            )
            X_rp = rp.fit_transform(X)

            # Calculate reconstruction error using pseudo-inverse
            # For reconstruction: X_reconstructed ≈ X_rp · P^T
            # Where P is the pseudoinverse of the components matrix
            components = rp.components_  # Shape (n_components, n_features)
            P = pinv(components)  # Shape (n_features, n_components)
            X_reconstructed = np.dot(X_rp, P.T)

            # Mean squared error quantifies reconstruction quality
            error = mean_squared_error(X, X_reconstructed)
            reconstruction_errors.append(error)

            trial_time = time.time() - start_time
            trial_times.append(trial_time)

        # Calculate average across trials to account for randomness
        avg_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        avg_time = np.mean(trial_times)

        avg_reconstruction_errors.append(avg_error)
        std_errors.append(std_error)
        execution_times.append(avg_time)

        table_data.append([
            n_comp,
            f"{avg_error:.4f} ± {std_error:.4f}",
            f"{avg_time:.2f}s"
        ])

    # Find optimal number of components using error threshold
    # Find the smallest number of components that gives error below threshold
    optimal_components = None
    for i, n_comp in enumerate(components_range):
        if avg_reconstruction_errors[i] <= error_threshold:
            optimal_components = n_comp
            break

    # If no component set meets the threshold, use the kneed method as a fallback
    if optimal_components is None:
        # Try to find elbow point
        try:
            from kneed import KneeLocator
            kl = KneeLocator(list(components_range), avg_reconstruction_errors, curve='convex', direction='decreasing')
            optimal_components = kl.elbow if kl.elbow is not None else components_range[-1]
        except (ImportError, ValueError):
            # If kneed is not available or fails, just take the component with lowest error
            optimal_components = components_range[np.argmin(avg_reconstruction_errors)]

    # Refit with optimal components
    start_time = time.time()
    rp_optimal = GaussianRandomProjection(n_components=optimal_components, random_state=RANDOM_SEED)
    X_rp_optimal = rp_optimal.fit_transform(X)
    final_execution_time = time.time() - start_time

    # Create results dictionary
    results = {
        'X_reduced': X_rp_optimal,
        'components_range': list(components_range),
        'avg_reconstruction_errors': avg_reconstruction_errors,
        'std_errors': std_errors,
        'execution_times': execution_times,
        'optimal_components': optimal_components,
        'final_execution_time': final_execution_time,
        'components': rp_optimal.components_,
        'model': rp_optimal
    }

    # Print results in a table
    headers = ["Components", "Reconstruction Error", "Time"]
    print_table(table_data, headers, "Random Projection Component Evaluation")

    # Print summary of optimal configuration
    summary_table = [
        ["Optimal Components", optimal_components],
        ["Reconstruction Error", f"{avg_reconstruction_errors[components_range.index(optimal_components)]:.4f}"],
        ["Error Variability (Std)", f"{std_errors[components_range.index(optimal_components)]:.4f}"],
        ["Final Execution Time", f"{final_execution_time:.2f} seconds"],
        ["Dimension Reduction", f"{X.shape[1]} → {optimal_components} ({optimal_components / X.shape[1] * 100:.1f}%)"],
        ["Johnson-Lindenstrauss Error Guarantee", "Distance preservation within stated epsilon"]
    ]
    print_table(summary_table, ["Metric", "Value"], "Random Projection Optimal Configuration")

    return results


def visualize_pca_results(pca_results, dataset_name, save_path='results/dim_reduction'):
    """
    Visualizes PCA results including explained variance and component analysis.
    """
    # Extract results
    explained_variance = pca_results['explained_variance']
    cumulative_variance = pca_results['cumulative_variance']
    n_components = len(explained_variance)
    components = pca_results['components']
    components_threshold = pca_results['n_components_threshold']

    # Create figure for individual explained variance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, color='darkblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Explained Variance')
    plt.xticks(range(1, n_components + 1, max(1, n_components // 10)))

    # Create figure for cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components + 1), cumulative_variance, 'o-', color='red')
    plt.axhline(y=0.95, linestyle='--', color='green', alpha=0.7)
    plt.axvline(x=components_threshold, linestyle='--', color='darkblue', alpha=0.7)
    plt.text(components_threshold + 0.5, 0.5, f'Threshold: {components_threshold} components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.xticks(range(1, n_components + 1, max(1, n_components // 10)))

    plt.tight_layout()
    plt.suptitle(f'PCA Results: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_path}/pca_variance_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot top components heatmap (focus on first few components)
    n_plot_components = min(10, components.shape[0])
    n_plot_features = min(20, components.shape[1])

    plt.figure(figsize=(14, 8))
    sns.heatmap(components[:n_plot_components, :n_plot_features],
                cmap='coolwarm', center=0,
                xticklabels=range(1, n_plot_features + 1),
                yticklabels=[f'PC{i + 1}' for i in range(n_plot_components)])
    plt.xlabel('Feature Index')
    plt.ylabel('Principal Component')
    plt.title(f'PCA Component Weights Heatmap: {dataset_name}')
    plt.savefig(f'{save_path}/pca_components_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def visualize_ica_results(ica_results, dataset_name, save_path='results/dim_reduction'):
    """
    Visualizes ICA results including kurtosis and component analysis.

    This function creates three main visualizations:
    1. Average kurtosis across different component counts
    2. Individual kurtosis values for each component
    3. Execution time across different component counts

    Parameters:
    -----------
    ica_results : dict
        Dictionary containing ICA results
    dataset_name : str
        Name of the dataset for titles and filenames
    save_path : str
        Directory to save the visualization files
    """
    # Extract results
    components_range = ica_results['components_range']
    avg_kurtosis = ica_results['avg_kurtosis']
    kurtosis_values = ica_results['kurtosis_values']
    optimal_components = ica_results['optimal_components']

    # Create figure for average kurtosis using the utility function
    create_metric_plot(
        x_values=components_range,
        y_values=avg_kurtosis,
        title=f'ICA Average Kurtosis by Number of Components: {dataset_name}',
        xlabel='Number of Components',
        ylabel='Average Absolute Kurtosis',
        best_value=optimal_components,
        best_value_label=f'Optimal: {optimal_components} components',
        filename=f'{save_path}/ica_avg_kurtosis_{dataset_name.replace(" ", "_").lower()}.png',
        color='purple'
    )

    # Create figure for individual kurtosis values - this is a specialized bar chart
    # with a reference line, so we'll keep the original implementation
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(kurtosis_values) + 1), kurtosis_values, alpha=0.7, color='teal')
    plt.axhline(y=3, linestyle='--', color='red', alpha=0.7, label='Gaussian Kurtosis')
    plt.xlabel('Independent Component')
    plt.ylabel('Absolute Kurtosis')
    plt.title(f'Kurtosis of Each Independent Component: {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/ica_components_kurtosis_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot execution time using the utility function
    create_metric_plot(
        x_values=components_range,
        y_values=ica_results['execution_times'],
        title=f'ICA Execution Time by Number of Components: {dataset_name}',
        xlabel='Number of Components',
        ylabel='Execution Time (seconds)',
        color='tomato',
        filename=f'{save_path}/ica_time_{dataset_name.replace(" ", "_").lower()}.png'
    )


def visualize_rp_results(rp_results, dataset_name, save_path='results/dim_reduction'):
    """
    Visualizes Random Projection results including reconstruction error.
    """
    # Extract results
    components_range = rp_results['components_range']
    avg_reconstruction_errors = rp_results['avg_reconstruction_errors']
    std_errors = rp_results['std_errors']
    optimal_components = rp_results['optimal_components']

    # Create figure for reconstruction error with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(components_range, avg_reconstruction_errors, yerr=std_errors,
                 fmt='o-', color='blue', ecolor='lightblue', elinewidth=2, capsize=4)
    plt.axvline(x=optimal_components, linestyle='--', color='red', alpha=0.7)
    plt.text(optimal_components + 0.5,
             min(avg_reconstruction_errors) + (max(avg_reconstruction_errors) - min(avg_reconstruction_errors)) * 0.1,
             f'Optimal: {optimal_components} components')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Reconstruction Error')
    plt.title(f'RP Reconstruction Error by Number of Components: {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/rp_reconstruction_error_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot execution time
    create_metric_plot(
        x_values=components_range,
        y_values=rp_results['execution_times'],
        title=f'RP Execution Time by Number of Components: {dataset_name}',
        xlabel='Number of Components',
        ylabel='Execution Time (seconds)',
        color='tomato',
        filename=f'{save_path}/rp_time_{dataset_name.replace(" ", "_").lower()}.png'
    )



def visualize_manifold_embeddings(manifold_results, dataset_name, save_path='results/extra'):
    """
    Creates a combined visualization of all manifold embeddings side by side.

    """
    embeddings = manifold_results['embeddings']

    # Create a figure with subplots for each embedding method
    plt.figure(figsize=(18, 5))

    for i, (name, embedding) in enumerate(embeddings.items(), 1):
        plt.subplot(1, len(embeddings), i)
        plt.scatter(embedding[:, 0], embedding[:, 1],
                    c=np.arange(len(embedding)),
                    cmap='viridis',
                    alpha=0.8,
                    s=30)
        plt.title(f'{name}\nExecution Time: {manifold_results["execution_times"][name]:.2f}s')
        plt.colorbar(label='Sample Index')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f'Manifold Learning Embeddings Comparison: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_path}/manifold_embeddings_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()




def compare_dim_reduction(pca_results, ica_results, rp_results, dataset_name, save_path='results/dim_reduction'):
    """
    Compares dimensionality reduction techniques.
    """
    # Create comparison table
    comparison_data = {
        'Algorithm': ['PCA', 'ICA', 'RP'],
        'Optimal Components': [
            pca_results['n_components_threshold'],
            ica_results['optimal_components'],
            rp_results['optimal_components']
        ],
        'Execution Time (seconds)': [
            pca_results['execution_time'],
            ica_results['final_execution_time'],
            rp_results['final_execution_time']
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison to CSV
    comparison_df.to_csv(f'{save_path}/dim_reduction_comparison_{dataset_name.replace(" ", "_").lower()}.csv',
                         index=False)

    # Create a visualization to compare execution times
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ['PCA', 'ICA', 'RP'],
        [pca_results['execution_time'], ica_results['final_execution_time'], rp_results['final_execution_time']],
        color=['blue', 'purple', 'green']
    )

    # Add time labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}s', ha='center', va='bottom')

    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Dimensionality Reduction Execution Time Comparison: {dataset_name}')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'{save_path}/dim_reduction_time_comparison_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create t-SNE visualization of the reduced data
    plt.figure(figsize=(15, 5))

    # Get reduced data from each method
    X_pca = pca_results['X_reduced'][:, :pca_results['n_components_threshold']]
    X_ica = ica_results['X_reduced']
    X_rp = rp_results['X_reduced']

    # Apply t-SNE to each
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)

    plt.subplot(1, 3, 1)
    X_pca_tsne = tsne.fit_transform(X_pca)
    plt.scatter(X_pca_tsne[:, 0], X_pca_tsne[:, 1], alpha=0.7, s=20)
    plt.title(f'PCA ({pca_results["n_components_threshold"]} components)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.subplot(1, 3, 2)
    X_ica_tsne = tsne.fit_transform(X_ica)
    plt.scatter(X_ica_tsne[:, 0], X_ica_tsne[:, 1], alpha=0.7, s=20, color='purple')
    plt.title(f'ICA ({ica_results["optimal_components"]} components)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.subplot(1, 3, 3)
    X_rp_tsne = tsne.fit_transform(X_rp)
    plt.scatter(X_rp_tsne[:, 0], X_rp_tsne[:, 1], alpha=0.7, s=20, color='green')
    plt.title(f'RP ({rp_results["optimal_components"]} components)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.tight_layout()
    plt.suptitle(f't-SNE Visualization of Reduced Data: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_path}/dim_reduction_tsne_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return comparison_df


# ==========================================================================
# CLUSTERING AFTER DIMENSIONALITY REDUCTION
# ==========================================================================

def cluster_after_reduction(X_pca, X_ica, X_rp, kmeans_k, em_k, dataset_name,
                            original_kmeans_labels, original_em_labels, save_path='results/combined'):
    """
    Applies clustering algorithms to dimensionally reduced data and compares to original clustering.
    """
    # Dictionary to store results
    results = {}

    # Apply K-Means to each reduced dataset
    kmeans_pca = KMeans(n_clusters=kmeans_k, random_state=RANDOM_SEED)
    kmeans_ica = KMeans(n_clusters=kmeans_k, random_state=RANDOM_SEED)
    kmeans_rp = KMeans(n_clusters=kmeans_k, random_state=RANDOM_SEED)

    start_time = time.time()
    kmeans_pca_labels = kmeans_pca.fit_predict(X_pca)
    kmeans_pca_time = time.time() - start_time

    start_time = time.time()
    kmeans_ica_labels = kmeans_ica.fit_predict(X_ica)
    kmeans_ica_time = time.time() - start_time

    start_time = time.time()
    kmeans_rp_labels = kmeans_rp.fit_predict(X_rp)
    kmeans_rp_time = time.time() - start_time

    # Apply EM to each reduced dataset
    em_pca = GaussianMixture(n_components=em_k, random_state=RANDOM_SEED)
    em_ica = GaussianMixture(n_components=em_k, random_state=RANDOM_SEED)
    em_rp = GaussianMixture(n_components=em_k, random_state=RANDOM_SEED)

    start_time = time.time()
    em_pca.fit(X_pca)
    em_pca_labels = em_pca.predict(X_pca)
    em_pca_time = time.time() - start_time

    start_time = time.time()
    em_ica.fit(X_ica)
    em_ica_labels = em_ica.predict(X_ica)
    em_ica_time = time.time() - start_time

    start_time = time.time()
    em_rp.fit(X_rp)
    em_rp_labels = em_rp.predict(X_rp)
    em_rp_time = time.time() - start_time

    # Calculate silhouette scores
    kmeans_pca_silhouette = silhouette_score(X_pca, kmeans_pca_labels)
    kmeans_ica_silhouette = silhouette_score(X_ica, kmeans_ica_labels)
    kmeans_rp_silhouette = silhouette_score(X_rp, kmeans_rp_labels)

    em_pca_silhouette = silhouette_score(X_pca, em_pca_labels)
    em_ica_silhouette = silhouette_score(X_ica, em_ica_labels)
    em_rp_silhouette = silhouette_score(X_rp, em_rp_labels)

    # Calculate NMI scores (comparing with original clustering)

    # K-Means NMI scores
    kmeans_pca_nmi = normalized_mutual_info_score(original_kmeans_labels, kmeans_pca_labels)
    kmeans_ica_nmi = normalized_mutual_info_score(original_kmeans_labels, kmeans_ica_labels)
    kmeans_rp_nmi = normalized_mutual_info_score(original_kmeans_labels, kmeans_rp_labels)

    # EM NMI scores
    em_pca_nmi = normalized_mutual_info_score(original_em_labels, em_pca_labels)
    em_ica_nmi = normalized_mutual_info_score(original_em_labels, em_ica_labels)
    em_rp_nmi = normalized_mutual_info_score(original_em_labels, em_rp_labels)

    # Store results with consistent key naming
    results['kmeans_pca'] = {
        'labels': kmeans_pca_labels,
        'silhouette': kmeans_pca_silhouette,
        'execution_time': kmeans_pca_time,
        'nmi': kmeans_pca_nmi,
        'model': kmeans_pca
    }

    results['kmeans_ica'] = {
        'labels': kmeans_ica_labels,
        'silhouette': kmeans_ica_silhouette,
        'execution_time': kmeans_ica_time,
        'nmi': kmeans_ica_nmi,
        'model': kmeans_ica
    }

    results['kmeans_rp'] = {
        'labels': kmeans_rp_labels,
        'silhouette': kmeans_rp_silhouette,
        'execution_time': kmeans_rp_time,
        'nmi': kmeans_rp_nmi,
        'model': kmeans_rp
    }

    results['em_pca'] = {
        'labels': em_pca_labels,
        'silhouette': em_pca_silhouette,
        'execution_time': em_pca_time,
        'nmi': em_pca_nmi,
        'model': em_pca
    }

    results['em_ica'] = {
        'labels': em_ica_labels,
        'silhouette': em_ica_silhouette,
        'execution_time': em_ica_time,
        'nmi': em_ica_nmi,
        'model': em_ica
    }

    results['em_rp'] = {
        'labels': em_rp_labels,
        'silhouette': em_rp_silhouette,
        'execution_time': em_rp_time,
        'nmi': em_rp_nmi,
        'model': em_rp
    }

    # Create comparison table
    comparison_data = {
        'Algorithm': ['K-Means + PCA', 'K-Means + ICA', 'K-Means + RP', 'EM + PCA', 'EM + ICA', 'EM + RP'],
        'Silhouette Score': [
            kmeans_pca_silhouette, kmeans_ica_silhouette, kmeans_rp_silhouette,
            em_pca_silhouette, em_ica_silhouette, em_rp_silhouette
        ],
        'NMI Score': [
            kmeans_pca_nmi, kmeans_ica_nmi, kmeans_rp_nmi,
            em_pca_nmi, em_ica_nmi, em_rp_nmi
        ],
        'Execution Time (s)': [
            kmeans_pca_time, kmeans_ica_time, kmeans_rp_time,
            em_pca_time, em_ica_time, em_rp_time
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by silhouette score
    comparison_df = comparison_df.sort_values('Silhouette Score', ascending=False).reset_index(drop=True)

    # Save comparison to CSV
    comparison_df.to_csv(f'{save_path}/clustering_after_reduction_{dataset_name.replace(" ", "_").lower()}.csv',
                         index=False)

    # Create visualization
    plt.figure(figsize=(15, 6))

    # Silhouette scores
    plt.subplot(1, 3, 1)
    bars = plt.bar(comparison_df['Algorithm'], comparison_df['Silhouette Score'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Comparison')

    # Add score labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # NMI scores
    plt.subplot(1, 3, 2)
    bars = plt.bar(comparison_df['Algorithm'], comparison_df['NMI Score'], color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('NMI Score')
    plt.title('NMI Score Comparison')

    # Add score labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Execution times
    plt.subplot(1, 3, 3)
    bars = plt.bar(comparison_df['Algorithm'], comparison_df['Execution Time (s)'], color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')

    # Add time labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.suptitle(f'Clustering After Dimensionality Reduction: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_path}/clustering_after_reduction_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize best and worst combinations
    best_combo = comparison_df.iloc[0]['Algorithm']
    worst_combo = comparison_df.iloc[-1]['Algorithm']

    # Create a direct mapping between display names and dictionary keys
    # This fixes the key error issue by providing explicit mapping
    key_mapping = {
        'K-Means + PCA': 'kmeans_pca',
        'K-Means + ICA': 'kmeans_ica',
        'K-Means + RP': 'kmeans_rp',
        'EM + PCA': 'em_pca',
        'EM + ICA': 'em_ica',
        'EM + RP': 'em_rp'
    }

    # Use the mapping to get the correct keys
    best_key = key_mapping[best_combo]
    worst_key = key_mapping[worst_combo]

    best_labels = results[best_key]['labels']
    worst_labels = results[worst_key]['labels']

    # Create visualization of clusters
    plt.figure(figsize=(12, 6))

    # Determine which reduced data to use for the best and worst
    best_reduction = best_combo.split(' + ')[1]
    worst_reduction = worst_combo.split(' + ')[1]

    X_best = X_pca if best_reduction == 'PCA' else (X_ica if best_reduction == 'ICA' else X_rp)
    X_worst = X_pca if worst_reduction == 'PCA' else (X_ica if worst_reduction == 'ICA' else X_rp)

    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)

    plt.subplot(1, 2, 1)
    X_best_tsne = tsne.fit_transform(X_best)
    plt.scatter(X_best_tsne[:, 0], X_best_tsne[:, 1], c=best_labels, cmap='viridis', alpha=0.7, s=20)
    plt.title(
        f'Best: {best_combo}\nSilhouette: {comparison_df.iloc[0]["Silhouette Score"]:.3f}, NMI: {comparison_df.iloc[0]["NMI Score"]:.3f}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.subplot(1, 2, 2)
    X_worst_tsne = tsne.fit_transform(X_worst)
    plt.scatter(X_worst_tsne[:, 0], X_worst_tsne[:, 1], c=worst_labels, cmap='viridis', alpha=0.7, s=20)
    plt.title(
        f'Worst: {worst_combo}\nSilhouette: {comparison_df.iloc[-1]["Silhouette Score"]:.3f}, NMI: {comparison_df.iloc[-1]["NMI Score"]:.3f}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.tight_layout()
    plt.suptitle(f'Best vs Worst Clustering After Reduction: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{save_path}/best_worst_clustering_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return results, comparison_df


# ==========================================================================
# NEURAL NETWORK WITH DIMENSIONALITY REDUCTION
# ==========================================================================

def run_neural_network(X_train, X_test, y_train, y_test, params=None, verbose=True, cv_folds=5):
    """
    Trains and evaluates a neural network with cross-validation.

    This function implements a more rigorous approach to neural network evaluation:
    1. Cross-validation to estimate performance variance
    2. Multiple metrics (accuracy, F1, training time)
    3. Statistical confidence through CV

    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature sets
    y_train, y_test : array-like
        Training and test target values
    params : dict, optional
        Neural network hyperparameters
    verbose : bool, default=True
        Whether to print detailed results
    cv_folds : int, default=5
        Number of cross-validation folds

    Returns:
    --------
    dict
        Dictionary containing model, predictions, and performance metrics
    """
    if params is None:
        params = NN_PARAMS

    # Create model
    model = MLPClassifier(**params)

    # Cross-validation for statistical significance
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calculate 95% confidence interval from CV
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    confidence_interval = (cv_mean - 1.96 * cv_std / np.sqrt(cv_folds),
                           cv_mean + 1.96 * cv_std / np.sqrt(cv_folds))

    if verbose:
        summary_table = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["F1 Score", f"{f1:.4f}"],
            ["CV Accuracy (mean)", f"{cv_mean:.4f}"],
            ["CV Accuracy (std)", f"{cv_std:.4f}"],
            ["95% Confidence Interval", f"({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})"],
            ["Training Time", f"{training_time:.2f} seconds"],
            ["Feature Count", X_train.shape[1]]
        ]
        print_table(summary_table, ["Metric", "Value"], "Neural Network Performance")

    return {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confidence_interval': confidence_interval,
        'training_time': training_time,
        'y_pred': y_pred
    }


def perform_statistical_comparison(baseline_results, comparison_models, alpha=0.05):
    """
    Performs statistical tests to compare neural network models with the baseline.
    """
    from scipy import stats

    # Check for required keys in baseline_results
    required_baseline_keys = ['y_pred', 'y_true', 'accuracy']
    for key in required_baseline_keys:
        if key not in baseline_results:
            print(f"Warning: Missing required key '{key}' in baseline results")
            if key == 'y_true' or key == 'y_pred':
                print("Statistical testing cannot proceed without predictions and ground truth")
                return {}
            # For accuracy, we could default it to 0
            if key == 'accuracy':
                baseline_results['accuracy'] = 0
                print("Defaulting baseline accuracy to 0")

    baseline_preds = baseline_results['y_pred']
    y_test = baseline_results['y_true']

    # Initialize results dictionary
    stat_results = {}

    # Print table header
    print("\nStatistical Comparison with Baseline")
    table_data = []

    # Filter out models with missing data
    valid_comparison_models = {}
    for model_name, model_results in comparison_models.items():
        # Check for required keys in comparison model results
        missing_keys = []
        for key in ['y_pred', 'accuracy']:
            if key not in model_results:
                missing_keys.append(key)

        if missing_keys:
            print(f"Warning: Model '{model_name}' is missing required keys: {missing_keys}")
            print(f"Skipping statistical comparison for {model_name}")
            continue

        valid_comparison_models[model_name] = model_results

    if not valid_comparison_models:
        print("No valid comparison models with required data. Statistical testing aborted.")
        return {}

    for model_name, model_results in valid_comparison_models.items():
        model_preds = model_results['y_pred']

        # Create contingency table for McNemar's test
        baseline_correct = (baseline_preds == y_test)
        model_correct = (model_preds == y_test)

        # Calculate each cell of the 2x2 contingency table
        both_correct = np.sum(baseline_correct & model_correct)  # n11
        baseline_only = np.sum(baseline_correct & ~model_correct)  # n10
        model_only = np.sum(~baseline_correct & model_correct)  # n01
        both_wrong = np.sum(~baseline_correct & ~model_correct)  # n00

        # Check if we can perform the test (need at least some disagreement)
        if baseline_only + model_only == 0:
            print(f"Warning: No disagreement between baseline and {model_name}. Cannot perform McNemar's test.")
            stat_results[model_name] = {
                'test_name': 'None',
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'model_advantage': False,
                'contingency_table': {
                    'both_correct': both_correct,
                    'baseline_only': baseline_only,
                    'model_only': model_only,
                    'both_wrong': both_wrong
                }
            }

            # Add to table data for display
            accuracy_diff = model_results['accuracy'] - baseline_results['accuracy']
            comparison_text = f"{baseline_results['accuracy']:.4f} vs {model_results['accuracy']:.4f} ({accuracy_diff:+.4f})"

            table_data.append([
                model_name,
                comparison_text,
                "N/A",
                "N/A",
                "No",
                "No difference"
            ])
            continue

        # Determine which version of McNemar's test to use based on sample size
        try:
            if min(baseline_only, model_only) < 5:
                # Use exact binomial test version (more accurate for small counts)
                statistic = baseline_only + model_only
                p_value = 2 * stats.binom.cdf(min(baseline_only, model_only),
                                              n=baseline_only + model_only, p=0.5)
                test_name = "McNemar's Exact"
            else:
                # Use chi-square version with continuity correction
                statistic = (np.abs(baseline_only - model_only) - 1) ** 2 / (baseline_only + model_only)
                p_value = 1 - stats.chi2.cdf(statistic, df=1)  # chi-square with 1 degree of freedom
                test_name = "McNemar's Chi-square"

            # Determine if difference is statistically significant at alpha level
            significant = p_value < alpha

            # Determine which model performed better (if difference is significant)
            model_advantage = model_only > baseline_only
        except Exception as e:
            # Handle any unexpected errors in statistical testing
            print(f"Error performing statistical test for {model_name}: {e}")
            statistic = 0
            p_value = 1.0
            significant = False
            model_advantage = False
            test_name = "Error"

        # Store detailed results for further analysis
        stat_results[model_name] = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'model_advantage': model_advantage,
            'contingency_table': {
                'both_correct': both_correct,
                'baseline_only': baseline_only,
                'model_only': model_only,
                'both_wrong': both_wrong
            }
        }

        # Format model comparison for table display
        accuracy_diff = model_results['accuracy'] - baseline_results['accuracy']
        comparison_text = f"{baseline_results['accuracy']:.4f} vs {model_results['accuracy']:.4f} ({accuracy_diff:+.4f})"

        # Add to table data for display
        table_data.append([
            model_name,
            comparison_text,
            f"{test_name}",
            f"{p_value:.4f}" if test_name != "Error" and test_name != "N/A" else "N/A",
            "Yes" if significant else "No",
            f"{model_name}" if model_advantage and significant else
            "Baseline" if significant else "No difference"
        ])

    # Print table with results if there are any valid comparisons
    if table_data:
        headers = ["Model", "Accuracy Comparison", "Test", "p-value", "Significant", "Better Model"]
        print_table(table_data, headers, "Statistical Significance Testing")

    return stat_results

def nn_with_dimensionality_reduction(X_train, X_test, y_train, y_test,
                                     pca_results, ica_results, rp_results,
                                     dataset_name, save_path='results/neural_net'):
    """
    Trains and evaluates neural networks with dimensionally reduced data.
    """
    # Get optimally reduced datasets
    X_train_pca = pca_results['model'].transform(X_train)[:, :pca_results['n_components_threshold']]
    X_test_pca = pca_results['model'].transform(X_test)[:, :pca_results['n_components_threshold']]

    X_train_ica = ica_results['model'].transform(X_train)
    X_test_ica = ica_results['model'].transform(X_test)

    X_train_rp = rp_results['model'].transform(X_train)
    X_test_rp = rp_results['model'].transform(X_test)

    # Train baseline model
    print("\nTraining baseline neural network...")
    baseline_results = run_neural_network(X_train, X_test, y_train, y_test)

    # Train model with PCA
    print("\nTraining neural network with PCA-reduced data...")
    pca_nn_results = run_neural_network(X_train_pca, X_test_pca, y_train, y_test)

    # Train model with ICA
    print("\nTraining neural network with ICA-reduced data...")
    ica_nn_results = run_neural_network(X_train_ica, X_test_ica, y_train, y_test)

    # Train model with RP
    print("\nTraining neural network with RP-reduced data...")
    rp_nn_results = run_neural_network(X_train_rp, X_test_rp, y_train, y_test)

    # Create comparison table
    comparison_data = {
        'Model': ['Baseline (No Reduction)', 'NN + PCA', 'NN + ICA', 'NN + RP'],
        'Accuracy': [
            baseline_results['accuracy'],
            pca_nn_results['accuracy'],
            ica_nn_results['accuracy'],
            rp_nn_results['accuracy']
        ],
        'F1 Score': [
            baseline_results['f1_score'],
            pca_nn_results['f1_score'],
            ica_nn_results['f1_score'],
            rp_nn_results['f1_score']
        ],
        'Training Time (seconds)': [
            baseline_results['training_time'],
            pca_nn_results['training_time'],
            ica_nn_results['training_time'],
            rp_nn_results['training_time']
        ],
        'Number of Features': [
            X_train.shape[1],
            X_train_pca.shape[1],
            X_train_ica.shape[1],
            X_train_rp.shape[1]
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison to CSV
    comparison_df.to_csv(f'{save_path}/nn_reduction_comparison_{dataset_name.replace(" ", "_").lower()}.csv',
                         index=False)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Accuracy
    plt.subplot(2, 2, 1)
    bars = plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')

    # Add accuracy labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    # F1 Score
    plt.subplot(2, 2, 2)
    bars = plt.bar(comparison_df['Model'], comparison_df['F1 Score'], color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison')

    # Add F1 labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    # Training Time
    plt.subplot(2, 2, 3)
    bars = plt.bar(comparison_df['Model'], comparison_df['Training Time (seconds)'], color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')

    # Add time labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}s', ha='center', va='bottom')

    # Feature count
    plt.subplot(2, 2, 4)
    bars = plt.bar(comparison_df['Model'], comparison_df['Number of Features'], color='mediumpurple')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Features')
    plt.title('Feature Dimensionality')

    # Add feature count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.suptitle(f'Neural Network Performance with Dimensionality Reduction: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.savefig(f'{save_path}/nn_reduction_comparison_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot learning curves for each model
    models = {
        'Baseline': (baseline_results['model'], X_train, X_test),
        'NN + PCA': (pca_nn_results['model'], X_train_pca, X_test_pca),
        'NN + ICA': (ica_nn_results['model'], X_train_ica, X_test_ica),
        'NN + RP': (rp_nn_results['model'], X_train_rp, X_test_rp)
    }

    # Create learning curves
    plt.figure(figsize=(12, 10))

    for i, (name, (model, X_tr, X_te)) in enumerate(models.items(), 1):
        plt.subplot(2, 2, i)

        train_sizes, train_scores, test_scores = learning_curve(
            model, X_tr, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3, scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Validation Accuracy')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')

        plt.title(f'Learning Curve: {name}')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f'Neural Network Learning Curves: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.savefig(f'{save_path}/nn_learning_curves_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Store true labels for statistical testing
    baseline_results['y_true'] = y_test
    pca_nn_results['y_true'] = y_test
    ica_nn_results['y_true'] = y_test
    rp_nn_results['y_true'] = y_test

    # Perform statistical testing
    print("\nPerforming statistical testing on dimensionality reduction models...")
    comparison_models = {
        'NN + PCA': pca_nn_results,
        'NN + ICA': ica_nn_results,
        'NN + RP': rp_nn_results
    }
    statistical_results = perform_statistical_comparison(baseline_results, comparison_models)

    # Add statistical results to the return dictionary
    return {
        'baseline': baseline_results,
        'pca': pca_nn_results,
        'ica': ica_nn_results,
        'rp': rp_nn_results,
        'comparison': comparison_df,
        'statistical_results': statistical_results
    }

# ==========================================================================
# NEURAL NETWORK WITH CLUSTERING FEATURES
# ==========================================================================

def nn_with_clustering_features(X_train, X_test, y_train, y_test,
                                kmeans_results, em_results,
                                dataset_name, save_path='results/neural_net'):
    """
    Trains neural networks with cluster assignments as additional features.
    """
    # Get the best K from clustering results
    kmeans_k = kmeans_results['best_k_silhouette']
    em_k = em_results['best_k_bic']

    # Create cluster models with the best K
    kmeans = KMeans(n_clusters=kmeans_k, random_state=RANDOM_SEED)
    em = GaussianMixture(n_components=em_k, random_state=RANDOM_SEED)

    # Train clustering models on training data
    kmeans.fit(X_train)
    em.fit(X_train)

    # Get cluster assignments
    kmeans_train_labels = kmeans.predict(X_train)
    kmeans_test_labels = kmeans.predict(X_test)

    em_train_labels = em.predict(X_train)
    em_test_labels = em.predict(X_test)

    # One-hot encode cluster assignments
    # Updated to use sparse_output instead of sparse for scikit-learn 1.0+
    onehot_encoder = OneHotEncoder(sparse_output=False)

    kmeans_train_onehot = onehot_encoder.fit_transform(kmeans_train_labels.reshape(-1, 1))
    kmeans_test_onehot = onehot_encoder.transform(kmeans_test_labels.reshape(-1, 1))

    # Create a fresh encoder for EM clusters to avoid category mismatches
    onehot_encoder_em = OneHotEncoder(sparse_output=False)
    em_train_onehot = onehot_encoder_em.fit_transform(em_train_labels.reshape(-1, 1))
    em_test_onehot = onehot_encoder_em.transform(em_test_labels.reshape(-1, 1))

    # Combine original features with cluster assignments
    X_train_kmeans = np.hstack((X_train, kmeans_train_onehot))
    X_test_kmeans = np.hstack((X_test, kmeans_test_onehot))

    X_train_em = np.hstack((X_train, em_train_onehot))
    X_test_em = np.hstack((X_test, em_test_onehot))

    # Train baseline model
    print("\nTraining baseline neural network...")
    baseline_results = run_neural_network(X_train, X_test, y_train, y_test)

    # Train model with K-Means features
    print("\nTraining neural network with K-Means cluster features...")
    kmeans_nn_results = run_neural_network(X_train_kmeans, X_test_kmeans, y_train, y_test)

    # Train model with EM features
    print("\nTraining neural network with EM cluster features...")
    em_nn_results = run_neural_network(X_train_em, X_test_em, y_train, y_test)

    # Create comparison table
    comparison_data = {
        'Model': ['Baseline (No Clustering)', 'NN + K-Means Clusters', 'NN + EM Clusters'],
        'Accuracy': [
            baseline_results['accuracy'],
            kmeans_nn_results['accuracy'],
            em_nn_results['accuracy']
        ],
        'F1 Score': [
            baseline_results['f1_score'],
            kmeans_nn_results['f1_score'],
            em_nn_results['f1_score']
        ],
        'Training Time (seconds)': [
            baseline_results['training_time'],
            kmeans_nn_results['training_time'],
            em_nn_results['training_time']
        ],
        'Number of Features': [
            X_train.shape[1],
            X_train_kmeans.shape[1],
            X_train_em.shape[1]
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison to CSV
    comparison_df.to_csv(f'{save_path}/nn_clustering_comparison_{dataset_name.replace(" ", "_").lower()}.csv',
                         index=False)

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Accuracy
    plt.subplot(2, 2, 1)
    bars = plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')

    # Add accuracy labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    # F1 Score
    plt.subplot(2, 2, 2)
    bars = plt.bar(comparison_df['Model'], comparison_df['F1 Score'], color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison')

    # Add F1 labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    # Training Time
    plt.subplot(2, 2, 3)
    bars = plt.bar(comparison_df['Model'], comparison_df['Training Time (seconds)'], color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')

    # Add time labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}s', ha='center', va='bottom')

    # Feature count
    plt.subplot(2, 2, 4)
    bars = plt.bar(comparison_df['Model'], comparison_df['Number of Features'], color='mediumpurple')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Features')
    plt.title('Feature Dimensionality')

    # Add feature count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.suptitle(f'Neural Network Performance with Clustering Features: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.savefig(f'{save_path}/nn_clustering_comparison_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot feature importance for each model
    models = {
        'Baseline': (baseline_results['model'], X_train, X_test),
        'NN + K-Means': (kmeans_nn_results['model'], X_train_kmeans, X_test_kmeans),
        'NN + EM': (em_nn_results['model'], X_train_em, X_test_em)
    }

    # Analyze feature importance using permutation
    plt.figure(figsize=(15, 10))

    for i, (name, (model, X_tr, X_te)) in enumerate(models.items(), 1):
        plt.subplot(1, 3, i)

        # Limit to top 15 features for visualization
        n_features = min(15, X_tr.shape[1])

        # Use permutation importance
        perm_importance = permutation_importance(model, X_te, y_test, n_repeats=10, random_state=RANDOM_SEED)
        sorted_idx = perm_importance.importances_mean.argsort()[-n_features:]

        # Create feature names
        if name == 'Baseline':
            feature_names = [f'F{j}' for j in range(X_tr.shape[1])]
        elif name == 'NN + K-Means':
            original_features = [f'F{j}' for j in range(X_train.shape[1])]
            cluster_features = [f'KM{j}' for j in range(kmeans_train_onehot.shape[1])]
            feature_names = original_features + cluster_features
        else:  # NN + EM
            original_features = [f'F{j}' for j in range(X_train.shape[1])]
            cluster_features = [f'EM{j}' for j in range(em_train_onehot.shape[1])]
            feature_names = original_features + cluster_features

        # Select only the top features for plotting
        selected_features = [feature_names[j] for j in sorted_idx]
        selected_importances = perm_importance.importances_mean[sorted_idx]

        # Create horizontal bar chart
        plt.barh(range(n_features), selected_importances, align='center')
        plt.yticks(range(n_features), selected_features)
        plt.title(f'Feature Importance: {name}')
        plt.xlabel('Permutation Importance')

    plt.tight_layout()
    plt.suptitle(f'Feature Importance Analysis: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.savefig(f'{save_path}/feature_importance_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Store true labels for statistical testing
    baseline_results['y_true'] = y_test
    kmeans_nn_results['y_true'] = y_test
    em_nn_results['y_true'] = y_test

    # Perform statistical testing
    print("\nPerforming statistical testing on clustering feature models...")
    comparison_models = {
        'NN + K-Means': kmeans_nn_results,
        'NN + EM': em_nn_results
    }
    statistical_results = perform_statistical_comparison(baseline_results, comparison_models)

    # Add statistical results to the return dictionary
    return {
        'baseline': baseline_results,
        'kmeans': kmeans_nn_results,
        'em': em_nn_results,
        'comparison': comparison_df,
        'statistical_results': statistical_results
    }


# ==========================================================================
# SUMMARY TABLES
# ==========================================================================

def print_table(data, headers, title=None):
    """
    Prints a formatted table with the given data, headers, and optional title.

    Parameters:
    -----------
    data : list of lists
        The data rows to be displayed in the table
    headers : list
        The column headers for the table
    title : str, optional
        An optional title to display above the table
    """
    if title:
        print(f"\n{title}")
    print(tabulate(data, headers=headers, tablefmt="pretty"))
    print()


def print_section_header(title, level=1):
    """
    Prints a formatted section header.

    Parameters:
    -----------
    title : str
        The section title
    level : int, default=1
        The header level (1 for main sections, 2 for subsections)
    """
    if level == 1:
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)
    else:
        print("\n" + "-" * 60)
        print(f"{title}")
        print("-" * 60)


def create_summary_tables(all_results):
    """
    Creates and prints comprehensive summary tables from all results.

    Parameters:
    -----------
    all_results : dict
        Dictionary containing all results from different experiments
    """
    # Print overall section header
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    # 1. Clustering Summary
    clustering_data = []
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        kmeans_results = all_results[f'{dataset_lower}_kmeans']
        em_results = all_results[f'{dataset_lower}_em']

        clustering_data.append([
            dataset,
            "K-Means",
            kmeans_results['best_k_silhouette'],
            max(kmeans_results['silhouette_scores']),
            np.mean(kmeans_results['execution_times'])
        ])

        clustering_data.append([
            dataset,
            "EM (GMM)",
            em_results['best_k_bic'],
            np.nan,  # No silhouette score directly available
            np.mean(em_results['execution_times'])
        ])

    print_table(
        clustering_data,
        ["Dataset", "Algorithm", "Optimal K", "Best Silhouette", "Avg. Execution Time (s)"],
        "Clustering Summary"
    )

    # 2. Dimensionality Reduction Summary
    dr_data = []
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        pca_results = all_results[f'{dataset_lower}_pca']
        ica_results = all_results[f'{dataset_lower}_ica']
        rp_results = all_results[f'{dataset_lower}_rp']

        original_dims = max(
            pca_results['explained_variance'].shape[0],
            len(ica_results['components_range']),
            len(rp_results['components_range'])
        )

        dr_data.append([
            dataset,
            "PCA",
            pca_results['n_components_threshold'],
            f"{(original_dims - pca_results['n_components_threshold']) / original_dims * 100:.1f}%",
            pca_results['execution_time']
        ])

        dr_data.append([
            dataset,
            "ICA",
            ica_results['optimal_components'],
            f"{(original_dims - ica_results['optimal_components']) / original_dims * 100:.1f}%",
            ica_results['final_execution_time']
        ])

        dr_data.append([
            dataset,
            "RP",
            rp_results['optimal_components'],
            f"{(original_dims - rp_results['optimal_components']) / original_dims * 100:.1f}%",
            rp_results['final_execution_time']
        ])

    print_table(
        dr_data,
        ["Dataset", "Algorithm", "Optimal Components", "Dimension Reduction", "Execution Time (s)"],
        "Dimensionality Reduction Summary"
    )

    # 3. Neural Network Performance Summary
    if 'marketing_nn_dr' in all_results:
        nn_data = []

        # Baseline
        nn_data.append([
            "Marketing",
            "Baseline",
            all_results['marketing_nn_dr']['baseline']['accuracy'],
            all_results['marketing_nn_dr']['baseline']['f1_score'],
            all_results['marketing_nn_dr']['baseline']['training_time']
        ])

        # With dimensionality reduction
        for dr_method in ['PCA', 'ICA', 'RP']:
            nn_data.append([
                "Marketing",
                f"NN + {dr_method}",
                all_results['marketing_nn_dr'][dr_method.lower()]['accuracy'],
                all_results['marketing_nn_dr'][dr_method.lower()]['f1_score'],
                all_results['marketing_nn_dr'][dr_method.lower()]['training_time']
            ])

        # With clustering features
        for cluster_method in ['kmeans', 'em']:
            nn_data.append([
                "Marketing",
                f"NN + {cluster_method.upper()}",
                all_results['marketing_nn_cluster'][cluster_method]['accuracy'],
                all_results['marketing_nn_cluster'][cluster_method]['f1_score'],
                all_results['marketing_nn_cluster'][cluster_method]['training_time']
            ])

        print_table(
            nn_data,
            ["Dataset", "Model", "Accuracy", "F1 Score", "Training Time (s)"],
            "Neural Network Performance Summary"
        )

    # 4. Wall Clock Times Summary
    time_data = []

    # Clustering times - Use consistent key naming
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        # Use consistent key names that match how they were stored
        kmeans_key = f"{dataset_lower}_kmeans"
        em_key = f"{dataset_lower}_em"

        if kmeans_key in all_results:
            exec_times = all_results[kmeans_key]['execution_times']
            time_data.append([
                "Clustering",
                dataset,
                "K-Means",
                np.mean(exec_times),
                np.min(exec_times),
                np.max(exec_times)
            ])

        if em_key in all_results:
            exec_times = all_results[em_key]['execution_times']
            time_data.append([
                "Clustering",
                dataset,
                "EM (GMM)",
                np.mean(exec_times),
                np.min(exec_times),
                np.max(exec_times)
            ])

    # Dimensionality reduction times
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        for algorithm, key in [("PCA", "execution_time"), ("ICA", "final_execution_time"),
                               ("RP", "final_execution_time")]:
            algo_lower = algorithm.lower()
            results_key = f"{dataset_lower}_{algo_lower}"
            if results_key in all_results:
                exec_time = all_results[results_key][key]
                time_data.append([
                    "Dim. Reduction",
                    dataset,
                    algorithm,
                    exec_time,
                    exec_time,
                    exec_time
                ])

    # Neural network times
    if 'marketing_nn_dr' in all_results:
        for model_type in ['baseline', 'pca', 'ica', 'rp']:
            time_data.append([
                "Neural Network",
                "Marketing",
                f"NN + {model_type.upper() if model_type != 'baseline' else 'Baseline'}",
                all_results['marketing_nn_dr'][model_type]['training_time'],
                all_results['marketing_nn_dr'][model_type]['training_time'],
                all_results['marketing_nn_dr'][model_type]['training_time']
            ])

        for model_type in ['kmeans', 'em']:
            time_data.append([
                "Neural Network",
                "Marketing",
                f"NN + {model_type.upper()}",
                all_results['marketing_nn_cluster'][model_type]['training_time'],
                all_results['marketing_nn_cluster'][model_type]['training_time'],
                all_results['marketing_nn_cluster'][model_type]['training_time']
            ])

    # Add manifold learning times if available
    if 'manifold_results' in all_results and 'execution_times' in all_results['manifold_results']:
        for method, exec_time in all_results['manifold_results']['execution_times'].items():
            time_data.append([
                "Manifold Learning",
                "Marketing",
                method,
                exec_time,
                exec_time,
                exec_time
            ])

    print_table(
        time_data,
        ["Category", "Dataset", "Algorithm", "Avg. Time (s)", "Min Time (s)", "Max Time (s)"],
        "Wall Clock Times Summary"
    )

    # 5. Extra Credit Summary (if available)
    if 'manifold_results' in all_results:
        # 5.1 Manifold Execution Times
        manifold_data = []

        for method, exec_time in all_results['manifold_results']['execution_times'].items():
            manifold_data.append([
                "Marketing",
                method,
                exec_time
            ])

        print_table(
            manifold_data,
            ["Dataset", "Method", "Execution Time (s)"],
            "Manifold Learning Summary (Extra Credit)"
        )

        # 5.2 Clustering on Manifolds Performance
        if 'manifold_clustering_comparison' in all_results:
            comparison_df = all_results['manifold_clustering_comparison']
            if isinstance(comparison_df, pd.DataFrame):
                # Convert DataFrame to list for tabulate
                headers = ["Embedding", "K-Means Silhouette", "EM Silhouette", "Average Silhouette"]
                table_data = []
                for idx, row in comparison_df.iterrows():
                    table_data.append([
                        row['Embedding'],
                        row['K-Means Silhouette'],
                        row['EM Silhouette'],
                        row['Average Silhouette']
                    ])

                print_table(
                    table_data,
                    headers,
                    "Clustering Performance on Manifold Embeddings"
                )

                # 5.3 Best Manifold Method
                if 'Average Silhouette' in comparison_df.columns:
                    best_idx = comparison_df['Average Silhouette'].idxmax()
                    best_method = comparison_df.iloc[best_idx]['Embedding']
                    best_score = comparison_df.iloc[best_idx]['Average Silhouette']

                    best_summary = [
                        ["Best Overall Manifold Method", best_method],
                        ["Average Silhouette Score", f"{best_score:.4f}"],
                        ["K-Means Silhouette", f"{comparison_df.iloc[best_idx]['K-Means Silhouette']:.4f}"],
                        ["EM Silhouette", f"{comparison_df.iloc[best_idx]['EM Silhouette']:.4f}"]
                    ]
                    print_table(best_summary, ["Metric", "Value"], "Best Manifold Learning Method")


def get_best_algorithm_summary(all_results):
    """
    Creates a summary of the best-performing algorithms across tasks.

    Parameters:
    -----------
    all_results : dict
        Dictionary containing all results from different experiments
    """
    best_summary = [
        ["Task", "Dataset", "Best Algorithm", "Metric", "Value"]
    ]

    # Best clustering by silhouette score
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        kmeans_key = f'{dataset_lower}_kmeans'
        em_key = f'{dataset_lower}_clustering_comparison'

        if kmeans_key in all_results:
            kmeans_sil = max(all_results[kmeans_key]['silhouette_scores'])

            if em_key in all_results and isinstance(all_results[em_key], pd.DataFrame) and 'Silhouette Score' in \
                    all_results[em_key].columns:
                em_sil_rows = all_results[em_key][all_results[em_key]['Algorithm'] == 'EM (GMM)']
                if not em_sil_rows.empty:
                    em_sil = em_sil_rows['Silhouette Score'].values[0]
                    best_algo = "K-Means" if kmeans_sil > em_sil else "EM (GMM)"
                    best_value = max(kmeans_sil, em_sil)
                else:
                    best_algo = "K-Means"
                    best_value = kmeans_sil
            else:
                best_algo = "K-Means"
                best_value = kmeans_sil

            best_summary.append([
                "Clustering",
                dataset,
                best_algo,
                "Silhouette Score",
                f"{best_value:.4f}"
            ])

    # Best dimensionality reduction for NN
    nn_dr_key = 'marketing_nn_dr'
    if nn_dr_key in all_results and 'comparison' in all_results[nn_dr_key]:
        nn_comparison = all_results[nn_dr_key]['comparison']
        if not nn_comparison.empty:
            best_idx = nn_comparison['Accuracy'].idxmax()
            best_summary.append([
                "Dim. Reduction for NN",
                "Marketing",
                nn_comparison.loc[best_idx, 'Model'],
                "Accuracy",
                f"{nn_comparison.loc[best_idx, 'Accuracy']:.4f}"
            ])

    # Best clustering for NN
    nn_cluster_key = 'marketing_nn_cluster'
    if nn_cluster_key in all_results and 'comparison' in all_results[nn_cluster_key]:
        cluster_comparison = all_results[nn_cluster_key]['comparison']
        if not cluster_comparison.empty:
            best_idx = cluster_comparison['Accuracy'].idxmax()
            best_summary.append([
                "Clustering for NN",
                "Marketing",
                cluster_comparison.loc[best_idx, 'Model'],
                "Accuracy",
                f"{cluster_comparison.loc[best_idx, 'Accuracy']:.4f}"
            ])

    # Best combined approach (DR + clustering)
    combined_key = 'marketing_combined'
    if combined_key in all_results and isinstance(all_results[combined_key], tuple) and len(
            all_results[combined_key]) > 1:
        combined_comparison = all_results[combined_key][1]  # This is the comparison DataFrame
        if not combined_comparison.empty and 'Silhouette Score' in combined_comparison.columns:
            best_idx = combined_comparison['Silhouette Score'].idxmax()
            best_summary.append([
                "DR + Clustering",
                "Marketing",
                combined_comparison.loc[best_idx, 'Algorithm'],
                "Silhouette Score",
                f"{combined_comparison.loc[best_idx, 'Silhouette Score']:.4f}"
            ])

    if len(best_summary) > 1:  # Only print if we have results
        print_table(best_summary[1:], best_summary[0], "Best Algorithm Summary")


def compare_sensitivity_analyses(pca_sensitivity, ica_sensitivity, rp_sensitivity, dataset_name,
                                 save_path='results/dim_reduction'):
    """
    Compares sensitivity analysis results across algorithms.
    """
    # Extract threshold ranges and component ranges
    comparison_data = []

    # PCA data
    pca_thresholds = sorted(list(pca_sensitivity.keys()))
    pca_components = [pca_sensitivity[t]['n_components'] for t in pca_thresholds]
    pca_reduction = [pca_sensitivity[t]['dimension_reduction'] for t in pca_thresholds]
    comparison_data.append({
        'Algorithm': 'PCA',
        'Threshold Range': f"{min(pca_thresholds):.0%} - {max(pca_thresholds):.0%}",
        'Component Range': f"{min(pca_components)} - {max(pca_components)}",
        'Reduction Range': f"{min(pca_reduction) * 100:.1f}% - {max(pca_reduction) * 100:.1f}%"
    })

    # ICA data
    ica_thresholds = sorted(list(ica_sensitivity.keys()))
    ica_components = [ica_sensitivity[t]['n_components'] for t in ica_thresholds]
    ica_reduction = [ica_sensitivity[t]['dimension_reduction'] for t in ica_thresholds]
    comparison_data.append({
        'Algorithm': 'ICA',
        'Threshold Range': f"{min(ica_thresholds):.1f} - {max(ica_thresholds):.1f}",
        'Component Range': f"{min(ica_components)} - {max(ica_components)}",
        'Reduction Range': f"{min(ica_reduction) * 100:.1f}% - {max(ica_reduction) * 100:.1f}%"
    })

    # RP data
    rp_thresholds = sorted(list(rp_sensitivity.keys()))
    rp_components = [rp_sensitivity[t]['optimal_components'] for t in rp_thresholds]
    rp_reduction = [rp_sensitivity[t]['dimension_reduction'] for t in rp_thresholds]
    comparison_data.append({
        'Algorithm': 'RP',
        'Threshold Range': f"{min(rp_thresholds):.2f} - {max(rp_thresholds):.2f}",
        'Component Range': f"{min(rp_components)} - {max(rp_components)}",
        'Reduction Range': f"{min(rp_reduction) * 100:.1f}% - {max(rp_reduction) * 100:.1f}%"
    })

    # Create DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{save_path}/sensitivity_comparison_{dataset_name.replace(" ", "_").lower()}.csv',
                         index=False)

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Component ranges
    plt.subplot(2, 1, 1)
    x = range(len(comparison_data))
    labels = [row['Algorithm'] for row in comparison_data]
    plt.bar(x, [max([int(r['Component Range'].split(' - ')[1]) for r in comparison_data])], alpha=0.3, width=0.4,
            label='Max Components')
    plt.bar([i + 0.4 for i in x], [min([int(r['Component Range'].split(' - ')[0]) for r in comparison_data])],
            alpha=0.7, width=0.4, label='Min Components')
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel('Number of Components')
    plt.title('Component Range by Algorithm')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}/sensitivity_comparison_{dataset_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()

    return comparison_df


def evaluate_sensitivity_effects_on_nn(X_train, X_test, y_train, y_test,
                                       sensitivity_results, model,
                                       param_name, dataset_name):
    """
    Evaluates how sensitivity parameter changes affect neural network performance.

    Parameters:
    -----------
    X_train, X_test : array-like
        Original training and test data
    y_train, y_test : array-like
        Training and test labels
    sensitivity_results : dict
        Results dictionary from sensitivity analysis
    model : estimator
        Original dimensionality reduction model
    param_name : str
        Name of the parameter being varied (for labeling)
    dataset_name : str
        Name of the dataset (for labeling)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing performance metrics for each parameter value
    """
    performance_results = []

    # Check if sensitivity_results is empty or None
    if not sensitivity_results:
        print(f"Warning: Empty sensitivity results for {param_name}")
        return pd.DataFrame(columns=['Parameter Value', 'Components', 'Accuracy', 'Training Time'])

    # For each parameter value in the sensitivity analysis
    for param_value, results in sensitivity_results.items():
        # Skip if results is None or empty
        if results is None or not results:
            print(f"Warning: Empty results for param_value={param_value}")
            continue

        # Verify required keys exist
        if 'n_components' not in results and 'optimal_components' not in results:
            print(f"Warning: No component count found for param_value={param_value}")
            continue

        # Get the transformed training data
        if 'X_reduced' in results:
            X_train_transformed = results['X_reduced']
        elif 'model' in results:
            # If we have the model but not the reduced data, generate it
            try:
                X_train_transformed = results['model'].transform(X_train)
            except Exception as e:
                print(f"Warning: Could not transform data using stored model for param_value={param_value}: {e}")
                continue
        else:
            print(f"Warning: X_reduced not found for param_value={param_value}")
            # Instead of continuing, let's try to recreate the transformed data
            try:
                # Get component count
                n_components = results.get('n_components', results.get('optimal_components'))
                if n_components is None:
                    print(f"Warning: Could not determine component count for param_value={param_value}")
                    continue

                # Create a new model with the right number of components
                if hasattr(model, 'n_components'):
                    temp_model = clone(model)
                    temp_model.set_params(n_components=n_components)
                    X_train_transformed = temp_model.fit_transform(X_train)
                else:
                    print(f"Warning: Cannot recreate model for param_value={param_value}")
                    continue
            except Exception as e:
                print(f"Error recreating transformed data: {e}")
                continue

        # Get component count for this parameter value
        n_components = results.get('n_components', results.get('optimal_components', X_train_transformed.shape[1]))

        # Transform test data based on the parameter value
        try:
            # If we have a stored model, use it
            if 'model' in results:
                X_test_transformed = results['model'].transform(X_test)
            # Otherwise create a copy of the model with the specific parameter setting
            elif hasattr(model, 'n_components'):
                temp_model = clone(model)
                temp_model.set_params(n_components=n_components)
                temp_model.fit(X_train)  # Fit to training data
                X_test_transformed = temp_model.transform(X_test)
            else:
                # If we can't adjust the model, use the original model
                X_test_transformed = model.transform(X_test)

            # If needed, subset columns to match training data
            if X_test_transformed.shape[1] > X_train_transformed.shape[1]:
                X_test_transformed = X_test_transformed[:, :X_train_transformed.shape[1]]

        except Exception as e:
            print(f"Error transforming test data: {e}")
            continue

        # Train and evaluate neural network
        try:
            nn_results = run_neural_network(X_train_transformed, X_test_transformed,
                                            y_train, y_test, verbose=False)

            # Store performance data
            performance_results.append({
                'Parameter Value': param_value,
                'Components': n_components,
                'Accuracy': nn_results['accuracy'],
                'Training Time': nn_results['training_time']
            })
        except Exception as e:
            print(f"Error in neural network evaluation: {e}")
            continue

    # Create DataFrame for analysis
    df = pd.DataFrame(performance_results)

    # Return empty DataFrame with correct structure if no results
    if df.empty:
        print(f"Warning: No results to analyze for {param_name}")
        return pd.DataFrame(columns=['Parameter Value', 'Components', 'Accuracy', 'Training Time'])

    # Create visualization only if we have results
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Parameter Value'], df['Accuracy'], 'o-', color='blue')
        plt.title(f'Effect of {param_name} on Neural Network Accuracy: {dataset_name}')
        plt.xlabel(param_name)
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        save_path = f'results/sensitivity_nn_{param_name.replace(" ", "_").lower()}_{dataset_name.replace(" ", "_").lower()}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved sensitivity analysis plot to {save_path}")
    except Exception as e:
        print(f"Error creating visualization: {e}")

    return df



# ==========================================================================
# BEGINNING OF MAIN FUNCTION
# ==========================================================================

# ==========================================================================
# MAIN EXECUTION
# ==========================================================================
def main():
    """
    Main execution function that runs the entire pipeline with enhanced analysis.
    """
    start_time_total = time.time()

    print_section_header("CS7641 Assignment 3 - Unsupervised Learning and Dimensionality Reduction", level=1)

    # Dictionary to store all results for final summary
    all_results = {}

    # Load datasets
    marketing_X, marketing_y = load_marketing_data()
    spotify_X, spotify_y = load_spotify_data()

    # Formulate hypotheses
    hypotheses_data = formulate_hypotheses((marketing_X, marketing_y), (spotify_X, spotify_y))
    all_results['hypotheses'] = hypotheses_data

    # Prepare data
    print_section_header("Preparing Data", level=2)
    mk_X_train, mk_X_test, mk_y_train, mk_y_test, mk_X_train_scaled, mk_X_test_scaled = prepare_data(marketing_X,
                                                                                                     marketing_y)
    sp_X_train, sp_X_test, sp_y_train, sp_y_test, sp_X_train_scaled, sp_X_test_scaled = prepare_data(spotify_X,
                                                                                                     spotify_y)

    # Get feature names for analysis
    marketing_features = marketing_X.columns.tolist()
    spotify_features = spotify_X.columns.tolist()

    # ===========================
    # Part 1: Apply clustering algorithms to original datasets
    # ===========================
    print_section_header("PART 1: APPLYING CLUSTERING ALGORITHMS TO ORIGINAL DATASETS", level=1)

    # K-Means on Marketing data
    print_section_header("K-Means on Marketing Dataset", level=2)
    mk_kmeans_results = find_optimal_kmeans(mk_X_train_scaled, n_init_values=[5, 10, 20])
    visualize_kmeans_results(mk_kmeans_results, "Marketing Dataset")
    all_results['marketing_kmeans'] = mk_kmeans_results

    # EM on Marketing data - IMPROVED: Testing different n_init values
    print_section_header("EM on Marketing Dataset", level=2)
    mk_em_results = find_optimal_em(mk_X_train_scaled, cov_types=['full', 'tied', 'diag'], n_init_values=[1, 2, 5])
    visualize_em_results(mk_em_results, "Marketing Dataset")
    all_results['marketing_em'] = mk_em_results

    # Compare clustering on Marketing data
    print_section_header("Comparing Clustering on Marketing Dataset", level=2)
    mk_clustering_comparison, mk_kmeans_labels, mk_em_labels = compare_clustering_algorithms(
        mk_X_train_scaled, mk_y_train,
        mk_kmeans_results['best_k_silhouette'],
        mk_em_results['best_k_bic'],
        marketing_features,
        "Marketing Dataset"
    )
    all_results['marketing_clustering_comparison'] = mk_clustering_comparison

    # Add cluster-label alignment analysis
    mk_kmeans_alignment = analyze_cluster_label_alignment(
        mk_kmeans_labels, mk_y_train,
        "Marketing Dataset", "K-Means"
    )
    all_results['marketing_kmeans_alignment'] = mk_kmeans_alignment

    mk_em_alignment = analyze_cluster_label_alignment(
        mk_em_labels, mk_y_train,
        "Marketing Dataset", "EM"
    )
    all_results['marketing_em_alignment'] = mk_em_alignment

    print_section_header("Detailed Cluster Analysis on Marketing Dataset", level=2)
    mk_kmeans_feature_analysis = analyze_feature_distributions_by_cluster(
        mk_X_train_scaled, mk_kmeans_labels, marketing_features
    )
    visualize_cluster_feature_distributions(mk_kmeans_feature_analysis, "Marketing Dataset (K-Means)")
    all_results['marketing_kmeans_feature_analysis'] = mk_kmeans_feature_analysis

    mk_em_feature_analysis = analyze_feature_distributions_by_cluster(
        mk_X_train_scaled, mk_em_labels, marketing_features
    )
    visualize_cluster_feature_distributions(mk_em_feature_analysis, "Marketing Dataset (EM)")
    all_results['marketing_em_feature_analysis'] = mk_em_feature_analysis

    # K-Means on Spotify data
    print_section_header("K-Means on Spotify Dataset", level=2)
    sp_kmeans_results = find_optimal_kmeans(sp_X_train_scaled, n_init_values=[5, 10, 20])
    visualize_kmeans_results(sp_kmeans_results, "Spotify Dataset")
    all_results['spotify_kmeans'] = sp_kmeans_results

    # EM on Spotify data - IMPROVED: Testing different n_init values
    print_section_header("EM on Spotify Dataset", level=2)
    sp_em_results = find_optimal_em(sp_X_train_scaled, cov_types=['full', 'tied', 'diag'], n_init_values=[1, 2, 5])
    visualize_em_results(sp_em_results, "Spotify Dataset")
    all_results['spotify_em'] = sp_em_results

    # Compare clustering on Spotify data
    print_section_header("Comparing Clustering on Spotify Dataset", level=2)
    sp_clustering_comparison, sp_kmeans_labels, sp_em_labels = compare_clustering_algorithms(
        sp_X_train_scaled, sp_y_train,
        sp_kmeans_results['best_k_silhouette'],
        sp_em_results['best_k_bic'],
        spotify_features,
        "Spotify Dataset"
    )
    all_results['spotify_clustering_comparison'] = sp_clustering_comparison

    # Add cluster-label alignment analysis for Spotify
    sp_kmeans_alignment = analyze_cluster_label_alignment(
        sp_kmeans_labels, sp_y_train,
        "Spotify Dataset", "K-Means"
    )
    all_results['spotify_kmeans_alignment'] = sp_kmeans_alignment

    sp_em_alignment = analyze_cluster_label_alignment(
        sp_em_labels, sp_y_train,
        "Spotify Dataset", "EM"
    )
    all_results['spotify_em_alignment'] = sp_em_alignment

    print_section_header("Detailed Cluster Analysis on Spotify Dataset", level=2)
    sp_kmeans_feature_analysis = analyze_feature_distributions_by_cluster(
        sp_X_train_scaled, sp_kmeans_labels, spotify_features
    )
    visualize_cluster_feature_distributions(sp_kmeans_feature_analysis, "Spotify Dataset (K-Means)")
    all_results['spotify_kmeans_feature_analysis'] = sp_kmeans_feature_analysis

    sp_em_feature_analysis = analyze_feature_distributions_by_cluster(
        sp_X_train_scaled, sp_em_labels, spotify_features
    )
    visualize_cluster_feature_distributions(sp_em_feature_analysis, "Spotify Dataset (EM)")
    all_results['spotify_em_feature_analysis'] = sp_em_feature_analysis

    # ===========================
    # Part 2: Apply dimensionality reduction to datasets
    # ===========================
    print_section_header("PART 2: APPLYING DIMENSIONALITY REDUCTION ALGORITHMS", level=1)

    # PCA on Marketing data
    print_section_header("PCA on Marketing Dataset", level=2)
    mk_pca_results = apply_pca(mk_X_train_scaled)
    visualize_pca_results(mk_pca_results, "Marketing Dataset")
    all_results['marketing_pca'] = mk_pca_results

    # Add data properties analysis for PCA
    mk_pca_properties = analyze_data_properties(
        mk_X_train_scaled,
        mk_pca_results['X_reduced'][:, :mk_pca_results['n_components_threshold']],
        "PCA", "Marketing Dataset"
    )
    all_results['marketing_pca_properties'] = mk_pca_properties

    print_section_header("PCA Sensitivity Analysis on Marketing Dataset", level=2)
    mk_pca_sensitivity = pca_sensitivity_analysis(mk_X_train_scaled,
                                                  variance_thresholds=[0.85, 0.9, 0.95, 0.99])
    visualize_pca_sensitivity(mk_pca_sensitivity, "Marketing Dataset")
    all_results['marketing_pca_sensitivity'] = mk_pca_sensitivity

    # ICA on Marketing data
    print_section_header("ICA on Marketing Dataset", level=2)
    mk_ica_results = apply_ica(mk_X_train_scaled)
    visualize_ica_results(mk_ica_results, "Marketing Dataset")
    all_results['marketing_ica'] = mk_ica_results

    # Add data properties analysis for ICA
    mk_ica_properties = analyze_data_properties(
        mk_X_train_scaled,
        mk_ica_results['X_reduced'],
        "ICA", "Marketing Dataset"
    )
    all_results['marketing_ica_properties'] = mk_ica_properties

    # ICA Sensitivity Analysis
    print_section_header("ICA Sensitivity Analysis on Marketing Dataset", level=2)
    mk_ica_sensitivity = ica_sensitivity_analysis(mk_X_train_scaled,
                                                  kurtosis_thresholds=[3.0, 4.0, 5.0, 6.0])
    visualize_ica_sensitivity(mk_ica_sensitivity, "Marketing Dataset")
    all_results['marketing_ica_sensitivity'] = mk_ica_sensitivity

    # RP on Marketing data
    print_section_header("Random Projection on Marketing Dataset", level=2)
    mk_rp_results = apply_rp(mk_X_train_scaled)
    visualize_rp_results(mk_rp_results, "Marketing Dataset")
    all_results['marketing_rp'] = mk_rp_results

    # Add data properties analysis for RP
    mk_rp_properties = analyze_data_properties(
        mk_X_train_scaled,
        mk_rp_results['X_reduced'],
        "RP", "Marketing Dataset"
    )
    all_results['marketing_rp_properties'] = mk_rp_properties

    print_section_header("Random Projection Sensitivity Analysis on Marketing Dataset", level=2)
    mk_rp_sensitivity = rp_sensitivity_analysis(mk_X_train_scaled,
                                                error_thresholds=[0.3, 0.35, 0.4, 0.45, 0.5])
    visualize_rp_sensitivity(mk_rp_sensitivity, "Marketing Dataset")
    all_results['marketing_rp_sensitivity'] = mk_rp_sensitivity

    # Compare all sensitivity analyses and add implementation for compare_sensitivity_analyses
    print_section_header("Comparing All Sensitivity Analyses on Marketing Dataset", level=2)
    mk_sensitivity_comparison = compare_sensitivity_analyses(
        mk_pca_sensitivity, mk_ica_sensitivity, mk_rp_sensitivity, "Marketing Dataset"
    )
    all_results['marketing_sensitivity_comparison'] = mk_sensitivity_comparison

    # Evaluate effect of sensitivity parameters on neural network performance
    print_section_header("Evaluating Sensitivity Effects on Neural Network (Marketing)", level=2)
    # For PCA
    mk_pca_nn_sensitivity = evaluate_sensitivity_effects_on_nn(
        mk_X_train_scaled, mk_X_test_scaled, mk_y_train, mk_y_test,
        mk_pca_sensitivity, mk_pca_results['model'],
        "Variance Threshold", "Marketing Dataset"
    )

    all_results['marketing_pca_nn_sensitivity'] = mk_pca_nn_sensitivity

    # For ICA
    mk_ica_nn_sensitivity = evaluate_sensitivity_effects_on_nn(
        mk_X_train_scaled, mk_X_test_scaled, mk_y_train, mk_y_test,
        mk_ica_sensitivity, mk_ica_results['model'],
        "Kurtosis Threshold", "Marketing Dataset"
    )
    all_results['marketing_ica_nn_sensitivity'] = mk_ica_nn_sensitivity

    # For RP
    mk_rp_nn_sensitivity = evaluate_sensitivity_effects_on_nn(
        mk_X_train_scaled, mk_X_test_scaled, mk_y_train, mk_y_test,
        mk_rp_sensitivity, mk_rp_results['model'],
        "Error Threshold", "Marketing Dataset"
    )
    all_results['marketing_rp_nn_sensitivity'] = mk_rp_nn_sensitivity

    # Compare dimensionality reduction on Marketing data
    print_section_header("Comparing Dimensionality Reduction on Marketing Dataset", level=2)
    mk_dim_reduction_comparison = compare_dim_reduction(
        mk_pca_results, mk_ica_results, mk_rp_results,
        "Marketing Dataset"
    )
    all_results['marketing_dr_comparison'] = mk_dim_reduction_comparison

    # PCA on Spotify data
    print_section_header("PCA on Spotify Dataset", level=2)
    sp_pca_results = apply_pca(sp_X_train_scaled)
    visualize_pca_results(sp_pca_results, "Spotify Dataset")
    all_results['spotify_pca'] = sp_pca_results

    # Add data properties analysis for PCA on Spotify
    sp_pca_properties = analyze_data_properties(
        sp_X_train_scaled,
        sp_pca_results['X_reduced'][:, :sp_pca_results['n_components_threshold']],
        "PCA", "Spotify Dataset"
    )
    all_results['spotify_pca_properties'] = sp_pca_properties

    print_section_header("PCA Sensitivity Analysis on Spotify Dataset", level=2)
    sp_pca_sensitivity = pca_sensitivity_analysis(sp_X_train_scaled,
                                                  variance_thresholds=[0.85, 0.9, 0.95, 0.99])
    visualize_pca_sensitivity(sp_pca_sensitivity, "Spotify Dataset")
    all_results['spotify_pca_sensitivity'] = sp_pca_sensitivity

    # ICA on Spotify data
    print_section_header("ICA on Spotify Dataset", level=2)
    sp_ica_results = apply_ica(sp_X_train_scaled)
    visualize_ica_results(sp_ica_results, "Spotify Dataset")
    all_results['spotify_ica'] = sp_ica_results

    # Add data properties analysis for ICA on Spotify
    sp_ica_properties = analyze_data_properties(
        sp_X_train_scaled,
        sp_ica_results['X_reduced'],
        "ICA", "Spotify Dataset"
    )
    all_results['spotify_ica_properties'] = sp_ica_properties

    # ICA Sensitivity Analysis
    print_section_header("ICA Sensitivity Analysis on Spotify Dataset", level=2)
    sp_ica_sensitivity = ica_sensitivity_analysis(sp_X_train_scaled,
                                                  kurtosis_thresholds=[3.0, 4.0, 5.0, 6.0])
    visualize_ica_sensitivity(sp_ica_sensitivity, "Spotify Dataset")
    all_results['spotify_ica_sensitivity'] = sp_ica_sensitivity

    # RP on Spotify data
    print_section_header("Random Projection on Spotify Dataset", level=2)
    sp_rp_results = apply_rp(sp_X_train_scaled)
    visualize_rp_results(sp_rp_results, "Spotify Dataset")
    all_results['spotify_rp'] = sp_rp_results

    # Add data properties analysis for RP on Spotify
    sp_rp_properties = analyze_data_properties(
        sp_X_train_scaled,
        sp_rp_results['X_reduced'],
        "RP", "Spotify Dataset"
    )
    all_results['spotify_rp_properties'] = sp_rp_properties

    print_section_header("Random Projection Sensitivity Analysis on Spotify Dataset", level=2)
    sp_rp_sensitivity = rp_sensitivity_analysis(sp_X_train_scaled,
                                                error_thresholds=[0.3, 0.35, 0.4, 0.45, 0.5])
    visualize_rp_sensitivity(sp_rp_sensitivity, "Spotify Dataset")
    all_results['spotify_rp_sensitivity'] = sp_rp_sensitivity

    # Compare all sensitivity analyses
    print_section_header("Comparing All Sensitivity Analyses on Spotify Dataset", level=2)
    sp_sensitivity_comparison = compare_sensitivity_analyses(
        sp_pca_sensitivity, sp_ica_sensitivity, sp_rp_sensitivity, "Spotify Dataset"
    )
    all_results['spotify_sensitivity_comparison'] = sp_sensitivity_comparison

    # Evaluate effect of sensitivity parameters on neural network performance
    print_section_header("Evaluating Sensitivity Effects on Neural Network (Spotify)", level=2)
    # For PCA
    sp_pca_nn_sensitivity = evaluate_sensitivity_effects_on_nn(
        sp_X_train_scaled, sp_X_test_scaled, sp_y_train, sp_y_test,
        sp_pca_sensitivity, sp_pca_results['model'],
        "Variance Threshold", "Spotify Dataset"
    )
    all_results['spotify_pca_nn_sensitivity'] = sp_pca_nn_sensitivity

    # For ICA
    sp_ica_nn_sensitivity = evaluate_sensitivity_effects_on_nn(
        sp_X_train_scaled, sp_X_test_scaled, sp_y_train, sp_y_test,
        sp_ica_sensitivity, sp_ica_results['model'],
        "Kurtosis Threshold", "Spotify Dataset"
    )
    all_results['spotify_ica_nn_sensitivity'] = sp_ica_nn_sensitivity

    # For RP
    sp_rp_nn_sensitivity = evaluate_sensitivity_effects_on_nn(
        sp_X_train_scaled, sp_X_test_scaled, sp_y_train, sp_y_test,
        sp_rp_sensitivity, sp_rp_results['model'],
        "Error Threshold", "Spotify Dataset"
    )
    all_results['spotify_rp_nn_sensitivity'] = sp_rp_nn_sensitivity

    # Compare dimensionality reduction on Spotify data
    print_section_header("Comparing Dimensionality Reduction on Spotify Dataset", level=2)
    sp_dim_reduction_comparison = compare_dim_reduction(
        sp_pca_results, sp_ica_results, sp_rp_results,
        "Spotify Dataset"
    )
    all_results['spotify_dr_comparison'] = sp_dim_reduction_comparison

    # ===========================
    # Part 3: Re-apply clustering after dimensionality reduction
    # ===========================
    print_section_header("PART 3: RE-APPLYING CLUSTERING AFTER DIMENSIONALITY REDUCTION", level=1)

    # Marketing dataset
    print_section_header("Clustering after Dimensionality Reduction on Marketing Dataset", level=2)
    mk_combined_results, mk_combined_comparison = cluster_after_reduction(
        mk_pca_results['X_reduced'][:, :mk_pca_results['n_components_threshold']],
        mk_ica_results['X_reduced'],
        mk_rp_results['X_reduced'],
        mk_kmeans_results['best_k_silhouette'],
        mk_em_results['best_k_bic'],
        "Marketing Dataset",
        mk_kmeans_labels,
        mk_em_labels
    )
    all_results['marketing_combined'] = (mk_combined_results, mk_combined_comparison)

    # Add cluster stability analysis for K-Means with each DR method
    for dr_method, dr_key in [('PCA', 'kmeans_pca'), ('ICA', 'kmeans_ica'), ('RP', 'kmeans_rp')]:
        dr_labels = mk_combined_results[dr_key]['labels']
        mk_kmeans_stability = analyze_cluster_stability(
            mk_kmeans_labels, dr_labels,
            dr_method, "K-Means",
            "Marketing Dataset"
        )
        all_results[f'marketing_kmeans_{dr_method.lower()}_stability'] = mk_kmeans_stability

    # Add cluster stability analysis for EM with each DR method
    for dr_method, dr_key in [('PCA', 'em_pca'), ('ICA', 'em_ica'), ('RP', 'em_rp')]:
        dr_labels = mk_combined_results[dr_key]['labels']
        mk_em_stability = analyze_cluster_stability(
            mk_em_labels, dr_labels,
            dr_method, "EM",
            "Marketing Dataset"
        )
        all_results[f'marketing_em_{dr_method.lower()}_stability'] = mk_em_stability

    # Spotify dataset
    print_section_header("Clustering after Dimensionality Reduction on Spotify Dataset", level=2)
    sp_combined_results, sp_combined_comparison = cluster_after_reduction(
        sp_pca_results['X_reduced'][:, :sp_pca_results['n_components_threshold']],
        sp_ica_results['X_reduced'],
        sp_rp_results['X_reduced'],
        sp_kmeans_results['best_k_silhouette'],
        sp_em_results['best_k_bic'],
        "Spotify Dataset",
        sp_kmeans_labels,
        sp_em_labels
    )
    all_results['spotify_combined'] = (sp_combined_results, sp_combined_comparison)




    # Create comprehensive summary of all 12 combinations
    print_section_header("Comprehensive Clustering Performance Summary", level=2)

    # Create a comprehensive summary table
    summary_data = []

    # Add Marketing dataset results
    for key, value in mk_combined_results.items():
        algorithm_name = key.replace('_', ' + ').upper()
        summary_data.append({
            'Dataset': 'Marketing',
            'Algorithm': algorithm_name,
            'NMI Score': value['nmi'],
            'Silhouette Score': value['silhouette'],
            'Execution Time (s)': value['execution_time']
        })

    # Add Spotify dataset results
    for key, value in sp_combined_results.items():
        algorithm_name = key.replace('_', ' + ').upper()
        summary_data.append({
            'Dataset': 'Spotify',
            'Algorithm': algorithm_name,
            'NMI Score': value['nmi'],
            'Silhouette Score': value['silhouette'],
            'Execution Time (s)': value['execution_time']
        })

    # Create and display summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Dataset', 'NMI Score'], ascending=[True, False])
    print("\nComplete Summary of All 12 Combinations:")
    print(summary_df.to_string(index=False))

    # Find overall best combination
    best_combo = summary_df.loc[summary_df['NMI Score'].idxmax()]
    print(f"\nBest overall combination: {best_combo['Dataset']} dataset with {best_combo['Algorithm']}")
    print(
        f"NMI score: {best_combo['NMI Score']:.4f}, Silhouette score: {best_combo['Silhouette Score']:.4f}, Execution time: {best_combo['Execution Time (s)']:.2f}s")

    # Save the summary to CSV
    summary_df.to_csv('results/combined/all_combinations_summary.csv', index=False)

    # Store in all_results for final summary
    all_results['all_combinations_summary'] = summary_df





    # Add cluster stability analysis for Spotify K-Means with each DR method
    for dr_method, dr_key in [('PCA', 'kmeans_pca'), ('ICA', 'kmeans_ica'), ('RP', 'kmeans_rp')]:
        dr_labels = sp_combined_results[dr_key]['labels']
        sp_kmeans_stability = analyze_cluster_stability(
            sp_kmeans_labels, dr_labels,
            dr_method, "K-Means",
            "Spotify Dataset"
        )
        all_results[f'spotify_kmeans_{dr_method.lower()}_stability'] = sp_kmeans_stability

    # Add cluster stability analysis for Spotify EM with each DR method
    for dr_method, dr_key in [('PCA', 'em_pca'), ('ICA', 'em_ica'), ('RP', 'em_rp')]:
        dr_labels = sp_combined_results[dr_key]['labels']
        sp_em_stability = analyze_cluster_stability(
            sp_em_labels, dr_labels,
            dr_method, "EM",
            "Spotify Dataset"
        )
        all_results[f'spotify_em_{dr_method.lower()}_stability'] = sp_em_stability

    # ===========================
    # Part 4: Neural network with dimensionality reduction
    # ===========================
    print_section_header("PART 4: NEURAL NETWORK WITH DIMENSIONALITY REDUCTION", level=1)

    # Marketing dataset - IMPROVED: Using enhanced statistical comparison
    print_section_header("Neural Network with Dimensionality Reduction on Marketing Dataset", level=2)
    mk_nn_reduction_results = nn_with_dimensionality_reduction(
        mk_X_train_scaled, mk_X_test_scaled, mk_y_train, mk_y_test,
        mk_pca_results, mk_ica_results, mk_rp_results,
        "Marketing Dataset"
    )
    all_results['marketing_nn_dr'] = mk_nn_reduction_results

    # Add neural network performance analysis
    dr_results = {
        'pca': mk_nn_reduction_results['pca'],
        'ica': mk_nn_reduction_results['ica'],
        'rp': mk_nn_reduction_results['rp']
    }
    mk_nn_performance_analysis = analyze_neural_network_performance_changes(
        mk_nn_reduction_results['baseline'],
        dr_results,
        "Marketing Dataset"
    )
    all_results['marketing_nn_performance_analysis'] = mk_nn_performance_analysis

    # ===========================
    # Part 5: Neural network with clustering features
    # ===========================
    print_section_header("PART 5: NEURAL NETWORK WITH CLUSTERING FEATURES", level=1)

    # Marketing dataset - IMPROVED: Using enhanced statistical comparison
    print_section_header("Neural Network with Clustering Features on Marketing Dataset", level=2)
    mk_nn_clustering_results = nn_with_clustering_features(
        mk_X_train_scaled, mk_X_test_scaled, mk_y_train, mk_y_test,
        mk_kmeans_results, mk_em_results,
        "Marketing Dataset"
    )
    all_results['marketing_nn_cluster'] = mk_nn_clustering_results

    # Add neural network performance analysis for clustering features
    cluster_results = {
        'kmeans': mk_nn_clustering_results['kmeans'],
        'em': mk_nn_clustering_results['em']
    }
    mk_cluster_nn_performance_analysis = analyze_neural_network_performance_changes(
        mk_nn_clustering_results['baseline'],
        cluster_results,
        "Marketing Dataset (Clustering Features)"
    )
    all_results['marketing_cluster_nn_performance_analysis'] = mk_cluster_nn_performance_analysis

    # ===========================
    # Final Summary
    # ===========================
    print_section_header("FINAL SUMMARY", level=1)

    marketing_summary = [
        ["Best K-Means clusters", mk_kmeans_results['best_k_silhouette']],
        ["Best K-Means n_init", mk_kmeans_results.get('best_n_init', 10)],  # Display best n_init
        ["Best EM components", mk_em_results['best_k_bic']],
        ["Best EM covariance type", mk_em_results.get('best_cov_type', 'full')],  # Display best covariance
        ["Best EM n_init", mk_em_results.get('best_n_init', 5)],  # Display best n_init
        ["Optimal PCA components", mk_pca_results['n_components_threshold']],
        ["Optimal ICA components", mk_ica_results['optimal_components']],
        ["Optimal RP components", mk_rp_results['optimal_components']],
        ["Best dimensionality reduction for NN", mk_nn_reduction_results['comparison'].iloc[0]['Model']],
        ["Best clustering method for NN", mk_nn_clustering_results['comparison'].iloc[0]['Model']]
    ]
    print_table(marketing_summary, ["Metric", "Value"], "Marketing Dataset Summary")

    spotify_summary = [
        ["Best K-Means clusters", sp_kmeans_results['best_k_silhouette']],
        ["Best K-Means n_init", sp_kmeans_results.get('best_n_init', 10)],  # Display best n_init
        ["Best EM components", sp_em_results['best_k_bic']],
        ["Best EM covariance type", sp_em_results.get('best_cov_type', 'full')],  # Display best covariance
        ["Best EM n_init", sp_em_results.get('best_n_init', 5)],  # Display best n_init
        ["Optimal PCA components", sp_pca_results['n_components_threshold']],
        ["Optimal ICA components", sp_ica_results['optimal_components']],
        ["Optimal RP components", sp_rp_results['optimal_components']]
    ]
    print_table(spotify_summary, ["Metric", "Value"], "Spotify Dataset Summary")

    # ===========================
    # EXTENDED ANALYSIS SUMMARY - New addition
    # ===========================
    print_section_header("EXTENDED ANALYSIS SUMMARY", level=1)

    # Hyperparameter sensitivity summary
    print("\nHyperparameter Sensitivity Summary")

    # PCA sensitivity summary
    pca_summary = []
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        sensitivity_key = f'{dataset_lower}_pca_sensitivity'
        if sensitivity_key in all_results:
            for threshold, results in all_results[sensitivity_key].items():
                pca_summary.append([
                    dataset,
                    f"{threshold:.0%}",
                    results['n_components'],
                    f"{100 * results['dimension_reduction']:.1f}%"
                ])

    if pca_summary:
        print_table(pca_summary, ["Dataset", "Variance Threshold", "Components", "Reduction"],
                    "PCA Sensitivity Summary")

    # ICA sensitivity summary
    ica_summary = []
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        sensitivity_key = f'{dataset_lower}_ica_sensitivity'
        if sensitivity_key in all_results:
            for threshold, results in all_results[sensitivity_key].items():
                ica_summary.append([
                    dataset,
                    f"{threshold:.1f}",
                    results['n_components'],
                    f"{100 * results['dimension_reduction']:.1f}%"
                ])

    if ica_summary:
        print_table(ica_summary, ["Dataset", "Kurtosis Threshold", "Components", "Reduction"],
                    "ICA Sensitivity Summary")

    # RP sensitivity summary
    rp_summary = []
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        sensitivity_key = f'{dataset_lower}_rp_sensitivity'
        if sensitivity_key in all_results:
            for threshold, results in all_results[sensitivity_key].items():
                rp_summary.append([
                    dataset,
                    f"{threshold:.2f}",
                    results['optimal_components'],
                    f"{100 * results['dimension_reduction']:.1f}%"
                ])

    if rp_summary:
        print_table(rp_summary, ["Dataset", "Error Threshold", "Components", "Reduction"],
                    "Random Projection Sensitivity Summary")

    # Neural network performance sensitivity summary
    nn_sensitivity_summary = []
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        for algo in ['pca', 'ica', 'rp']:
            sensitivity_key = f'{dataset_lower}_{algo}_nn_sensitivity'
            if sensitivity_key in all_results and isinstance(all_results[sensitivity_key], pd.DataFrame):
                df = all_results[sensitivity_key]

                # Check if required columns exist in the DataFrame
                required_columns = ['Parameter Value', 'Components', 'Accuracy']
                if all(col in df.columns for col in required_columns):
                    try:
                        param_min = df['Parameter Value'].min()
                        param_max = df['Parameter Value'].max()
                        comp_min = df['Components'].min()
                        comp_max = df['Components'].max()
                        acc_min = df['Accuracy'].min()
                        acc_max = df['Accuracy'].max()

                        best_idx = df['Accuracy'].idxmax()
                        best_param = df.loc[best_idx, 'Parameter Value'] if best_idx in df.index else 'N/A'

                        nn_sensitivity_summary.append([
                            dataset,
                            algo.upper(),
                            f"{param_min} - {param_max}",
                            f"{comp_min} - {comp_max}",
                            f"{acc_min:.4f} - {acc_max:.4f}",
                            f"{best_idx} ({best_param})"
                        ])
                    except Exception as e:
                        print(f"Warning: Error processing {sensitivity_key}: {e}")
                else:
                    missing = [col for col in required_columns if col not in df.columns]
                    print(f"Warning: Missing required columns in {sensitivity_key}: {missing}")
                    print(f"Available columns: {df.columns.tolist()}")

    if nn_sensitivity_summary:
        print_table(nn_sensitivity_summary,
                    ["Dataset", "Algorithm", "Parameter Range", "Component Range",
                     "Accuracy Range", "Best Configuration"],
                    "Neural Network Sensitivity Summary")

    # Comparative sensitivity summary
    sensitivity_comparison_summary = []
    for dataset in ['Marketing', 'Spotify']:
        dataset_lower = dataset.lower()
        comparison_key = f'{dataset_lower}_sensitivity_comparison'
        if comparison_key in all_results and isinstance(all_results[comparison_key], pd.DataFrame):
            for _, row in all_results[comparison_key].iterrows():
                sensitivity_comparison_summary.append([
                    dataset,
                    row['Algorithm'],
                    row['Threshold Range'],
                    row['Component Range'],
                    row['Reduction Range']
                ])

    if sensitivity_comparison_summary:
        print_table(sensitivity_comparison_summary,
                    ["Dataset", "Algorithm", "Threshold Range", "Component Range", "Reduction Range"],
                    "Comparative Sensitivity Summary")

    # IMPROVED: Enhanced statistical testing summary
    statistical_summary = []

    # DR statistical results
    dr_stats_key = 'marketing_nn_dr'
    if dr_stats_key in all_results and 'statistical_results' in all_results[dr_stats_key]:
        stats = all_results[dr_stats_key]['statistical_results']
        for model, results in stats.items():
            statistical_summary.append([
                "Marketing",
                model,
                results.get('test_name', 'McNemar'),  # IMPROVED: Include test name
                f"{results['p_value']:.4f}",
                "Yes" if results['significant'] else "No",
                "Improved" if results['significant'] and results['model_advantage'] else
                "Degraded" if results['significant'] else "No Change",
                # IMPROVED: Include contingency information
                f"{results['contingency_table']['model_only']} vs {results['contingency_table']['baseline_only']}"
            ])

    # Clustering statistical results
    cluster_stats_key = 'marketing_nn_cluster'
    if cluster_stats_key in all_results and 'statistical_results' in all_results[cluster_stats_key]:
        stats = all_results[cluster_stats_key]['statistical_results']
        for model, results in stats.items():
            statistical_summary.append([
                "Marketing",
                model,
                results.get('test_name', 'McNemar'),  # IMPROVED: Include test name
                f"{results['p_value']:.4f}",
                "Yes" if results['significant'] else "No",
                "Improved" if results['significant'] and results['model_advantage'] else
                "Degraded" if results['significant'] else "No Change",
                # IMPROVED: Include contingency information
                f"{results['contingency_table']['model_only']} vs {results['contingency_table']['baseline_only']}"
            ])

    if statistical_summary:
        print_table(statistical_summary,
                    ["Dataset", "Model", "Test", "p-value", "Significant", "Effect", "Disagreement"],
                    "Statistical Testing Summary")



    # ===========================
    # EXTRA CREDIT SECTION
    # ===========================
    run_extra_credit_flag = True  # Set to True to run the extra credit

    if run_extra_credit_flag:
        print_section_header("EXTRA CREDIT: NON-LINEAR MANIFOLD LEARNING", level=1)

        # Apply manifold learning - pass the results from linear methods
        manifold_results = apply_manifold_learning(
            mk_X_train_scaled, n_components=2, dataset_name="Marketing Dataset",
            pca_results=mk_pca_results, ica_results=mk_ica_results, rp_results=mk_rp_results
        )

        # Create visualizations for manifold embeddings
        visualize_manifold_embeddings(manifold_results, "Marketing Dataset")

        # Compare clustering with manifold
        manifold_clustering_results, manifold_clustering_comparison = compare_clustering_with_manifold(
            mk_X_train_scaled, manifold_results['embeddings'],
            mk_kmeans_results['best_k_silhouette'],
            mk_em_results['best_k_bic'],
            "Marketing Dataset"
        )

        # Create a dictionary of results for the summary tables
        extra_credit_results = {
            'manifold_results': manifold_results,
            'manifold_clustering_results': manifold_clustering_results,
            'manifold_clustering_comparison': manifold_clustering_comparison
        }

        # Update all_results with extra credit results
        all_results.update(extra_credit_results)

        print("\nExtra credit completed successfully!")

    # Add hypothesis evaluation before final summary
    print_section_header("HYPOTHESIS EVALUATION", level=1)
    hypothesis_evaluation = evaluate_hypotheses(all_results)
    all_results['hypothesis_evaluation'] = hypothesis_evaluation

    # Create comprehensive summary tables
    create_summary_tables(all_results)
    get_best_algorithm_summary(all_results)

    # Show total execution time
    total_time = time.time() - start_time_total
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_summary = [
        ["Total Execution Time", f"{total_time:.2f} seconds"],
        ["", f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"]
    ]
    print_table(time_summary, ["Metric", "Value"], "Execution Time Summary")

    print("\nAll analyses completed and results saved to the 'results' directory.")




# ==========================================================================
# END OF MAIN FUNCTION
# ==========================================================================



# ==========================================================================
# EXTRA CREDIT: NON-LINEAR MANIFOLD LEARNING
# ==========================================================================

def apply_manifold_learning(X, n_components=2, dataset_name='', save_path='results/extra',
                            pca_results=None, ica_results=None, rp_results=None):
    """
    Applies multiple non-linear manifold learning methods for comparison and visualization.
    This is for the extra credit portion of the assignment.

    Parameters:
    -----------
    X : array-like
        Input data for dimensionality reduction
    n_components : int, default=2
        Number of components to reduce to
    dataset_name : str
        Name of the dataset for visualization titles and filenames
    save_path : str
        Directory path to save results
    pca_results : dict
        Results from PCA dimensionality reduction
    ica_results : dict
        Results from ICA dimensionality reduction
    rp_results : dict
        Results from Random Projection dimensionality reduction

    Returns:
    --------
    dict
        Results from manifold learning including embeddings and execution times
    """
    print("\n--- Applying Non-Linear Manifold Learning (Extra Credit) ---")

    from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding

    # Apply various manifold learning methods
    methods = {
        'TSNE': TSNE(n_components=n_components, random_state=RANDOM_SEED),
        'Isomap': Isomap(n_components=n_components, n_neighbors=10),
        'LLE': LocallyLinearEmbedding(n_components=n_components, n_neighbors=10, random_state=RANDOM_SEED),
        'Spectral': SpectralEmbedding(n_components=n_components, random_state=RANDOM_SEED)
    }

    embeddings = {}
    execution_times = {}

    for name, method in methods.items():
        print(f"Applying {name}...")
        start_time = time.time()
        embedding = method.fit_transform(X)
        end_time = time.time()

        embeddings[name] = embedding
        execution_times[name] = end_time - start_time

        print(f"  {name} completed in {execution_times[name]:.2f} seconds")

    # Create comparison visualizations
    plt.figure(figsize=(15, 15))

    for i, (name, embedding) in enumerate(embeddings.items(), 1):
        plt.subplot(2, 2, i)
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)
        plt.title(f'{name}\nExecution Time: {execution_times[name]:.2f}s')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f'Non-Linear Manifold Learning Comparison: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{save_path}/manifold_comparison_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create comparison table
    comparison_data = {
        'Method': list(methods.keys()),
        'Execution Time (seconds)': [execution_times[name] for name in methods.keys()]
    }

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Execution Time (seconds)').reset_index(drop=True)

    # Save comparison to CSV
    comparison_df.to_csv(f'{save_path}/manifold_comparison_{dataset_name.replace(" ", "_").lower()}.csv', index=False)

    # Compare with linear dimensionality reduction
    plt.figure(figsize=(8, 6))

    # Get names and execution times for linear methods
    linear_names = ['PCA', 'ICA', 'RP']
    # Use the parameters passed to the function instead of trying to access global variables
    linear_times = [
        pca_results['execution_time'] if pca_results else 0,
        ica_results['final_execution_time'] if ica_results else 0,
        rp_results['final_execution_time'] if rp_results else 0
    ]

    # Get names and execution times for non-linear methods
    nonlinear_names = list(methods.keys())
    nonlinear_times = [execution_times[name] for name in methods.keys()]

    # All methods
    all_names = linear_names + nonlinear_names
    all_times = linear_times + nonlinear_times

    # Create bar chart
    bars = plt.bar(all_names, all_times, color=['blue', 'purple', 'green', 'red', 'orange', 'brown', 'pink'])

    # Add time labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=8)

    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Linear vs Non-Linear DR Execution Time: {dataset_name}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_path}/linear_vs_nonlinear_time_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Return best non-linear method and its embedding
    best_method = comparison_df.iloc[0]['Method']
    best_embedding = embeddings[best_method]

    return {
        'embeddings': embeddings,
        'execution_times': execution_times,
        'comparison': comparison_df,
        'best_method': best_method,
        'best_embedding': best_embedding
    }


def compare_clustering_with_manifold(X, embeddings, kmeans_k, em_k, dataset_name, save_path='results/extra'):
    """
    Compares clustering results when applied to manifold embeddings.
    """
    print("\n--- Comparing Clustering on Manifold Embeddings ---")

    results = {}

    for name, embedding in embeddings.items():
        print(f"Applying clustering to {name} embedding...")

        # Apply K-Means
        kmeans = KMeans(n_clusters=kmeans_k, random_state=RANDOM_SEED)
        kmeans_labels = kmeans.fit_predict(embedding)
        kmeans_silhouette = silhouette_score(embedding, kmeans_labels)

        # Apply EM
        em = GaussianMixture(n_components=em_k, random_state=RANDOM_SEED)
        em.fit(embedding)
        em_labels = em.predict(embedding)
        em_silhouette = silhouette_score(embedding, em_labels)

        # Store results
        results[name] = {
            'kmeans_labels': kmeans_labels,
            'kmeans_silhouette': kmeans_silhouette,
            'em_labels': em_labels,
            'em_silhouette': em_silhouette
        }

        print(f"  K-Means Silhouette: {kmeans_silhouette:.4f}")
        print(f"  EM Silhouette: {em_silhouette:.4f}")

    # Create comparison visualization
    plt.figure(figsize=(15, 10))

    # K-Means comparison
    plt.subplot(2, 1, 1)
    kmeans_silhouettes = [results[name]['kmeans_silhouette'] for name in embeddings.keys()]
    bars = plt.bar(embeddings.keys(), kmeans_silhouettes, color='skyblue')

    # Add silhouette labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.ylabel('Silhouette Score')
    plt.title(f'K-Means Clustering on Manifold Embeddings (K={kmeans_k})')
    plt.grid(True, alpha=0.3, axis='y')

    # EM comparison
    plt.subplot(2, 1, 2)
    em_silhouettes = [results[name]['em_silhouette'] for name in embeddings.keys()]
    bars = plt.bar(embeddings.keys(), em_silhouettes, color='salmon')

    # Add silhouette labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.ylabel('Silhouette Score')
    plt.title(f'EM Clustering on Manifold Embeddings (K={em_k})')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.suptitle(f'Clustering Performance on Manifold Embeddings: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.savefig(f'{save_path}/manifold_clustering_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create scatter plots with cluster assignments
    for name, embedding in embeddings.items():
        plt.figure(figsize=(15, 6))

        # K-Means
        plt.subplot(1, 2, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=results[name]['kmeans_labels'], cmap='viridis', alpha=0.7, s=20)
        plt.title(f'K-Means Clusters on {name} (Silhouette: {results[name]["kmeans_silhouette"]:.4f})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(label='Cluster')

        # EM
        plt.subplot(1, 2, 2)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=results[name]['em_labels'], cmap='viridis', alpha=0.7, s=20)
        plt.title(f'EM Clusters on {name} (Silhouette: {results[name]["em_silhouette"]:.4f})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(label='Cluster')

        plt.tight_layout()
        plt.suptitle(f'Clustering on {name} Embedding: {dataset_name}', fontsize=16)
        plt.subplots_adjust(top=0.85)
        plt.savefig(f'{save_path}/clusters_{name.lower()}_{dataset_name.replace(" ", "_").lower()}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create comparison table
    comparison_data = {
        'Embedding': list(embeddings.keys()),
        'K-Means Silhouette': [results[name]['kmeans_silhouette'] for name in embeddings.keys()],
        'EM Silhouette': [results[name]['em_silhouette'] for name in embeddings.keys()]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by average silhouette score
    comparison_df['Average Silhouette'] = (comparison_df['K-Means Silhouette'] + comparison_df['EM Silhouette']) / 2
    comparison_df = comparison_df.sort_values('Average Silhouette', ascending=False).reset_index(drop=True)

    # Save comparison to CSV
    comparison_df.to_csv(f'{save_path}/manifold_clustering_comparison_{dataset_name.replace(" ", "_").lower()}.csv',
                         index=False)

    return results, comparison_df


# ==========================================================================
# EXTRA CREDIT
# ==========================================================================

def run_extra_credit(X_train_scaled, dataset_name, kmeans_k, em_k, pca_results, ica_results, rp_results):
    """
    Runs the extra credit portion (non-linear manifold learning).
    """
    print_section_header("EXTRA CREDIT: NON-LINEAR MANIFOLD LEARNING", level=1)

    # Apply manifold learning - pass the results from linear methods
    manifold_results = apply_manifold_learning(
        X_train_scaled, n_components=2, dataset_name=dataset_name,
        pca_results=pca_results, ica_results=ica_results, rp_results=rp_results
    )

    # Create visualizations for manifold embeddings
    visualize_manifold_embeddings(manifold_results, dataset_name)

    # Compare clustering with manifold
    manifold_clustering_results, manifold_clustering_comparison = compare_clustering_with_manifold(
        X_train_scaled, manifold_results['embeddings'], kmeans_k, em_k, dataset_name
    )

    # Create a dictionary of results for the summary tables
    extra_credit_results = {
        'manifold_results': manifold_results,
        'manifold_clustering_results': manifold_clustering_results,
        'manifold_clustering_comparison': manifold_clustering_comparison
    }

    return manifold_results, manifold_clustering_results, manifold_clustering_comparison, extra_credit_results



# ==========================================================================
# ANCILLARY FUNCTIONS
# ==========================================================================


def formulate_hypotheses(marketing_data, spotify_data):
    """
    Formulates explicit hypotheses about the datasets and algorithms.
    """

    # Analyze class distribution
    marketing_imbalance = sum(marketing_data[1]) / len(marketing_data[1])
    spotify_imbalance = sum(spotify_data[1]) / len(spotify_data[1])

    print("\n====== Dataset Characteristics and Hypotheses ======")
    print(f"Marketing dataset: {len(marketing_data[0])} samples, {marketing_data[0].shape[1]} features")
    print(f"Class distribution: {marketing_imbalance:.2f} positive rate")
    print(f"Spotify dataset: {len(spotify_data[0])} samples, {spotify_data[0].shape[1]} features")
    print(f"Class distribution: {spotify_imbalance:.2f} positive rate")

    print("\n====== Hypotheses ======")
    print("H1: On highly imbalanced data (Marketing dataset), algorithms with inherent geometric")
    print("    flexibility (EM) will outperform rigid boundary-based approaches (K-Means).")
    print("H2: Dimensionality reduction techniques will impact clustering quality differently")
    print("    based on how well they preserve the underlying data structure.")
    print("H3: PCA will maintain classification accuracy while significantly reducing dimensionality")
    print("    and training time.")
    print("H4: Clusters derived from manifold learning methods will have higher silhouette scores")
    print("    than those from linear dimensionality reduction techniques.")

    return {
        "marketing_imbalance": marketing_imbalance,
        "spotify_imbalance": spotify_imbalance,
        "hypotheses": [
            "EM outperforms K-Means on imbalanced data",
            "DR impacts clustering based on structure preservation",
            "PCA maintains accuracy while reducing dimensions",
            "Manifold learning produces better clusters"
        ]
    }


def analyze_cluster_label_alignment(cluster_labels, true_labels, dataset_name, algorithm_name,
                                    save_path='results/clustering'):
    """
    Analyzes how well cluster assignments align with original class labels.
    """

    # Calculate alignment metrics
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, cluster_labels)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel(f'Cluster Label ({algorithm_name})')
    plt.ylabel('True Class')
    plt.title(f'Cluster-Class Alignment: {dataset_name} with {algorithm_name}')
    plt.tight_layout()
    plt.savefig(f'{save_path}/cluster_alignment_{algorithm_name}_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate dominant class for each cluster
    dominant_classes = {}
    for cluster in np.unique(cluster_labels):
        cluster_mask = (cluster_labels == cluster)
        if np.sum(cluster_mask) > 0:  # Ensure cluster is not empty
            class_counts = np.bincount(true_labels[cluster_mask].astype(int))
            dominant_class = np.argmax(class_counts)
            purity = class_counts[dominant_class] / np.sum(class_counts)
            dominant_classes[cluster] = {
                'dominant_class': dominant_class,
                'purity': purity,
                'size': np.sum(cluster_mask)
            }

    print(f"\nCluster-Label Alignment for {algorithm_name} on {dataset_name}:")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print("\nDominant Class in Each Cluster:")

    for cluster, info in dominant_classes.items():
        print(f"Cluster {cluster}: Class {info['dominant_class']} "
              f"(Purity: {info['purity']:.2f}, Size: {info['size']} samples)")

    return {
        'ari': ari,
        'nmi': nmi,
        'dominant_classes': dominant_classes,
        'confusion_matrix': cm
    }


def analyze_data_properties(X, X_reduced, algorithm_name, dataset_name, save_path='results/dim_reduction'):
    """
    Analyzes properties of data before and after dimensionality reduction.
    """

    # Calculate data rank
    rank_original = np.linalg.matrix_rank(X)
    rank_reduced = np.linalg.matrix_rank(X_reduced)

    # Calculate correlation matrices
    corr_original = np.corrcoef(X, rowvar=False)
    corr_reduced = np.corrcoef(X_reduced, rowvar=False)

    # Calculate average absolute correlation (measure of collinearity)
    # Get upper triangle without diagonal
    upper_triangle_idx = np.triu_indices(corr_original.shape[0], k=1)
    collinearity_original = np.mean(np.abs(corr_original[upper_triangle_idx]))

    if X_reduced.shape[1] > 1:  # Ensure reduced data has at least 2 dimensions
        upper_triangle_idx_reduced = np.triu_indices(corr_reduced.shape[0], k=1)
        collinearity_reduced = np.mean(np.abs(corr_reduced[upper_triangle_idx_reduced]))
    else:
        collinearity_reduced = 0

    # Calculate condition number (measure of numerical stability)
    cond_original = np.linalg.cond(X)
    cond_reduced = np.linalg.cond(X_reduced) if X_reduced.shape[1] > 1 else 0

    # Visualize correlation matrices
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    # Limit to first 15x15 for visibility if larger
    display_size = min(15, corr_original.shape[0])
    sns.heatmap(corr_original[:display_size, :display_size],
                cmap='coolwarm', center=0,
                xticklabels=range(1, display_size + 1),
                yticklabels=range(1, display_size + 1))
    plt.title('Original Feature Correlations')

    plt.subplot(1, 2, 2)
    display_size_reduced = min(15, corr_reduced.shape[0])
    sns.heatmap(corr_reduced[:display_size_reduced, :display_size_reduced],
                cmap='coolwarm', center=0,
                xticklabels=range(1, display_size_reduced + 1),
                yticklabels=range(1, display_size_reduced + 1))
    plt.title(f'Reduced Feature Correlations ({algorithm_name})')

    plt.suptitle(f'Correlation Structure: {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'{save_path}/correlation_{algorithm_name}_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary of data properties
    print(f"\nData Properties Analysis for {dataset_name} with {algorithm_name}:")
    print(f"Original Data Rank: {rank_original} / {X.shape[1]}")
    print(f"Reduced Data Rank: {rank_reduced} / {X_reduced.shape[1]}")
    print(f"Original Data Collinearity: {collinearity_original:.4f}")
    print(f"Reduced Data Collinearity: {collinearity_reduced:.4f}")
    print(f"Original Condition Number: {cond_original:.2e}")
    print(f"Reduced Condition Number: {cond_reduced:.2e}")

    return {
        'rank_original': rank_original,
        'rank_reduced': rank_reduced,
        'collinearity_original': collinearity_original,
        'collinearity_reduced': collinearity_reduced,
        'condition_original': cond_original,
        'condition_reduced': cond_reduced
    }


def analyze_cluster_stability(original_labels, reduced_labels, dr_algorithm, clustering_algorithm,
                              dataset_name, save_path='results/combined'):
    """
    Analyzes stability of clusters before and after dimensionality reduction.
    """

    # Calculate stability metrics
    ari = adjusted_rand_score(original_labels, reduced_labels)
    nmi = normalized_mutual_info_score(original_labels, reduced_labels)

    # Create confusion matrix
    cm = confusion_matrix(original_labels, reduced_labels)

    # Normalize by row (original clusters)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel(f'Cluster After {dr_algorithm}')
    plt.ylabel('Original Cluster')
    plt.title(f'Cluster Stability: {clustering_algorithm} on {dataset_name}\nBefore and After {dr_algorithm}')
    plt.tight_layout()
    plt.savefig(
        f'{save_path}/cluster_stability_{clustering_algorithm}_{dr_algorithm}_{dataset_name.replace(" ", "_").lower()}.png',
        dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate cluster size changes
    original_counts = np.bincount(original_labels)
    reduced_counts = np.bincount(reduced_labels)

    # Adjust lengths if different
    max_clusters = max(len(original_counts), len(reduced_counts))
    if len(original_counts) < max_clusters:
        original_counts = np.pad(original_counts, (0, max_clusters - len(original_counts)))
    if len(reduced_counts) < max_clusters:
        reduced_counts = np.pad(reduced_counts, (0, max_clusters - len(reduced_counts)))

    # Visualize cluster size changes
    plt.figure(figsize=(12, 6))
    x = np.arange(max_clusters)
    width = 0.35

    plt.bar(x - width / 2, original_counts, width, label='Original Data')
    plt.bar(x + width / 2, reduced_counts, width, label=f'After {dr_algorithm}')

    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title(f'Cluster Size Changes: {clustering_algorithm} on {dataset_name}')
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(
        f'{save_path}/cluster_sizes_{clustering_algorithm}_{dr_algorithm}_{dataset_name.replace(" ", "_").lower()}.png',
        dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCluster Stability Analysis for {clustering_algorithm} on {dataset_name} before and after {dr_algorithm}:")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print("\nCluster Size Changes:")
    for i in range(max_clusters):
        if i < len(original_counts) and i < len(reduced_counts):
            orig = original_counts[i]
            red = reduced_counts[i]
            change_pct = ((red - orig) / orig * 100) if orig > 0 else float('inf')
            print(f"Cluster {i}: {orig} → {red} samples ({change_pct:+.1f}%)")

    return {
        'ari': ari,
        'nmi': nmi,
        'confusion_matrix': cm,
        'original_counts': original_counts,
        'reduced_counts': reduced_counts
    }


def analyze_neural_network_performance_changes(baseline_results, dr_results,
                                               dataset_name, save_path='results/neural_net'):
    """
    Analyzes how neural network performance changes with dimensionality reduction.
    """


    # Extract metrics
    models = ['baseline'] + list(dr_results.keys())
    accuracies = [baseline_results['accuracy']] + [dr_results[k]['accuracy'] for k in dr_results]
    f1_scores = [baseline_results['f1_score']] + [dr_results[k]['f1_score'] for k in dr_results]
    training_times = [baseline_results['training_time']] + [dr_results[k]['training_time'] for k in dr_results]

    # Calculate relative changes
    baseline_accuracy = baseline_results['accuracy']
    baseline_f1 = baseline_results['f1_score']
    baseline_time = baseline_results['training_time']

    accuracy_changes = [(acc - baseline_accuracy) / baseline_accuracy * 100 for acc in accuracies]
    f1_changes = [(f1 - baseline_f1) / baseline_f1 * 100 for f1 in f1_scores]
    time_changes = [(time - baseline_time) / baseline_time * 100 for time in training_times]

    # Create detailed analysis visualization
    plt.figure(figsize=(15, 10))

    # Accuracy changes
    plt.subplot(2, 2, 1)
    bars = plt.bar(models, accuracies, color='skyblue')
    plt.axhline(y=baseline_results['accuracy'], linestyle='--', color='red', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Absolute Accuracy')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Relative accuracy changes
    plt.subplot(2, 2, 2)
    bars = plt.bar(models, accuracy_changes, color='lightgreen')
    plt.axhline(y=0, linestyle='--', color='red', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy Change (%)')
    plt.title('Relative Accuracy Change')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Training time
    plt.subplot(2, 2, 3)
    bars = plt.bar(models, training_times, color='salmon')
    plt.axhline(y=baseline_results['training_time'], linestyle='--', color='red', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)')
    plt.title('Absolute Training Time')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}s', ha='center', va='bottom', fontsize=9)

    # Relative training time changes
    plt.subplot(2, 2, 4)
    bars = plt.bar(models, time_changes, color='mediumpurple')
    plt.axhline(y=0, linestyle='--', color='red', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Training Time Change (%)')
    plt.title('Relative Training Time Change')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.suptitle(f'Neural Network Performance Analysis: {dataset_name}', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'{save_path}/nn_performance_changes_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create efficiency visualization (accuracy vs. training time)
    plt.figure(figsize=(10, 8))
    plt.scatter(training_times, accuracies, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')

    # Add model labels
    for i, model in enumerate(models):
        plt.annotate(model, (training_times[i], accuracies[i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title(f'Efficiency Analysis: Accuracy vs. Training Time - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/nn_efficiency_{dataset_name.replace(" ", "_").lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Detailed analysis
    print(f"\nNeural Network Performance Analysis for {dataset_name}:")
    print("\nAbsolute Metrics:")
    for i, model in enumerate(models):
        print(f"{model}: Accuracy={accuracies[i]:.4f}, F1={f1_scores[i]:.4f}, Time={training_times[i]:.3f}s")

    print("\nRelative Changes (vs. Baseline):")
    for i, model in enumerate(models):
        if i > 0:  # Skip baseline
            print(
                f"{model}: Accuracy={accuracy_changes[i]:+.1f}%, F1={f1_changes[i]:+.1f}%, Time={time_changes[i]:+.1f}%")

    # Calculate efficiency score (accuracy / training time)
    efficiency_scores = [acc / time for acc, time in zip(accuracies, training_times)]
    print("\nEfficiency Scores (Accuracy per Second):")
    for i, model in enumerate(models):
        print(f"{model}: {efficiency_scores[i]:.4f}")

    best_efficiency_idx = np.argmax(efficiency_scores)
    print(f"\nMost efficient model: {models[best_efficiency_idx]} "
          f"({efficiency_scores[best_efficiency_idx]:.4f})")

    return {
        'models': models,
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'training_times': training_times,
        'accuracy_changes': accuracy_changes,
        'f1_changes': f1_changes,
        'time_changes': time_changes,
        'efficiency_scores': efficiency_scores,
        'best_efficiency_model': models[best_efficiency_idx]
    }



def evaluate_hypotheses(all_results, save_path='results'):
    """
    Evaluates the hypotheses formulated at the beginning of the analysis.
    """
    import pandas as pd

    print("\n====== Hypothesis Evaluation ======")

    # Hypothesis 1: EM vs K-Means on imbalanced data
    h1_result = {}
    if ('marketing_kmeans' in all_results and 'marketing_em' in all_results and
            'marketing_clustering_comparison' in all_results):

        kmeans_sil = max(all_results['marketing_kmeans']['silhouette_scores'])

        # Compare K-Means and EM on Marketing data
        print(
            "\nH1: On imbalanced data, algorithms with geometric flexibility (EM) outperform rigid approaches (K-Means)")
        print(f"K-Means silhouette score on Marketing: {kmeans_sil:.4f}")

        # If we have alignment metrics
        if 'marketing_kmeans_alignment' in all_results and 'marketing_em_alignment' in all_results:
            kmeans_ari = all_results['marketing_kmeans_alignment']['ari']
            em_ari = all_results['marketing_em_alignment']['ari']
            print(f"K-Means ARI with true labels: {kmeans_ari:.4f}")
            print(f"EM ARI with true labels: {em_ari:.4f}")

            h1_supported = em_ari > kmeans_ari
            h1_confidence = abs(em_ari - kmeans_ari) / max(em_ari, kmeans_ari) * 100

            h1_result = {
                'supported': h1_supported,
                'confidence': h1_confidence,
                'kmeans_ari': kmeans_ari,
                'em_ari': em_ari
            }

            print(f"Hypothesis 1 is {'supported' if h1_supported else 'not supported'} "
                  f"(Confidence: {h1_confidence:.1f}%)")
        else:
            print("Insufficient data to fully evaluate Hypothesis 1")
    else:
        print("Missing results needed to evaluate Hypothesis 1")

    # Hypothesis 2: Impact of DR on clustering
    h2_result = {}
    if ('marketing_combined' in all_results and isinstance(all_results['marketing_combined'], tuple) and
            len(all_results['marketing_combined']) > 1):

        combined_comparison = all_results['marketing_combined'][1]  # This is the comparison DataFrame

        print("\nH2: Dimensionality reduction techniques impact clustering quality differently")

        # Extract silhouette scores for each DR method
        dr_methods = []
        silhouette_scores = []

        for algo in combined_comparison['Algorithm']:
            if 'K-Means + ' in algo:
                dr_method = algo.split('K-Means + ')[1]
                dr_methods.append(dr_method)
                idx = combined_comparison[combined_comparison['Algorithm'] == algo].index[0]
                silhouette_scores.append(combined_comparison.loc[idx, 'Silhouette Score'])

        if dr_methods and silhouette_scores:
            best_dr = dr_methods[silhouette_scores.index(max(silhouette_scores))]
            worst_dr = dr_methods[silhouette_scores.index(min(silhouette_scores))]

            silhouette_range = max(silhouette_scores) - min(silhouette_scores)
            silhouette_avg = sum(silhouette_scores) / len(silhouette_scores)
            variation = silhouette_range / silhouette_avg * 100

            h2_supported = variation > 20  # Arbitrary threshold for "significant" variation
            h2_confidence = variation

            print(f"Best DR method for clustering: {best_dr} (Silhouette: {max(silhouette_scores):.4f})")
            print(f"Worst DR method for clustering: {worst_dr} (Silhouette: {min(silhouette_scores):.4f})")
            print(f"Variation in clustering quality: {variation:.1f}%")

            h2_result = {
                'supported': h2_supported,
                'confidence': h2_confidence,
                'best_dr': best_dr,
                'worst_dr': worst_dr,
                'silhouette_scores': dict(zip(dr_methods, silhouette_scores)),
                'variation_percentage': variation
            }

            print(f"Hypothesis 2 is {'supported' if h2_supported else 'not supported'} "
                  f"(Confidence: {variation:.1f}%)")
        else:
            print("Insufficient data to evaluate Hypothesis 2")
    else:
        print("Missing results needed to evaluate Hypothesis 2")

    # Hypothesis 3: PCA maintains accuracy while reducing dimensions and time
    h3_result = {}
    if 'marketing_nn_dr' in all_results and 'baseline' in all_results['marketing_nn_dr'] and 'pca' in all_results[
        'marketing_nn_dr']:
        baseline_acc = all_results['marketing_nn_dr']['baseline']['accuracy']
        pca_acc = all_results['marketing_nn_dr']['pca']['accuracy']

        baseline_time = all_results['marketing_nn_dr']['baseline']['training_time']
        pca_time = all_results['marketing_nn_dr']['pca']['training_time']

        # Calculate dimension reduction
        if 'marketing_pca' in all_results:
            original_dims = all_results['marketing_pca']['explained_variance'].shape[0]
            reduced_dims = all_results['marketing_pca']['n_components_threshold']
            dim_reduction = (original_dims - reduced_dims) / original_dims * 100
        else:
            dim_reduction = None

        print("\nH3: PCA maintains classification accuracy while reducing dimensionality and training time")
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        print(f"PCA accuracy: {pca_acc:.4f}")
        print(f"Accuracy change: {(pca_acc - baseline_acc) / baseline_acc * 100:+.1f}%")

        if dim_reduction is not None:
            print(f"Dimension reduction: {dim_reduction:.1f}%")

        print(f"Training time reduction: {(baseline_time - pca_time) / baseline_time * 100:.1f}%")

        acc_maintained = pca_acc >= baseline_acc * 0.95  # Allow for 5% drop
        time_reduced = pca_time < baseline_time * 0.9  # Require 10% reduction

        h3_supported = acc_maintained and time_reduced and dim_reduction and dim_reduction > 15

        confidence_factors = []
        if acc_maintained:
            confidence_factors.append(100 - abs((pca_acc - baseline_acc) / baseline_acc * 100))
        if time_reduced:
            confidence_factors.append((baseline_time - pca_time) / baseline_time * 100)
        if dim_reduction and dim_reduction > 15:
            confidence_factors.append(dim_reduction)

        h3_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0

        h3_result = {
            'supported': h3_supported,
            'confidence': h3_confidence,
            'accuracy_maintained': acc_maintained,
            'time_reduced': time_reduced,
            'dimension_reduction': dim_reduction,
            'accuracy_change_pct': (pca_acc - baseline_acc) / baseline_acc * 100,
            'time_reduction_pct': (baseline_time - pca_time) / baseline_time * 100
        }

        print(f"Hypothesis 3 is {'supported' if h3_supported else 'not supported'} "
              f"(Confidence: {h3_confidence:.1f}%)")
    else:
        print("Missing results needed to evaluate Hypothesis 3")

    # Hypothesis 4: Manifold learning produces better clusters
    h4_result = {}
    if ('manifold_clustering_comparison' in all_results and
            isinstance(all_results['manifold_clustering_comparison'], pd.DataFrame) and
            'marketing_combined' in all_results):

        manifold_comparison = all_results['manifold_clustering_comparison']
        combined_comparison = all_results['marketing_combined'][1]

        print("\nH4: Manifold learning methods produce better clusters than linear techniques")

        # Get best manifold silhouette
        if not manifold_comparison.empty and 'K-Means Silhouette' in manifold_comparison.columns:
            best_manifold = manifold_comparison['K-Means Silhouette'].max()
            best_manifold_name = manifold_comparison.loc[
                manifold_comparison['K-Means Silhouette'].idxmax(), 'Embedding']

            # Get best linear DR silhouette
            linear_methods = ['K-Means + PCA', 'K-Means + ICA', 'K-Means + RP']
            linear_silhouettes = []

            for method in linear_methods:
                if any(combined_comparison['Algorithm'] == method):
                    linear_silhouettes.append(
                        combined_comparison.loc[combined_comparison['Algorithm'] == method, 'Silhouette Score'].values[
                            0]
                    )

            if linear_silhouettes:
                best_linear = max(linear_silhouettes)
                best_linear_idx = linear_silhouettes.index(best_linear)
                best_linear_name = linear_methods[best_linear_idx]

                print(f"Best manifold method: {best_manifold_name} (Silhouette: {best_manifold:.4f})")
                print(f"Best linear method: {best_linear_name} (Silhouette: {best_linear:.4f})")
                print(f"Improvement: {(best_manifold - best_linear) / best_linear * 100:+.1f}%")

                h4_supported = best_manifold > best_linear
                h4_confidence = (best_manifold - best_linear) / best_linear * 100

                h4_result = {
                    'supported': h4_supported,
                    'confidence': h4_confidence,
                    'best_manifold_method': best_manifold_name,
                    'best_manifold_silhouette': best_manifold,
                    'best_linear_method': best_linear_name,
                    'best_linear_silhouette': best_linear,
                    'improvement_percentage': (best_manifold - best_linear) / best_linear * 100
                }

                print(f"Hypothesis 4 is {'supported' if h4_supported else 'not supported'} "
                      f"(Confidence: {abs(h4_confidence):.1f}%)")
            else:
                print("Insufficient linear DR results to evaluate Hypothesis 4")
        else:
            print("Insufficient manifold clustering results to evaluate Hypothesis 4")
    else:
        print("Missing results needed to evaluate Hypothesis 4")

    # Create a summary of hypothesis evaluation
    hypothesis_summary = {
        'H1': h1_result,
        'H2': h2_result,
        'H3': h3_result,
        'H4': h4_result
    }

    return hypothesis_summary



# Main execution
if __name__ == "__main__":
    main()
