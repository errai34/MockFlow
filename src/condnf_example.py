import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from flowjax.flows import masked_autoregressive_flow, coupling_flow
from flowjax.distributions import Normal, Transformed
from flowjax.train import fit_to_data
from flowjax.bijections import RationalQuadraticSpline
import equinox as eqx
from typing import Tuple
import matplotlib.pyplot as plt

# Create a conditional normalizing flow for binary labels
class ConditionalFlow(eqx.Module):
    """Conditional normalizing flow for binary classification p(x|y)"""
    flow_0: Transformed
    flow_1: Transformed

    def __init__(self, dim: int, key: jr.PRNGKey, n_layers: int = 12, nn_width: int = 128, use_coupling: bool = False):
        """
        Initialize conditional flow with separate flows for each class.

        Args:
            dim: Dimension of the data
            key: PRNG key
            n_layers: Number of flow layers
            nn_width: Width of neural networks in the flow
            use_coupling: Whether to use coupling flows instead of MAF
        """
        key_0, key_1 = jr.split(key)

        # Create separate flows for each class with more capacity
        if use_coupling:
            # Use coupling flows which can better handle multimodal distributions
            self.flow_0 = coupling_flow(
                key=key_0,
                base_dist=Normal(jnp.zeros(dim), jnp.ones(dim) * 0.5),
                transformer=RationalQuadraticSpline(knots=8, interval=4),
                flow_layers=n_layers,
                nn_width=nn_width,
                nn_depth=3,
                invert=True
            )

            self.flow_1 = coupling_flow(
                key=key_1,
                base_dist=Normal(jnp.zeros(dim), jnp.ones(dim) * 0.5),
                transformer=RationalQuadraticSpline(knots=8, interval=4),
                flow_layers=n_layers,
                nn_width=nn_width,
                nn_depth=3,
                invert=True
            )
        else:
            # Use MAF with improved settings
            self.flow_0 = masked_autoregressive_flow(
                key=key_0,
                base_dist=Normal(jnp.zeros(dim), jnp.ones(dim) * 0.5),  # Smaller variance base
                flow_layers=n_layers,
                nn_width=nn_width,
                nn_depth=3,  # Deeper networks
                invert=True
            )

            self.flow_1 = masked_autoregressive_flow(
                key=key_1,
                base_dist=Normal(jnp.zeros(dim), jnp.ones(dim) * 0.5),  # Smaller variance base
                flow_layers=n_layers,
                nn_width=nn_width,
                nn_depth=3,  # Deeper networks
                invert=True
            )

    def log_prob(self, x: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log p(x|label) for given samples.

        Args:
            x: Data samples (n_samples, dim)
            label: Binary labels (n_samples,)

        Returns:
            Log probabilities (n_samples,)
        """
        # Compute log probabilities for each flow
        log_prob_0 = vmap(self.flow_0.log_prob)(x)
        log_prob_1 = vmap(self.flow_1.log_prob)(x)

        # Select based on label
        return jnp.where(label == 0, log_prob_0, log_prob_1)

    def sample(self, key: jr.PRNGKey, label: int, n_samples: int = 1) -> jnp.ndarray:
        """
        Sample from p(x|label).

        Args:
            key: PRNG key
            label: Label to condition on (0 or 1)
            n_samples: Number of samples to generate

        Returns:
            Samples (n_samples, dim)
        """
        if label == 0:
            return self.flow_0.sample(key, sample_shape=(n_samples,))
        else:
            return self.flow_1.sample(key, sample_shape=(n_samples,))


def train_conditional_flow(
    flow: ConditionalFlow,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    key: jr.PRNGKey,
    n_epochs: int = 5000,
    batch_size: int = 128,
    lr: float = 1e-3
) -> Tuple[ConditionalFlow, Tuple[list, list]]:
    """
    Train the conditional flow using maximum likelihood.

    Args:
        flow: Conditional flow model
        x_train: Training data (n_samples, dim)
        y_train: Binary labels (n_samples,)
        key: PRNG key
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Trained flow and loss history
    """

    # Separate data by class
    x_0 = x_train[y_train == 0]
    x_1 = x_train[y_train == 1]

    # Train each flow separately (more efficient than joint training)
    print("Training flow for class 0...")
    key_0, key_1 = jr.split(key)
    flow_0_trained, losses_0 = fit_to_data(
        key_0,
        flow.flow_0,
        x_0,
        max_epochs=n_epochs,
        batch_size=min(batch_size, len(x_0)),
        learning_rate=lr,
        max_patience=20,
        show_progress=True
    )

    print("\nTraining flow for class 1...")
    flow_1_trained, losses_1 = fit_to_data(
        key_1,
        flow.flow_1,
        x_1,
        max_epochs=n_epochs,
        batch_size=min(batch_size, len(x_1)),
        learning_rate=lr,
        max_patience=20,
        show_progress=True
    )

    # Update flow with trained components
    flow = eqx.tree_at(lambda f: (f.flow_0, f.flow_1), flow, (flow_0_trained, flow_1_trained))

    return flow, (losses_0, losses_1)


# Example usage and visualization
def generate_toy_data(key: jr.PRNGKey, n_samples: int = 1000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate 2D toy dataset with two classes."""
    key_0, key_1, key_label = jr.split(key, 3)

    # Generate labels
    labels = jr.bernoulli(key_label, 0.5, (n_samples,)).astype(jnp.int32)

    # Class 0: Gaussian mixture with tighter clusters
    n_0 = int(jnp.sum(labels == 0))
    key_0a, key_0b = jr.split(key_0)
    x_0_1 = jr.normal(key_0a, (n_0 // 2, 2)) * 0.3 + jnp.array([-2.5, 0])  # Tighter, more separated
    x_0_2 = jr.normal(key_0b, (n_0 - n_0 // 2, 2)) * 0.3 + jnp.array([2.5, 0])
    x_0 = jnp.vstack([x_0_1, x_0_2])

    # Class 1: Different Gaussian mixture with tighter clusters
    n_1 = n_samples - n_0
    key_1a, key_1b = jr.split(key_1)
    x_1_1 = jr.normal(key_1a, (n_1 // 2, 2)) * 0.3 + jnp.array([0, 2.5])  # Tighter, more separated
    x_1_2 = jr.normal(key_1b, (n_1 - n_1 // 2, 2)) * 0.3 + jnp.array([0, -2.5])
    x_1 = jnp.vstack([x_1_1, x_1_2])

    # Combine data
    x = jnp.zeros((n_samples, 2))
    x = x.at[labels == 0].set(x_0)
    x = x.at[labels == 1].set(x_1)

    return x, labels


def visualize_results(flow: ConditionalFlow, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey):
    """Visualize the original data and generated samples."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original data
    axes[0, 0].scatter(x[y == 0, 0], x[y == 0, 1], alpha=0.5, label='Class 0', c='blue')
    axes[0, 0].scatter(x[y == 1, 0], x[y == 1, 1], alpha=0.5, label='Class 1', c='red')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)

    # Generated samples for class 0
    key_0, key_1 = jr.split(key)
    samples_0 = flow.sample(key_0, 0, n_samples=500)
    axes[0, 1].scatter(samples_0[:, 0], samples_0[:, 1], alpha=0.5, c='blue')
    axes[0, 1].set_title('Generated Samples (Class 0)')
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)

    # Generated samples for class 1
    samples_1 = flow.sample(key_1, 1, n_samples=500)
    axes[0, 2].scatter(samples_1[:, 0], samples_1[:, 1], alpha=0.5, c='red')
    axes[0, 2].set_title('Generated Samples (Class 1)')
    axes[0, 2].set_xlim(-4, 4)
    axes[0, 2].set_ylim(-4, 4)

    # Density plots
    xx, yy = jnp.meshgrid(jnp.linspace(-4, 4, 100), jnp.linspace(-4, 4, 100))
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    # Density for class 0
    log_probs_0 = vmap(flow.flow_0.log_prob)(grid)
    probs_0 = jnp.exp(log_probs_0).reshape(100, 100)
    axes[1, 0].contourf(xx, yy, probs_0, levels=20, cmap='Blues')
    axes[1, 0].set_title('Learned Density p(x|y=0)')

    # Density for class 1
    log_probs_1 = vmap(flow.flow_1.log_prob)(grid)
    probs_1 = jnp.exp(log_probs_1).reshape(100, 100)
    axes[1, 1].contourf(xx, yy, probs_1, levels=20, cmap='Reds')
    axes[1, 1].set_title('Learned Density p(x|y=1)')

    # Combined density (weighted by class prior)
    combined = 0.5 * probs_0 + 0.5 * probs_1
    axes[1, 2].contourf(xx, yy, combined, levels=20, cmap='Purples')
    axes[1, 2].set_title('Combined Density p(x)')

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Set random seed
    jax.config.update("jax_enable_x64", True)
    key = jr.key(42)
    key_data, key_flow, key_train, key_viz = jr.split(key, 4)

    # Generate toy data
    print("Generating toy dataset...")
    x_train, y_train = generate_toy_data(key_data, n_samples=2000)
    print(f"Data shape: {x_train.shape}, Labels shape: {y_train.shape}")
    print(f"Class distribution: {int(jnp.sum(y_train == 0))} class 0, {int(jnp.sum(y_train == 1))} class 1")

    # Create conditional flow with more capacity - try coupling flows
    print("\nInitializing conditional flow with coupling layers...")
    flow = ConditionalFlow(dim=2, key=key_flow, n_layers=10, nn_width=128, use_coupling=True)

    # Train the flow with more epochs and smaller learning rate
    print("\nTraining conditional flow...")
    flow, losses = train_conditional_flow(
        flow, x_train, y_train, key_train,
        n_epochs=5000,  # Moderate epochs for testing
        batch_size=256,  # Larger batch size
        lr=3e-4  # Adjusted learning rate
    )

    # Evaluate log-likelihood on training data
    train_log_probs = flow.log_prob(x_train, y_train)
    print(f"\nMean log-likelihood on training data: {jnp.mean(train_log_probs):.3f}")

    # Visualize results
    print("\nVisualizing results...")
    visualize_results(flow, x_train, y_train, key_viz)

    # Example: Computing probability for new samples
    print("\nExample predictions:")
    test_points = jnp.array([[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, -2.0]])
    for point in test_points:
        log_p_0 = flow.flow_0.log_prob(point)
        log_p_1 = flow.flow_1.log_prob(point)
        p_0 = jnp.exp(log_p_0)
        p_1 = jnp.exp(log_p_1)
        # Assuming equal class priors for posterior calculation
        posterior_1 = p_1 / (p_0 + p_1)
        print(f"Point {point}: P(y=1|x) ≈ {posterior_1:.3f}")
