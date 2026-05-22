"""Phase 3 GraphCast: coarse-scale activation perturbation.

Mechanism (Option A from the Phase 3 design):
    After the grid2mesh GNN encoder runs, multiply the latent features of the
    first ``n_coarse_nodes`` mesh nodes by ``(1 + sigma * N(0, 1))``. These
    indices correspond to the coarsest refinement levels of the icosahedral
    multi-mesh (level-0 has nodes 0..11, level-0+1 has nodes 0..41), so we
    perturb the model's planetary-scale latent representation only.

The mesh-vertex index preservation across refinement levels is guaranteed by
deepmind's ``icosahedral_mesh.merge_meshes``, which asserts that coarser-mesh
vertices appear as the initial prefix of the finer mesh vertices. So
``latent_mesh_nodes[:42]`` are exactly the level-0+1 vertices.

This is NOT weight perturbation: it cannot go through
``materialise_perturbed_package``. Instead, we monkey-patch
``graphcast.GraphCast`` with a subclass before earth2studio's loader runs, so
the resulting model object has the perturbation baked into its forward pass.
Per-member randomness is supplied via ``model.prng_key``, which the caller
should set to a member-specific ``jax.random.PRNGKey`` after model load.
"""

from __future__ import annotations

_INSTALLED = False
_ORIGINAL_GRAPHCAST_CLS = None


def install(sigma: float, n_coarse_nodes: int) -> None:
    """Monkey-patch deepmind's GraphCast with a coarse-perturbed subclass.

    Must be called BEFORE the GraphCastOperational loader runs. The subclass
    closes over ``sigma`` and ``n_coarse_nodes`` via module-level constants.

    Parameters
    ----------
    sigma : float
        Multiplicative perturbation magnitude. 0 disables the hook.
    n_coarse_nodes : int
        Number of leading mesh-node indices to perturb. Level-0 = 12,
        level-0+1 = 42 for any operational mesh_size >= 1.
    """
    global _INSTALLED, _ORIGINAL_GRAPHCAST_CLS
    if sigma <= 0 or n_coarse_nodes <= 0:
        return

    import haiku as hk
    import jax
    from graphcast import graphcast as _gc

    if _ORIGINAL_GRAPHCAST_CLS is None:
        _ORIGINAL_GRAPHCAST_CLS = _gc.GraphCast

    sigma_const = float(sigma)
    n_coarse_const = int(n_coarse_nodes)

    class CoarsePerturbedGraphCast(_ORIGINAL_GRAPHCAST_CLS):  # type: ignore[misc, valid-type]
        def _run_grid2mesh_gnn(self, grid_node_features):  # noqa: D401
            latent_mesh_nodes, latent_grid_nodes = super()._run_grid2mesh_gnn(grid_node_features)
            noise_key = hk.next_rng_key()
            n_total = latent_mesh_nodes.shape[0]
            n = min(n_coarse_const, n_total)
            # latent_mesh_nodes shape: [n_mesh_nodes, ..., latent_dim] (typically
            # [40962, 1, 512] for the operational mesh). Noise must match the
            # full slice shape, not just the leading axis.
            coarse_slice = latent_mesh_nodes[:n]
            noise = jax.random.normal(
                noise_key,
                shape=coarse_slice.shape,
                dtype=latent_mesh_nodes.dtype,
            )
            perturbed = coarse_slice * (1.0 + sigma_const * noise)
            latent_mesh_nodes = latent_mesh_nodes.at[:n].set(perturbed)
            return latent_mesh_nodes, latent_grid_nodes

    _gc.GraphCast = CoarsePerturbedGraphCast
    _INSTALLED = True


def uninstall() -> None:
    """Restore the original deepmind GraphCast class (for tests / cleanup)."""
    global _INSTALLED, _ORIGINAL_GRAPHCAST_CLS
    if _ORIGINAL_GRAPHCAST_CLS is None:
        return
    from graphcast import graphcast as _gc

    _gc.GraphCast = _ORIGINAL_GRAPHCAST_CLS
    _INSTALLED = False


__all__ = ["install", "uninstall"]
