r"""
Host-side helpers to assemble a *tiled* batch for
:meth:`EwaldCalculator.forward_batched <torchpme.EwaldCalculator.forward_batched>`.

The batched Ewald evaluation follows two ideas (see the ``forward_batched``
documentation for the evaluation itself):

1. A single target half-space k-vector count ``num_k`` is the knob that balances the
   real- and reciprocal-space work for every system in a batch. It fixes, per system
   :math:`b`, the reciprocal resolution
   :math:`\lambda_b = (\prod_i L_{b,i} / (2\,\mathrm{num\_k}))^{1/3}`, the smearing
   :math:`\sigma_b = 2\lambda_b` and the real-space cutoff
   :math:`r_c(b) = 4\sigma_b = 8\lambda_b` (use :func:`ewald_params_from_num_k` to
   compute the cutoff at which each system's neighbor list must be built).
2. The reciprocal-space atom :math:`\times` k-vector sum is block-diagonal across
   systems. :func:`prepare_tiled_batch` lays atoms out in a flat sum-padded array,
   pads every system's k-vectors to a common ``K_pad`` and enumerates only the
   on-diagonal ``(system, atom_tile, k_tile)`` work blocks in a dispatch table, so the
   dense ``[n_systems, max_atoms, max_kvectors]`` rectangle is never materialized.

All functions here run on the host once per batch (or once per dataset for
:func:`ewald_params_from_num_k`); nothing needs to be differentiable.
"""

import warnings

import torch

__all__ = [
    "ewald_params_from_num_k",
    "prepare_tiled_batch",
    "shrink_2d_cell",
]


def shrink_2d_cell(
    cell: torch.Tensor, periodic: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    r"""
    Replace the vacuum-padded axis of a 2D-periodic cell by a shorter, physics-safe one.

    For a slab with a large vacuum gap, the length of the non-periodic cell vector
    inflates the cell-length product and with it the derived smearing, k-vector count
    and real-space cutoff. This helper shrinks the non-periodic vector to
    ``thickness + 1.5 * L_max`` (with ``L_max`` the longest periodic vector), which
    keeps the residual error of the slab treatment at :math:`\approx e^{-3\pi}`.
    The cell is returned unchanged if it is not 2D-periodic or already short enough.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param periodic: torch.tensor of shape ``(3,)`` and dtype bool
    :param positions: torch.tensor of shape ``(n_atoms, 3)``
    :return: torch.tensor of shape ``(3, 3)`` with the (possibly) shrunk cell
    """
    if int(periodic.sum()) != 2:
        return cell

    axis = int(torch.argmax((~periodic).to(torch.int64)))
    vac = cell[axis]
    height = torch.linalg.norm(vac)
    normal = vac / height

    # slab thickness as the extent of the atoms along the vacuum-axis direction
    z = positions @ normal
    thickness = z.max() - z.min()

    lengths = torch.linalg.norm(cell, dim=-1)
    l_max = max(lengths[i] for i in range(3) if i != axis)  # longest periodic vector
    h_min = thickness + 1.5 * l_max

    if height <= h_min:
        return cell

    new_cell = cell.clone()
    new_cell[axis] = normal * h_min
    return new_cell


def ewald_params_from_num_k(
    cell: torch.Tensor, periodic: torch.Tensor, num_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Derive the per-system Ewald parameters from the target k-vector count ``num_k``.

    Inverting the half-space k-count formula
    :math:`\mathrm{num\_k} \approx \tfrac{1}{2}\prod_i \lceil L_i/\lambda\rceil` gives
    the reciprocal resolution :math:`\lambda`, from which the smearing
    :math:`\sigma = 2\lambda` and the real-space cutoff
    :math:`r_c = 4\sigma = 8\lambda` follow. ``r_c`` is the radius at which the
    system's neighbor list should be built before calling
    :func:`prepare_tiled_batch`; since it grows with the cell size, it can also be
    evaluated once for the largest cell of a dataset and used for every structure.

    For 2D-periodic systems pass the *effective* (vacuum-shrunk) cell, see
    :func:`shrink_2d_cell`. Non-periodic systems (``periodic`` all ``False``) have no
    reciprocal part; they get placeholder values ``lambda = 0``, ``sigma = 1``,
    ``r_c = 0`` (their real-space part is the bare potential over the caller-provided
    pair list).

    :param cell: torch.tensor of shape ``(3, 3)`` or batched ``(n_systems, 3, 3)``
    :param periodic: torch.tensor of dtype bool and shape ``(3,)`` or batched
        ``(n_systems, 3)``
    :param num_k: target half-space k-vector count per system
    :return: tuple ``(lambda, sigma, cutoff)``, each of shape ``()`` for a single
        system or ``(n_systems,)`` for a batch
    """
    single = cell.dim() == 2
    cells = cell.unsqueeze(0) if single else cell
    periodics = periodic.unsqueeze(0) if single else periodic

    lengths = torch.linalg.norm(cells, dim=-1)
    lam = (lengths.prod(dim=-1) / (2.0 * num_k)) ** (1.0 / 3.0)
    sigma = 2.0 * lam
    cutoff = 4.0 * sigma

    nonperiodic = ~periodics.any(dim=-1)
    lam = torch.where(nonperiodic, torch.zeros_like(lam), lam)
    sigma = torch.where(nonperiodic, torch.ones_like(sigma), sigma)
    cutoff = torch.where(nonperiodic, torch.zeros_like(cutoff), cutoff)

    if single:
        return lam[0], sigma[0], cutoff[0]
    return lam, sigma, cutoff


def _integer_kgrid(ns: list[int], halfspace: bool) -> torch.Tensor:
    """
    Integer k-grid (FFT-frequency convention) for per-axis extents ``ns``.

    With ``halfspace=True`` only one of each ±k pair is kept, following the
    lexicographic rule ``(kx>0) | (kx==0 & ky>0) | (kx==0 & ky==0 & kz>0)``, which
    also excludes k = 0; each kept vector then stands in for its -k partner
    (degeneracy factor 2 in the evaluation). With ``halfspace=False`` the full grid
    (including k = 0, which the potentials map to a zero kernel) is returned.
    """
    freqs = [(torch.fft.fftfreq(n) * n).round().to(torch.long) for n in ns]
    kx, ky, kz = torch.meshgrid(freqs[0], freqs[1], freqs[2], indexing="ij")
    grid = torch.stack([kx.flatten(), ky.flatten(), kz.flatten()], dim=-1)

    if not halfspace:
        return grid

    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
    keep = (x > 0) | ((x == 0) & (y > 0)) | ((x == 0) & (y == 0) & (z > 0))
    return grid[keep]


def prepare_tiled_batch(
    positions: list[torch.Tensor],
    charges: list[torch.Tensor],
    cells: list[torch.Tensor],
    periodic: list[torch.Tensor],
    neighbor_indices: list[torch.Tensor],
    neighbor_distances: list[torch.Tensor],
    num_k: int,
    block_atoms: int = 32,
    block_kvecs: int = 128,
    halfspace: bool = True,
    k_pad_fraction: float = 0.1,
    smearing: float | torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    r"""
    Collate a list of systems into a concatenated batch plus the static tiling data
    consumed by :meth:`EwaldCalculator.forward_batched
    <torchpme.EwaldCalculator.forward_batched>`.

    Each system may be 3D-periodic, 2D-periodic (a slab with vacuum along the
    non-periodic axis; the effective cell is shrunk internally, see
    :func:`shrink_2d_cell`) or non-periodic. 1D periodicity is not supported. The
    neighbor list of every *periodic* system must have been built beforehand at that
    system's own cutoff ``r_c(b)`` from :func:`ewald_params_from_num_k` (or any larger
    radius); non-periodic systems provide the pair list over which the *bare*
    (non-range-separated) potential is summed, typically all pairs.

    All returned tensors live on the CPU; move them to the target device with
    ``.to(device)`` before calling the calculator.

    :param positions: per-system tensors of shape ``(n_atoms_b, 3)``
    :param charges: per-system tensors of shape ``(n_atoms_b, n_channels)``
    :param cells: per-system tensors of shape ``(3, 3)``
    :param periodic: per-system bool tensors of shape ``(3,)``
    :param neighbor_indices: per-system tensors of shape ``(n_pairs_b, 2)`` with
        *local* atom indices (offset internally)
    :param neighbor_distances: per-system tensors of shape ``(n_pairs_b,)``
    :param num_k: target half-space k-vector count per system (the accuracy/cost knob)
    :param block_atoms: atom-tile size ``BM`` of the reciprocal-space kernel
    :param block_kvecs: k-tile size ``BK`` of the reciprocal-space kernel
    :param halfspace: keep only one of each ±k pair (with degeneracy factor 2)
    :param k_pad_fraction: padding window of the common k-count: the k-axis is padded
        to at least ``(1 + k_pad_fraction) * num_k`` vectors so that batches sharing
        the same ``num_k`` get identical shapes
    :param smearing: optional override of the derived per-system smearing (a scalar or
        a tensor of shape ``(n_systems,)``); the k-grids are still sized by ``num_k``
    :return: tuple ``(batch, tiling)`` of two dictionaries of tensors. ``batch`` holds
        the concatenated inputs of ``forward_batched`` (``positions``, ``charges``,
        ``cell``, ``periodic``, ``system_index``, ``neighbor_indices``,
        ``neighbor_distances``); ``tiling`` holds the static index tensors and
        parameters (its integer metadata is stored as 0-dim tensors so that it stays a
        ``Dict[str, Tensor]`` for TorchScript).
    """
    n_systems = len(positions)
    if not (
        len(charges)
        == len(cells)
        == len(periodic)
        == len(neighbor_indices)
        == len(neighbor_distances)
        == n_systems
    ):
        raise ValueError("All per-system input lists must have the same length")
    if n_systems == 0:
        raise ValueError("Cannot collate an empty list of systems")
    if block_atoms < 1 or block_kvecs < 1:
        raise ValueError("`block_atoms` and `block_kvecs` must be positive")

    dtype = positions[0].dtype
    if charges[0].dim() != 2:
        raise ValueError(
            "`charges` must be 2-dimensional tensors [n_atoms, n_channels]"
        )
    n_channels = charges[0].shape[1]
    for q in charges:
        if q.dim() != 2 or q.shape[1] != n_channels:
            raise ValueError("All systems must have the same number of charge channels")

    n_periodic_axes = [int(p.sum()) for p in periodic]
    if any(n == 1 for n in n_periodic_axes):
        raise ValueError("1D-periodic systems are not supported")
    is_periodic = [n > 0 for n in n_periodic_axes]

    # --- effective (2D-shrunk) cells and per-system Ewald parameters ---
    effective_cells = torch.stack(
        [
            shrink_2d_cell(c, p, pos)
            for c, p, pos in zip(cells, periodic, positions, strict=True)
        ]
    )
    periodic_batch = torch.stack(periodic)
    lam, sigma, _ = ewald_params_from_num_k(effective_cells, periodic_batch, num_k)
    if smearing is not None:
        override = torch.as_tensor(smearing, dtype=dtype)
        sigma = torch.where(
            periodic_batch.any(dim=-1), override.expand(n_systems), sigma
        )
    sigma = sigma.to(dtype)

    # --- concatenated atom layout ---
    n_atoms = [pos.shape[0] for pos in positions]
    atom_offsets = [0]
    for n in n_atoms:
        atom_offsets.append(atom_offsets[-1] + n)
    system_index = torch.repeat_interleave(
        torch.arange(n_systems), torch.tensor(n_atoms)
    )

    # --- pair layout: screened (periodic systems) first, bare (non-periodic) after ---
    screened_indices, screened_distances, screened_sigma = [], [], []
    bare_indices, bare_distances = [], []
    for b in range(n_systems):
        offset_pairs = neighbor_indices[b] + atom_offsets[b]
        if is_periodic[b]:
            screened_indices.append(offset_pairs)
            screened_distances.append(neighbor_distances[b])
            screened_sigma.append(sigma[b].expand(neighbor_indices[b].shape[0]))
        else:
            bare_indices.append(offset_pairs)
            bare_distances.append(neighbor_distances[b])
    empty_indices = torch.empty(0, 2, dtype=torch.long)
    empty_distances = torch.empty(0, dtype=dtype)
    all_indices = torch.cat(screened_indices + bare_indices + [empty_indices])
    all_distances = torch.cat(screened_distances + bare_distances + [empty_distances])
    sigma_pair = torch.cat(screened_sigma + [empty_distances])
    n_screened_pairs = sigma_pair.shape[0]

    # --- per-system integer k-grids, padded to a common K_pad ---
    pbc_system = [b for b in range(n_systems) if is_periodic[b]]
    k_grids = []
    for b in pbc_system:
        lengths = torch.linalg.norm(effective_cells[b], dim=-1)
        ns = [max(1, int(torch.ceil(lengths[i] / lam[b]))) for i in range(3)]
        k_grids.append(_integer_kgrid(ns, halfspace))

    # pad the k-axis to at least (1 + k_pad_fraction)·num_k so that batches sharing
    # the same num_k get the same K_pad; a system whose per-axis rounding overshoots
    # the window still fits (K_pad grows), but shape stability across batches is lost
    k_floor = int((1.0 + k_pad_fraction) * num_k * (1 if halfspace else 2))
    k_realized = max([0] + [g.shape[0] for g in k_grids])
    if k_realized > k_floor:
        warnings.warn(
            f"A system realizes {k_realized} k-vectors, above the padding window "
            f"of {k_floor} for num_k={num_k}; the padded k-count grows accordingly. "
            "Increase `k_pad_fraction` to keep identical shapes across batches.",
            stacklevel=2,
        )
    k_pad = -(-max(k_floor, k_realized) // block_kvecs) * block_kvecs
    n_pbc = len(pbc_system)
    k_int = torch.zeros(n_pbc, k_pad, 3, dtype=torch.long)
    for row, grid in enumerate(k_grids):
        k_int[row, : grid.shape[0]] = grid

    # --- flat sum-padded atom layout over the periodic systems ---
    n_padded = [-(-n_atoms[b] // block_atoms) * block_atoms for b in pbc_system]
    pbc_atom_off = [0]
    for n in n_padded:
        pbc_atom_off.append(pbc_atom_off[-1] + n)
    n_flat = pbc_atom_off[-1]

    flat_to_atom = torch.zeros(n_flat, dtype=torch.long)
    flat_mask = torch.zeros(n_flat, dtype=torch.bool)
    for row, b in enumerate(pbc_system):
        start = pbc_atom_off[row]
        flat_to_atom[start : start + n_atoms[b]] = torch.arange(
            atom_offsets[b], atom_offsets[b] + n_atoms[b]
        )
        flat_mask[start : start + n_atoms[b]] = True

    # --- dispatch table: on-diagonal (system, atom_tile, k_tile) triples ---
    # outer atom-tile / inner k-tile order is load-bearing: it makes each
    # (system, atom_tile) group contiguous, so the kernel's second pass can reduce
    # with a reshape instead of a scatter
    n_kvec_tiles = k_pad // block_kvecs
    n_atom_tiles = torch.tensor([n // block_atoms for n in n_padded], dtype=torch.long)
    counts = n_atom_tiles * n_kvec_tiles
    n_triples = int(counts.sum())
    cum = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    b_col = torch.repeat_interleave(torch.arange(n_pbc), counts)
    inner = torch.arange(n_triples) - torch.repeat_interleave(cum, counts)
    mt_col = inner // n_kvec_tiles
    kt_col = inner % n_kvec_tiles

    flat_slots = (
        torch.tensor(pbc_atom_off[:-1], dtype=torch.long)[b_col] + mt_col * block_atoms
    ).unsqueeze(-1) + torch.arange(block_atoms)
    atom_gather = flat_to_atom[flat_slots]
    gather_mask = flat_mask[flat_slots]

    batch = {
        "positions": torch.cat(list(positions)),
        "charges": torch.cat(list(charges)),
        "cell": torch.stack(list(cells)).to(dtype),
        "periodic": periodic_batch,
        "system_index": system_index,
        "neighbor_indices": all_indices,
        "neighbor_distances": all_distances,
    }
    tiling = {
        "k_int": k_int,
        "sigma": sigma,
        "sigma_pair": sigma_pair,
        "pbc_system": torch.tensor(pbc_system, dtype=torch.long),
        "periodic_rows": periodic_batch[pbc_system]
        if n_pbc > 0
        else torch.zeros(0, 3, dtype=torch.bool),
        "effective_cell_static": effective_cells[pbc_system].to(dtype)
        if n_pbc > 0
        else torch.zeros(0, 3, 3, dtype=dtype),
        "b_col": b_col,
        "kt_col": kt_col,
        "atom_gather": atom_gather,
        "gather_mask": gather_mask,
        "flat_to_atom": flat_to_atom,
        "flat_mask": flat_mask,
        "pbc_atom": torch.tensor(is_periodic, dtype=torch.bool)[system_index],
        "block_atoms": torch.tensor(block_atoms, dtype=torch.long),
        "block_kvecs": torch.tensor(block_kvecs, dtype=torch.long),
        "n_kvec_tiles": torch.tensor(n_kvec_tiles, dtype=torch.long),
        "n_screened_pairs": torch.tensor(n_screened_pairs, dtype=torch.long),
        "g_factor": torch.tensor(2.0 if halfspace else 1.0, dtype=dtype),
    }
    return batch, tiling
