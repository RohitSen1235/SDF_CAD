# Generative Design Physics and PDE Design Note

This document defines the planned v1 physics formulation for the Generative Design / Optimization Module. It is the reference for how the first structural optimization solver should behave inside this SDF-based CAD product.

The scope here is intentionally narrow:

- Static linear structural analysis only
- Shared Cartesian SDF/voxel grid
- Design-space and non-design-space masking
- User-defined supports and loads
- Topology optimization by material removal
- No thermal coupling, no dynamics, no CFD, no contact, no nonlinear material behavior

## 1. Problem Definition and Modeling Assumptions

This product already represents geometry as a signed distance field `phi(x)` sampled on a regular 3D grid. The v1 structural solver will run on that same grid instead of introducing a separate body-fitted mesh pipeline. That keeps geometry, simulation, and topology updates in one common volumetric representation.

The optimization workflow starts from two uploaded solids:

- `design space`: material that may be removed during optimization
- `non-design space`: protected material that must remain unchanged

The user then selects:

- fixed support regions
- load points or load surfaces

These selections are mapped from viewer/world coordinates onto the solver grid and become the boundary conditions for the structural solve.

The v1 modeling assumptions are:

- Small-strain linear elasticity
- Quasi-static loading
- Isotropic homogeneous material in the design region
- Non-design space remains fully solid throughout optimization
- No contact
- No plasticity
- No fracture
- No buckling
- No dynamics

Under these assumptions, the solver answers a single question each iteration: given the current material layout, what displacement, strain, and stress field does the structure develop under the applied supports and loads?

## 2. Governing PDE Formulation

### 2.1 Unknown field

The primary unknown is the displacement field

```math
\mathbf{u}(\mathbf{x}) = [u_x(\mathbf{x}), u_y(\mathbf{x}), u_z(\mathbf{x})]^T
```

defined over the solid region `Omega_s` induced by the current design.

### 2.2 Infinitesimal strain

For small deformations, strain is

```math
\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}\left(\nabla \mathbf{u} + \nabla \mathbf{u}^T\right)
```

In components,

```math
\varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)
```

### 2.3 Linear elastic constitutive law

For isotropic linear elasticity,

```math
\boldsymbol{\sigma} = \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}) \mathbf{I} + 2\mu \boldsymbol{\varepsilon}
```

where `sigma` is the Cauchy stress tensor, `lambda` and `mu` are the Lamé parameters, and

```math
\mu = \frac{E}{2(1+\nu)}, \qquad
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
```

with `E` the Young's modulus and `nu` the Poisson ratio.

### 2.4 Static equilibrium PDE

The v1 structural solve uses the static equilibrium equation

```math
-\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) = \mathbf{f} \qquad \text{in } \Omega_s
```

where `f` is the body-force or equivalent external-load density.

Substituting the constitutive law gives the Navier-Cauchy form

```math
-(\lambda + \mu)\nabla(\nabla \cdot \mathbf{u}) - \mu \nabla^2 \mathbf{u} = \mathbf{f}
```

This is the authoritative PDE for v1.

### 2.5 Boundary conditions

Fixed supports are Dirichlet conditions:

```math
\mathbf{u} = \mathbf{0} \qquad \text{on } \Gamma_D
```

Applied loads are Neumann conditions:

```math
\boldsymbol{\sigma} \mathbf{n} = \mathbf{t} \qquad \text{on } \Gamma_N
```

where `n` is the outward unit normal and `t` is the applied traction. In practice, the UI may define point loads or surface loads; these are rasterized onto the grid and distributed onto the discrete right-hand side.

### 2.6 Objective and derived fields

The structural solve produces:

- displacement `u`
- strain tensor `epsilon`
- stress tensor `sigma`

The primary scalar visualization field should be von Mises stress:

```math
\sigma_{vm} = \sqrt{\frac{3}{2}\, \mathbf{s} : \mathbf{s}}
```

where `s` is the deviatoric stress tensor.

The primary optimization signal should be strain energy density or an equivalent compliance sensitivity proxy:

```math
w(\mathbf{x}) = \frac{1}{2}\boldsymbol{\sigma} : \boldsymbol{\varepsilon}
```

Global compliance can be written as

```math
C = \int_{\Omega_s} \mathbf{f}\cdot\mathbf{u}\, d\Omega
```

or equivalently as the total elastic energy under the linear static assumptions.

## 3. SDF and Material Representation

### 3.1 Geometry source of truth

The signed distance field `phi(x)` remains the canonical geometry representation. The zero level set defines the current boundary.

- `phi < 0`: inside solid
- `phi = 0`: boundary
- `phi > 0`: outside solid

For the optimization workflow, the uploaded meshes are voxelized onto a shared Cartesian grid to produce masks for:

- active design material
- protected non-design material
- void

### 3.2 Occupancy and density fields

The simplest v1 representation is a binary or near-binary occupancy field `rho(x)`:

```math
\rho(\mathbf{x}) \in [\rho_{min}, 1]
```

with:

- `rho = 1` for solid non-design voxels
- `rho = 1` initially for solid design voxels
- `rho = rho_min` in void or nearly removed regions

`rho_min > 0` is required so the discrete system never becomes singular when parts of the design space are almost fully removed.

The effective stiffness can be regularized using a SIMP-style interpolation even if the update logic is discrete:

```math
E_{eff}(\mathbf{x}) = E_{min} + \rho(\mathbf{x})^p (E_0 - E_{min})
```

where:

- `E_0` is the full material stiffness
- `E_min` is a small stiffness floor
- `p` is a penalization exponent

This gives a numerically robust bridge between fully solid and nearly void voxels.

### 3.3 Design and non-design masks

Two masks are maintained throughout optimization:

- `M_design(x)`: voxels that may change
- `M_keep(x)`: voxels that are permanently protected

The update rule must satisfy:

```math
M_{keep} \cap M_{design} = \emptyset
```

and the retained solid is always

```math
\Omega_s = \Omega_{keep} \cup \Omega_{design,active}
```

No optimization step is allowed to remove material from `Omega_keep`.

### 3.4 Rasterizing supports and loads

User selections are first captured in world coordinates and then rasterized to the grid:

- point picks become localized node or cell neighborhoods
- surface picks become boundary voxel sets

Fixed constraints create a constrained-degree-of-freedom mask. Loads create entries in the discrete force vector. Surface loads should be spread over the selected boundary voxels in a way that preserves total applied force.

### 3.5 Design update and SDF reconstruction

After each topology update, the active design occupancy field is converted back into an updated implicit representation. The v1 path should:

1. apply the new occupancy mask
2. union it with the non-design mask
3. rebuild or reinitialize a signed-distance-like field

The reinitialized field is then used for preview, meshing, and the next optimization iteration.

## 4. Discretization Scheme

### 4.1 Selected v1 path

The v1 solver should use:

- a regular Cartesian grid
- a voxel-based fixed-grid elasticity formulation
- node-centered displacement unknowns
- matrix-free stencil operator application

This is intentionally not a body-fitted FEM mesh. The discretization is designed around the same structured lattice already used by the SDF engine.

### 4.2 Unknown storage

Each active grid node stores three displacement unknowns:

```math
\mathbf{u}_{i,j,k} = [u_x, u_y, u_z]_{i,j,k}
```

Material fields such as `rho`, `E_eff`, and masks are stored on the same grid or on aligned cell-centered arrays, with interpolation rules fixed by implementation. The key v1 choice is that the displacement solve is node-centered.

### 4.3 Structured-grid elasticity operator

The continuous Navier-Cauchy operator is approximated using finite differences on the regular lattice. Spatial derivatives are evaluated by central stencils in the interior, producing a structured sparse operator with a fixed local neighborhood.

Instead of assembling a global sparse matrix explicitly, the solver applies the operator in matrix-free form:

```math
\mathbf{y} = A(\rho)\mathbf{x}
```

by evaluating the stencil directly over the grid.

This choice is better aligned with GPU execution than assembling and storing a large irregular sparse matrix.

### 4.4 Interface treatment

Near the SDF boundary, fully sharp accuracy is not expected in v1. Interface voxels are handled through embedded coefficients derived from occupancy and filtered density. This gives a robust approximation but introduces staircase/interface-smearing effects.

### 4.5 Boundary condition imposition

Dirichlet constraints should be enforced by constrained-DOF masking and elimination in the operator application:

- constrained nodes have zero displacement
- residual and search directions are projected to zero on constrained DOFs

This preserves a symmetric positive-definite reduced system, which is important for Conjugate Gradient.

Loads are inserted into the right-hand side vector after rasterization and distribution.

### 4.6 Accuracy expectations

The structured-grid solve is expected to be approximately second-order accurate in smooth interior regions when central differences are used. In practice, the dominant error sources in v1 will be:

- staircase boundary approximation
- interface coefficient smearing
- load/support rasterization error
- grid-resolution dependence near thin features and stress concentrations

## 5. GPU Implementation Strategy

### 5.1 Backend model

The implementation should follow the current backend pattern:

- `cupy` for CUDA execution
- `numpy` fallback for CPU execution
- shared API shape across both backends

The discrete solver should be written around array operations and small custom kernels so the same logic can run on both backends where feasible.

### 5.2 Core array layout

The planned field layout is:

- `ux`, `uy`, `uz`: displacement components
- `fx`, `fy`, `fz`: force components
- `rho`: material density/occupancy
- `E_eff`: effective stiffness field
- `mask_design`
- `mask_keep`
- `mask_fixed_x`, `mask_fixed_y`, `mask_fixed_z`
- derived scalar fields such as `sigma_vm` and `energy_density`

Component-wise arrays are preferred over an array-of-structs layout because they simplify coalesced stencil access and backend portability.

### 5.3 Matrix-free kernels

The main GPU work consists of:

- stencil application kernels for `y = A x`
- residual computation
- vector update kernels for CG
- dot-product and norm reductions
- post-processing kernels for strain, stress, and energy density
- local filter kernels for topology sensitivity smoothing

Because the grid is regular, memory access is predictable and domain decomposition is straightforward.

### 5.4 Memory strategy

The GPU plan should assume:

- reuse of masks and material fields across iterations
- persistent buffers for CG vectors where possible
- minimal host-device transfers during the solve
- host-device synchronization only at convergence checks, streamed summaries, or final payload generation

The current repo already has `cpu/cuda` backend selection and CuPy-based GPU memory cleanup. The structural solver should follow that model rather than inventing a separate execution stack.

## 6. Linear Solver and Numerical Scheme Details

### 6.1 Discrete system

After discretization and constraint application, the equilibrium solve becomes

```math
A(\rho)\mathbf{u} = \mathbf{b}
```

where:

- `u` is the stacked displacement vector
- `b` is the external load vector
- `A(rho)` is the density-dependent stiffness operator

With fixed supports enforced by elimination/projection and with a positive stiffness floor, the reduced system is treated as symmetric positive-definite.

### 6.2 Selected v1 iterative solver

The v1 solver should use:

- matrix-free Conjugate Gradient
- Jacobi preconditioning
- CPU fallback using the same algorithmic flow

CG is appropriate because:

- the constrained linear elasticity system is expected to be SPD
- it avoids storing a full matrix
- it maps cleanly to repeated stencil applications and global reductions on GPU

### 6.3 Preconditioner

The initial preconditioner should be diagonal/Jacobi:

```math
M^{-1} \approx \mathrm{diag}(A)^{-1}
```

This is cheap to compute, easy to apply in parallel, and a reasonable v1 starting point. Block-Jacobi or multigrid-style methods can be considered later if convergence is too slow on large problems.

### 6.4 Constrained DOF handling

The chosen v1 method is masked projection/elimination:

- zero prescribed displacement values on constrained DOFs
- zero residual entries on constrained DOFs
- zero search-direction entries on constrained DOFs
- ensure operator application respects the same mask

This keeps the CG iteration consistent with the reduced problem without explicitly rebuilding index maps for every solve.

### 6.5 Stopping conditions

The solve should stop when either:

- the relative residual norm falls below a configured tolerance, for example `1e-6`
- the iteration count reaches a configured maximum, for example `200` to `1000` depending on grid size

The document should treat these numbers as engineering defaults, not permanent API promises.

### 6.6 Conditioning and regularization

The system becomes ill-conditioned when:

- the design contains very thin connections
- large void fractions develop
- constrained regions are insufficient
- loads are applied in poorly supported regions

To avoid singular or nearly singular systems, v1 should enforce:

- minimum stiffness `E_min`
- minimum density `rho_min`
- validation that at least one meaningful fixed support exists
- validation that the remaining solid graph is not fully disconnected from the supports

## 7. Stress, Strain, and Removal Criterion

### 7.1 Post-processing

After solving for displacement, strain is computed from finite-difference displacement gradients:

```math
\boldsymbol{\varepsilon} = \frac{1}{2}\left(\nabla \mathbf{u} + \nabla \mathbf{u}^T\right)
```

Stress is then recovered from the constitutive law:

```math
\boldsymbol{\sigma} = \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}) \mathbf{I} + 2\mu \boldsymbol{\varepsilon}
```

### 7.2 Scalar output fields

Two scalar fields should be treated differently:

- `von Mises stress`: user-facing visualization field
- `strain energy density` or compliance sensitivity proxy: removal-ranking field

This distinction matters because pure stress thresholding is not a stable topology optimization strategy. Removing all low-stress material can disconnect load paths, produce brittle checkerboards, or preserve misleading local stress concentrations.

### 7.3 Removal ranking

The v1 removal signal should be based on low strain-energy contribution within the design region. Conceptually:

- low-energy voxels are candidates for removal
- high-energy load-path voxels are retained
- non-design voxels are never ranked for removal

This makes the update closer to compliance-driven topology optimization than to naive stress clipping.

### 7.4 Filtering and regularization

Before removal, the raw ranking field should be spatially filtered to impose a minimum feature size and suppress voxel-scale noise. A local averaging or density filter is sufficient for v1.

The filter serves several purposes:

- reduces checkerboarding
- suppresses single-voxel islands
- discourages numerically unstable thin members
- improves iteration-to-iteration smoothness

## 8. Topology Optimization and Update Scheme

The planned v1 optimization loop is:

1. Rasterize uploaded design and non-design solids onto the shared grid.
2. Initialize `rho` for the active design and protected non-design regions.
3. Apply supports and loads from the user-defined selections.
4. Solve `A(rho)u = b`.
5. Compute strain, stress, von Mises stress, and energy density.
6. Build a removal score from the energy-density or compliance-sensitivity field.
7. Filter the score spatially.
8. Remove or attenuate low-utility material in the design region only.
9. Preserve the full non-design region.
10. Rebuild the combined solid field and reinitialize the SDF.
11. Repeat until the target volume fraction, convergence criterion, or iteration budget is reached.

The v1 optimization style is topology-first:

- material removal on the voxel field
- no smooth level-set boundary velocity law yet
- no topological guarantees beyond filtering and protected masks
- target volume fraction is the main global constraint

Convergence can be judged using one or both of:

- objective stabilization
- small material-change fraction between successive iterations

## 9. Boundary Condition Handling

### 9.1 Viewer-to-grid conversion

The frontend lets the user select:

- fixed points or surfaces
- load points or surfaces

These selections are converted from viewer coordinates into grid-space masks using the same world-space bounds as the SDF domain.

### 9.2 Fixed supports

Fixed supports become zero-displacement constraints. In v1, full clamping of all three displacement components is the default supported behavior.

If partial directional constraints are added later, they should still be represented as component-wise constrained DOF masks.

### 9.3 Load application

Point loads should be distributed to a small local neighborhood to avoid pathological single-node forcing on a coarse grid. Surface loads should be spread across the selected boundary voxels so that the total integrated force matches the user input.

Each load definition should include:

- location or selected region
- direction vector
- magnitude

### 9.4 Validation rules

The preprocessing layer should reject or warn on:

- missing supports
- zero-magnitude loads
- supports outside the solid
- loads outside the solid
- design spaces that do not connect load paths to supports
- conflicting selections that over-constrain the entire model

## 10. Numerical Limitations and Roadmap

### 10.1 Known v1 limitations

The first solver will have the following limitations:

- Staircase boundary approximation on Cartesian grids
- Stress concentration errors near sharp corners and poorly resolved interfaces
- Resolution sensitivity for thin members
- Grid-dependent load and support rasterization
- Linear elasticity only
- No geometric nonlinearity
- No material nonlinearity
- No buckling
- No fracture
- No dynamics
- No thermal or multiphysics coupling

These limitations are acceptable for a first generative-design engine focused on concept-stage topology changes rather than certification-grade structural simulation.

### 10.2 Roadmap after v1

Later phases can extend this design toward:

- one-way and then stronger thermo-structural coupling
- transient or dynamic structural analysis
- adaptive refinement near boundaries and high-gradient regions
- better preconditioners or multigrid
- CFD and broader multiphysics optimization on the same SDF lattice

## 11. Test and Validation Plan

The solver and optimization pipeline should be validated against a compact benchmark suite.

### 11.1 Structural solve benchmarks

- Cantilever beam under end load: verify displacement trend, stress hotspot location, and qualitative compliance behavior
- Symmetric support/load case: verify that the solution remains symmetric
- Resolution sweep: verify convergence trend as grid resolution increases

### 11.2 Backend consistency

- CPU vs CUDA comparison on the same problem
- CG residual-history comparison between backends
- tolerance-based validation of displacement and energy fields

### 11.3 Optimization workflow checks

- Non-design space remains unchanged across all iterations
- Material is removed only from the design region
- Target volume fraction is approached monotonically or with controlled overshoot
- Topology updates do not collapse immediately into disconnected dust under normal settings

### 11.4 Numerical sanity checks

- Poorly constrained models fail validation before solve
- Near-void designs remain numerically stable because of `rho_min` and `E_min`
- SDF reconstruction remains usable for preview and downstream meshing after repeated updates

## 12. Implementation Defaults for v1

To keep the first implementation decision-complete, the following defaults are recommended:

- PDE: static linear elasticity on the shared Cartesian grid
- Unknown placement: node-centered displacement
- Discretization: matrix-free structured-grid finite-difference elasticity stencil
- Solver: Conjugate Gradient
- Preconditioner: Jacobi
- Constraint handling: masked projection/elimination on constrained DOFs
- Visualization scalar: von Mises stress
- Removal scalar: strain energy density or compliance sensitivity proxy
- Optimization style: discrete material removal in design voxels with minimum-feature filtering
- Protected material: non-design space always retained

These defaults are the planned behavior for the first structural optimization release unless the implementation design is revised explicitly later.
