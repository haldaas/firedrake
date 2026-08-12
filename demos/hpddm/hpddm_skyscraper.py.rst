A robust coarse space for highly heterogeneous diffusion
========================================================

This demo solves the *skyscraper* problem, a diffusion problem whose
coefficient jumps by six orders of magnitude between neighbouring regions:

.. math::

   -\nabla\cdot(\kappa\nabla u) &= 1 \quad \textrm{in}\ \Omega = (0,1)^2

   u &= 0 \quad \textrm{on}\ \Gamma_D

   \kappa\nabla u\cdot\vec{n} &= 0 \quad \textrm{on}\ \partial\Omega\setminus\Gamma_D

where :math:`\Gamma_D` is the left-hand edge :math:`x = 0` alone, the rest of the
boundary being left natural. Constraining only part of the boundary is the more
demanding case for a domain decomposition method: most subdomains then touch no
Dirichlet boundary at all, so their local Neumann problems are singular and the
coarse space has to supply the missing information.

The coefficient :math:`\kappa` is piecewise constant: a checkerboard of tall
"skyscrapers" where :math:`\kappa\sim10^5`, crossed by three slanted channels
where :math:`\kappa\sim10^6`, on a background where :math:`\kappa=1`. It is the
benchmark of :cite:`AlDaas2019`, and the coefficient used here is a
transcription of the one distributed with that paper.

The problem is chosen because it defeats a one-level domain decomposition
method. In a one-level overlapping Schwarz method the subdomains only talk to
their neighbours, so information crosses the domain one subdomain at a time and
the condition number grows both as the mesh is refined and as the coefficient
jumps grow. Adding a *coarse space* -- a small global problem solved on every
iteration -- fixes the first problem, but a coarse space built from geometry
alone, such as a coarse mesh, does not fix the second: it knows nothing about
where :math:`\kappa` jumps.

GenEO builds the coarse space from the operator instead. On each subdomain it
solves a generalised eigenvalue problem between the subdomain matrix and the
subdomain's *Neumann* matrix, and keeps the eigenvectors that the one-level
method handles worst. Those are exactly the modes the jumps in :math:`\kappa`
create, so the resulting method is robust in the coefficient. Firedrake exposes
this through :class:`~.HPDDMPC`, which wraps PETSc's ``PCHPDDM``.

This demo is intended to be run in parallel, since the subdomains are the
partition of the mesh across MPI processes::

  from firedrake import *
  import numpy as np

The subdomains of an overlapping Schwarz method are the cells owned by each
process *together with their halo*, so the width of the overlap is the width of
the mesh halo. That is fixed when the mesh is built, not by the preconditioner,
so we ask for it explicitly rather than relying on the default::

  mesh = UnitSquareMesh(
      64, 64,
      distribution_parameters={"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})

Now the coefficient. ``channel`` marks a slanted band running from
:math:`(x_1,y_1)` to :math:`(x_2,y_2)`, of the given width, and ``skyscraper``
combines a checkerboard of high-contrast blocks with three such channels. Both
are written for NumPy arrays so that we can evaluate them at every cell at
once::

  def channel(a, b, x1, y1, x2, y2, width):
      slope = (y2 - y1) / (x2 - x1)
      lower = slope * (a - x2) + y2
      return (a >= x1) & (a <= x2) & (b >= lower) & (b <= lower + width)


  def skyscraper(a, b):
      da = np.floor(9 * a)
      db = np.floor(9 * b)
      kappa = np.ones_like(a)

      # Three slanted channels of very high conductivity
      kappa = np.where(channel(a, b, 0.3, 0.6, 0.9, 0.5, 0.2), (a + b) * 1e6, kappa)
      kappa = np.where(channel(a, b, 0.5, 0.15, 0.9, 0.05, 0.2), a * 1e5, kappa)
      kappa = np.where(channel(a, b, 0.1, 0.2, 0.5, 0.6, 0.15), b * 1e6, kappa)

      # The skyscrapers themselves take precedence over the channels
      blocks = (da % 2 == 0) & (db % 2 == 0)
      return np.where(blocks, 1e5 * (da + db + 1), kappa)

The coefficient is constant on each cell, so it lives in a piecewise constant
space. We evaluate it at the cell centroids, which are exactly the points that
carry the degrees of freedom of that space::

  W = FunctionSpace(mesh, "DG", 0)
  centroids = Function(VectorFunctionSpace(mesh, "DG", 0))
  centroids.interpolate(SpatialCoordinate(mesh))

  kappa = Function(W, name="kappa")
  kappa.dat.data[:] = skyscraper(centroids.dat.data_ro[:, 0],
                                 centroids.dat.data_ro[:, 1])

The variational problem itself is an ordinary weighted Poisson problem. The
boundary condition is applied on marker ``1``, which for a
:func:`~.UnitSquareMesh` is the left-hand edge; the zero-flux condition on the
other three sides is natural and so needs nothing here::

  V = FunctionSpace(mesh, "CG", 2)
  u = TrialFunction(V)
  v = TestFunction(V)
  a = inner(kappa * grad(u), grad(v)) * dx
  L = inner(Constant(1.0), v) * dx
  bcs = DirichletBC(V, zero(), 1)

The operator is symmetric positive definite, so we use conjugate gradients
throughout::

  base = {
      "mat_type": "aij",
      "ksp_type": "cg",
      "ksp_rtol": 1e-8,
      "ksp_max_it": 500,
  }

Every matrix here is symmetric positive definite, and so is the coarse problem
built from them, so all the direct solves are Cholesky factorisations rather
than LU. That halves their cost, and on the coarse problem it matters for more
than cost: asking for ``lu`` there was measured to cost several times as many
iterations on this problem, converging to the same answer but far more slowly.

For a baseline we take classical one-level additive Schwarz, PETSc's
``PCASM``::

  onelevel = dict(base, pc_type="asm", pc_asm_type="basic", pc_asm_overlap=1,
                  sub_pc_type="cholesky")

And now the GenEO method. Unlike a ``PCBDDC`` preconditioner,
:class:`~.HPDDMPC` preconditions an ordinary assembled operator, so ``mat_type``
stays ``"aij"``::

  twolevel = dict(
      base,
      pc_type="python",
      pc_python_type="firedrake.HPDDMPC",
      # Keep every subdomain eigenvector whose eigenvalue is below this, so
      # that each subdomain contributes as much to the coarse space as it
      # needs to rather than a number fixed in advance.  Without this, or
      # 'eps_nev', there is no coarse level at all
      hpddm_pc_hpddm_levels_1_eps_threshold_absolute=0.1,
      hpddm_pc_hpddm_levels_1_sub_pc_type="cholesky",
      # Solve the eigenproblem with the same factorisation as the subdomain
      # solve, rather than building a second one
      hpddm_pc_hpddm_levels_1_st_share_sub_ksp=True,
      # The coarse problem is small, so solve it directly.  Cholesky rather
      # than LU matters here, see below
      hpddm_pc_hpddm_coarse_pc_type="cholesky",
  )

Sharing is worth a word. Each subdomain would otherwise be factorised twice:
once for SLEPc's spectral transformation while computing the coarse space, and
once for the subdomain solve applied on every Krylov iteration.
``st_share_sub_ksp`` makes the two use one factorisation. It is available
because :class:`~.HPDDMPC` tells ``PCHPDDM`` that the auxiliary matrix it
supplies is the local Neumann matrix, which is what allows the two operators to
share a sparsity pattern. PETSc declines to share silently when it cannot, so
it is worth checking rather than assuming::

  def geneo_pc(solver):
      return solver.snes.ksp.pc.getPythonContext().pc

Note that :class:`~.HPDDMPC` chooses the *additive* Schwarz method and the
*balanced* coarse correction by default, rather than PETSc's restricted and
deflated ones. Those two are not symmetric, and conjugate gradients breaks down
on a preconditioner that is not. If you would rather use a method such as GMRES,
which converges in fewer iterations here, override
``hpddm_pc_hpddm_levels_1_pc_asm_type`` and ``hpddm_pc_hpddm_coarse_correction``
to get them back.

We solve the same problem with each::

  uh = Function(V, name="u")
  problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

  iterations = {}
  solvers = {}
  for name, parameters in (("one-level", onelevel), ("two-level", twolevel)):
      uh.assign(0)
      solver = LinearVariationalSolver(problem, solver_parameters=parameters)
      solver.solve()
      iterations[name] = solver.snes.ksp.getIterationNumber()
      solvers[name] = solver
      PETSc.Sys.Print(f"{name}: {iterations[name]} iterations")

The coarse space earns its keep. The one-level method costs more iterations the
more subdomains there are, because information still has to cross the domain one
subdomain at a time, while the two-level method stays nearly flat:

=========== =========== ===========
 Processes   One-level   Two-level
=========== =========== ===========
  2           47          10
  4           79          15
  8           123         16
=========== =========== ===========

That near-flatness is what makes it worth solving an eigenproblem on every
subdomain.

``PCHPDDM`` will report what the coarse space costs, as the ratio of unknowns
(``grid``) and of nonzeros (``operator``) across all levels to those on the
finest level alone. Both are close to one here, so the coarse space is cheap.
We also confirm that the factorisation really was shared::

  pc = geneo_pc(solvers["two-level"])
  grid, operator = pc.getHPDDMComplexities()
  PETSc.Sys.Print(f"grid complexity {grid:.3f}, operator complexity {operator:.3f}")
  PETSc.Sys.Print(f"shared subdomain KSP: {pc.getHPDDMSTShareSubKSP()}")

To see the geometry the coarse space has to cope with, write the coefficient out
alongside the solution with ``VTKFile("skyscraper.pvd").write(uh, kappa)`` and
open the result in ParaView: the solution flows along the high-conductivity
channels and around the blocks, which is exactly the structure that no purely
geometric coarse space would know about.

A python script version of this demo can be found :demo:`here
<hpddm_skyscraper.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
