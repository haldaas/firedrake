import logging
import pytest
import numpy as np
from mpi4py import MPI
from firedrake import *
from firedrake.petsc import PETSc


def all_ranks(comm, value):
    """True only if ``value`` holds on every rank.

    Assertions in a parallel test must succeed or fail everywhere at once.  A
    rank that fails alone stops taking part in collective calls, and the others
    then hang in the next one instead of reporting the failure.
    """
    return comm.allreduce(bool(value), op=MPI.LAND)


def hpddm_params(nev=10, **kwargs):
    sp = {
        "mat_type": "aij",
        "ksp_type": "cg",
        "ksp_rtol": 1E-8,
        "ksp_max_it": 300,
        "pc_type": "python",
        "pc_python_type": "firedrake.HPDDMPC",
        "hpddm_pc_hpddm_levels_1_eps_nev": nev,
        "hpddm_pc_hpddm_levels_1_sub_pc_type": "cholesky",
        # A GenEO coarse operator can be singular, so it needs a factorisation
        # that tolerates that.  Plain 'lu' silently costs many iterations.
        "hpddm_pc_hpddm_coarse_pc_type": "cholesky",
    }
    sp.update(kwargs)
    return sp


def riesz_map(V):
    """The bilinear form whose Riesz map we precondition, and its right-hand side."""
    u, v = TrialFunction(V), TestFunction(V)
    element = V.ufl_element()
    if V.finat_element.formdegree == 0:
        a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    else:
        d = {HCurl: curl, HDiv: div}[element.sobolev_space]
        a = inner(u, v)*dx + inner(d(u), d(v))*dx
    L = inner(Constant(np.ones(V.value_shape)), v)*dx
    # Marker 1 is the left-hand edge alone.  Constraining one side only leaves
    # most subdomains floating, which is the case a coarse space is for.
    bcs = DirichletBC(V, zero(V.value_shape), 1)
    return a, L, bcs


def solve_with(V, sp):
    a, L, bcs = riesz_map(V)
    uh = Function(V)
    solver = LinearVariationalSolver(
        LinearVariationalProblem(a, L, uh, bcs=bcs), solver_parameters=sp)
    solver.solve()
    return uh, solver.snes.ksp


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("with_bcs", (False, True), ids=("nobcs", "bcs"))
def test_overlapping_subdomain_indices(with_bcs):
    """The subdomain index set is what PCHPDDMSetAuxiliaryMat requires.

    It must list every locally owned degree of freedom, must not repeat any,
    and must have as many entries as the auxiliary matrix has rows.
    """
    from firedrake.preconditioners.matis import local_neumann_matrix

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 2)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx
    bcs = [DirichletBC(V, 0, "on_boundary")] if with_bcs else []

    local = local_neumann_matrix(a, "aij", bcs=bcs, ignore_halo=False,
                                 scale_bc_diagonal=False)
    indices = local.rmap.getIndices()

    assert np.unique(indices).size == indices.size
    assert indices.min() >= 0 and indices.max() < V.dof_dset.layout_vec.getSize()
    assert indices.size == local.mat.getSize()[0]

    # PCHPDDM errors out if the index set misses a locally owned row
    rstart, rend = V.dof_dset.layout_vec.getOwnershipRange()
    assert np.setdiff1d(np.arange(rstart, rend), indices).size == 0

    # The subdomains must actually overlap, otherwise this is the BDDC
    # decomposition and there is nothing for a Schwarz method to exchange
    if mesh.comm.size > 1:
        assert indices.size > rend - rstart


@pytest.mark.parallel(3)
def test_overlapping_interior_rows():
    """The overlapping subdomain matrix agrees with the operator where it can.

    A degree of freedom whose supporting cells all lie inside the subdomain
    receives no contribution from outside it, so its row of the subdomain
    matrix must equal its row of the globally assembled operator.
    """
    from firedrake.preconditioners.matis import local_neumann_matrix, local_space

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 2)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx + \
        inner(TrialFunction(V), TestFunction(V))*dx

    local = local_neumann_matrix(a, "aij", ignore_halo=False)
    indices = local.rmap.getIndices()
    A = assemble(a).petscmat

    def support_counts(W):
        """Number of cells supporting each degree of freedom of W."""
        count = Function(W)
        shapes = (W.finat_element.space_dimension(), W.block_size)
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        par_loop((domain, instructions), dx, {"w": (count, INC)})
        return count

    Vsub = local_space(V, False, ignore_halo=False)
    outer = Function(Vsub).assign(support_counts(V)).dat.data_ro
    inner_counts = support_counts(Vsub).dat.data_ro
    interior = np.flatnonzero(np.isclose(outer, inner_counts))

    # MatGetRow serves only locally owned rows
    rstart, rend = A.getOwnershipRange()
    interior = interior[(indices[interior] >= rstart) & (indices[interior] < rend)]
    assert all_ranks(mesh.comm, interior.size > 0)

    matches = True
    for i in interior:
        lcols, lvals = local.mat.getRow(i)
        gcols, gvals = A.getRow(indices[i])
        local_row = {indices[c]: val for c, val in zip(lcols, lvals) if abs(val) > 1E-14}
        global_row = {c: val for c, val in zip(gcols, gvals) if abs(val) > 1E-14}
        matches &= set(local_row) == set(global_row)
        matches &= all(np.isclose(local_row[c], global_row.get(c, 0.0), atol=1E-12)
                       for c in local_row)
    assert all_ranks(mesh.comm, matches)


@pytest.mark.parallel(3)
@pytest.mark.parametrize("ignore_halo", (True, False), ids=("nohalo", "halo"))
def test_boundary_condition_on_part_of_the_boundary(ignore_halo):
    """A subdomain that misses the constrained boundary carries no condition.

    Constraining only part of the boundary leaves whole subdomains floating.
    Restricting the condition to the markers present on such a subdomain leaves
    none, and a DirichletBC on no markers raises rather than doing nothing.
    """
    from firedrake.preconditioners.matis import local_neumann_matrix, local_space

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx
    # Marker 1 is the left-hand edge alone
    bcs = [DirichletBC(V, 0, 1)]

    local = local_neumann_matrix(a, "aij", bcs=bcs, ignore_halo=ignore_halo,
                                 scale_bc_diagonal=False)
    assert local.mat.getSize()[0] == local.rmap.getIndices().size

    # Rows are constrained only where the subdomain does reach that edge
    marker = Function(V)
    bcs[0].set(marker, 1)
    Vsub = local_space(V, False, ignore_halo)
    constrained = np.flatnonzero(Function(Vsub).assign(marker).dat.data_ro > 0.5)
    diagonal = local.mat.getDiagonal().getArray()
    assert all_ranks(mesh.comm, np.allclose(diagonal[constrained], 1.0))
    # Somewhere in the world the edge does get constrained
    assert mesh.comm.allreduce(constrained.size, op=MPI.SUM) > 0


@pytest.mark.parallel(3)
@pytest.mark.parametrize("scale", (True, False), ids=("scaled", "unscaled"))
def test_bc_diagonal_scaling(scale):
    """PCHPDDM needs an unscaled Dirichlet diagonal, PCBDDC a scaled one.

    A MATIS sums its subdomain matrices, so the Dirichlet diagonal is divided
    by the number of subdomains holding it.  PCHPDDM never sums them, so the
    Neumann matrix must carry a plain one there.
    """
    from firedrake.preconditioners.matis import local_neumann_matrix, local_space

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 2)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx
    bcs = [DirichletBC(V, 0, "on_boundary")]

    local = local_neumann_matrix(a, "aij", bcs=bcs, ignore_halo=False,
                                 scale_bc_diagonal=scale)

    marker = Function(V)
    bcs[0].set(marker, 1)
    Vsub = local_space(V, False, ignore_halo=False)
    rows = np.flatnonzero(Function(Vsub).assign(marker).dat.data_ro > 0.5)
    diagonal = local.mat.getDiagonal().getArray()[rows]

    assert mesh.comm.allreduce(diagonal.size, op=MPI.SUM) > 0
    if scale:
        # Degrees of freedom shared with a neighbour are divided down
        smallest = diagonal.min() if diagonal.size else np.inf
        assert mesh.comm.allreduce(smallest, op=MPI.MIN) < 1.0
    else:
        assert all_ranks(mesh.comm, np.allclose(diagonal, 1.0))


@pytest.mark.parallel(3)
def test_hpddm_builds_coarse_level(tmp_path):
    """PCHPDDM must report the two levels we asked for, not silently skip them."""
    mesh = UnitSquareMesh(12, 12)
    V = FunctionSpace(mesh, "CG", 2)
    _, ksp = solve_with(V, hpddm_params())
    assert ksp.getConvergedReason() > 0

    pc = ksp.pc.getPythonContext().pc
    # tmp_path is a different directory on every rank, but the viewer is
    # collective and would deadlock on mismatched file names, so agree on one
    path = mesh.comm.bcast(str(tmp_path / "pcview.txt"), root=0)
    viewer = PETSc.Viewer().createASCII(path, comm=mesh.comm)
    pc.view(viewer)
    viewer.destroy()

    # Only the first rank writes the file, so it reads it back for everyone
    levels = None
    if mesh.comm.rank == 0:
        with open(path) as fh:
            levels = [line for line in fh if "levels:" in line]
    levels = mesh.comm.bcast(levels, root=0)
    assert levels, "PCHPDDM did not report a level count"
    assert int(levels[0].split(":")[1]) == 2

    grid, operator = pc.getHPDDMComplexities()
    assert np.isfinite(grid) and grid > 1.0
    assert np.isfinite(operator) and operator > 1.0


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family,degree", [("CG", 2), ("N1curl", 2), ("RT", 2)])
def test_hpddm_matches_direct_solve(family, degree):
    """The preconditioner must not change the answer, on H1, H(curl) and H(div).

    Unlike PCBDDC these spaces need no divergence matrix or discrete gradient.
    """
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, family, degree)
    uh, ksp = solve_with(V, hpddm_params())
    assert ksp.getConvergedReason() > 0

    ref, _ = solve_with(V, {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"})
    assert errornorm(ref, uh) / norm(ref) < 1E-7


@pytest.mark.parallel(3)
def test_hpddm_vector_block_size():
    """A space with a block size larger than one is decomposed correctly."""
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 2)
    assert V.block_size == 2
    uh, ksp = solve_with(V, hpddm_params())
    assert ksp.getConvergedReason() > 0

    ref, _ = solve_with(V, {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"})
    assert errornorm(ref, uh) / norm(ref) < 1E-7


@pytest.mark.parallel(3)
def test_hpddm_refinement_growth():
    """Iteration counts must grow slowly as the mesh is refined.

    The coarse space holds a fixed number of vectors per subdomain, so it does
    not track the problem exactly and the count is not expected to be flat.  It
    must stay close to constant, though, where a one-level method would grow
    with every refinement.
    """
    onelevel = {
        "mat_type": "aij", "ksp_type": "cg", "ksp_rtol": 1E-8, "ksp_max_it": 500,
        "pc_type": "asm", "pc_asm_type": "basic", "pc_asm_overlap": 1,
        "sub_pc_type": "cholesky",
    }
    meshes = MeshHierarchy(UnitSquareMesh(8, 8), 2)
    geneo_its, asm_its = [], []
    for mesh in meshes:
        V = FunctionSpace(mesh, "CG", 2)
        for params, its in ((hpddm_params(), geneo_its), (onelevel, asm_its)):
            _, ksp = solve_with(V, params)
            assert ksp.getConvergedReason() > 0
            its.append(ksp.getIterationNumber())

    # Measured over two refinements, each quadrupling the problem:
    # one-level 18, 22, 29 against 18, 20, 23 with the coarse space
    assert geneo_its[-1] <= asm_its[-1], f"{geneo_its} vs {asm_its}"
    growth = geneo_its[-1] - geneo_its[0]
    assert growth < asm_its[-1] - asm_its[0], f"{geneo_its} vs {asm_its}"


@pytest.mark.parallel(3)
def test_hpddm_reassembly():
    """Updating the operator must rebuild the coarse space, not reuse a stale one."""
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    kappa = Function(FunctionSpace(mesh, "DG", 0)).assign(1.0)
    a = inner(kappa*grad(u), grad(v))*dx + inner(u, v)*dx
    L = inner(Constant(1.0), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")

    uh = Function(V)
    solver = LinearVariationalSolver(
        LinearVariationalProblem(a, L, uh, bcs=bcs), solver_parameters=hpddm_params())
    solver.solve()
    assert solver.snes.ksp.getConvergedReason() > 0

    # Change the operator and solve again with the same solver
    kappa.assign(50.0)
    solver.solve()
    assert solver.snes.ksp.getConvergedReason() > 0
    reused = solver.snes.ksp.getIterationNumber()

    # Converging is not enough to show the coarse space was rebuilt, since CG
    # converges under any symmetric positive definite preconditioner.  A solver
    # built from scratch on the updated operator is the yardstick: a stale
    # coarse space would cost the reused one iterations against it.
    fresh_uh = Function(V)
    fresh = LinearVariationalSolver(
        LinearVariationalProblem(a, L, fresh_uh, bcs=bcs),
        solver_parameters=hpddm_params())
    fresh.solve()
    assert reused <= fresh.snes.ksp.getIterationNumber() + 1

    ref = Function(V)
    solve(a == L, ref, bcs=bcs,
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    assert errornorm(ref, uh) / norm(ref) < 1E-6


@pytest.mark.parametrize("mat_type", ("matfree", "is"))
def test_hpddm_rejects_unassembled_operator(mat_type):
    """PCHPDDM needs an assembled operator, and must say so rather than guess."""
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    a, L, bcs = riesz_map(V)

    sp = hpddm_params()
    sp["mat_type"] = mat_type
    sp["ksp_max_it"] = 5
    solver = LinearVariationalSolver(
        LinearVariationalProblem(a, L, Function(V), bcs=bcs), solver_parameters=sp)
    # PETSc wraps the error raised while setting up a Python PC
    with pytest.raises(PETSc.Error) as excinfo:
        solver.solve()
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_hpddm_rejects_extruded():
    """A submesh of an extruded mesh cannot be built, so say so."""
    mesh = ExtrudedMesh(UnitIntervalMesh(4), 4)
    V = FunctionSpace(mesh, "CG", 1)
    a, L, bcs = riesz_map(V)

    sp = hpddm_params()
    sp["ksp_max_it"] = 5
    solver = LinearVariationalSolver(
        LinearVariationalProblem(a, L, Function(V), bcs=bcs), solver_parameters=sp)
    with pytest.raises(PETSc.Error) as excinfo:
        solver.solve()
    assert isinstance(excinfo.value.__cause__, NotImplementedError)


@pytest.mark.parallel(3)
def test_hpddm_warns_without_overlap(caplog):
    """A mesh distributed without an overlap gives subdomains that do not overlap."""
    from firedrake.preconditioners.hpddm import validate_overlap

    mesh = UnitSquareMesh(
        8, 8, distribution_parameters={
            "overlap_type": (DistributedMeshOverlapType.NONE, 0)})
    # firedrake.logging.warning is a logger call, not warnings.warn
    with caplog.at_level(logging.WARNING, logger="firedrake"):
        validate_overlap(mesh)
    assert any("overlap" in record.getMessage() for record in caplog.records)
