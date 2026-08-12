import sys
import pytest
import numpy as np
from functools import reduce
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=123456789))


def bddc_params(mat_type="is", cellwise=False, subdomain_size=None, adaptive=False,
                use_divergence=None, use_gradient=None, corner_selection=None, debug=0):
    chol = {
        "pc_type": "cholesky",
        "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
    }
    sp = {
        "mat_type": mat_type,
        "pc_type": "python",
        "pc_python_type": "firedrake.BDDCPC",
        "bddc_cellwise": cellwise,
        "bddc_pc_bddc_neumann": chol,
        "bddc_pc_bddc_dirichlet": chol,
        "bddc_pc_bddc_coarse": chol,
        "bddc_debug": debug,
    }
    if subdomain_size is not None:
        # several subdomains per process, of roughly this many cells each
        sp["bddc_subdomain_size"] = subdomain_size
    if use_gradient is not None:
        # defaults to True for 3D H(curl) spaces
        sp["bddc_use_discrete_gradient"] = use_gradient
    if use_divergence is not None:
        # defaults to True for 2D H(curl) and 2D/3D H(div) spaces
        sp["bddc_use_divergence_mat"] = use_divergence
    if corner_selection is not None:
        # defaults to True for H1 spaces
        sp["bddc_pc_bddc_corner_selection"] = corner_selection

    if adaptive:
        sp.update({
            "bddc_pc_bddc_use_deluxe_scaling": None,
            "bddc_pc_bddc_adaptive_userdefined": None,
            "bddc_pc_bddc_deluxe_zerorows": False,
            "bddc_pc_bddc_adaptive_threshold": 5,
        })
    # On MacOSX the distributed right-hand side is bugged!
    if DEFAULT_DIRECT_SOLVER == "mumps" and sys.platform == "darwin":
        sp.update({"bddc_pc_bddc_coarse_mat_mumps_icntl_20": 0})

    return sp


def solver_parameters(cellwise=False, subdomain_size=None, condense=False, variant=None,
                      rtol=1E-10, atol=0, **kwargs):
    # Several subdomains per process are only built from a 'matfree' P
    unassembled = cellwise or subdomain_size is not None
    mat_type = "matfree" if unassembled and variant != "fdm" else "is"
    sp_bddc = bddc_params(mat_type=mat_type, cellwise=cellwise,
                          subdomain_size=subdomain_size, **kwargs)
    if variant != "fdm":
        assert not condense
        sp = sp_bddc

    elif condense:
        sp = {
            "pc_type": "python",
            "pc_python_type": "firedrake.FacetSplitPC",
            "facet_pc_type": "python",
            "facet_pc_python_type": "firedrake.FDMPC",
            "facet_fdm_static_condensation": True,
            "facet_fdm_pc_use_amat": False,
            "facet_fdm_mat_type": "is",
            "facet_fdm_mat_is_allow_repeated": cellwise,
            "facet_fdm_pc_type": "fieldsplit",
            "facet_fdm_pc_fieldsplit_type": "symmetric_multiplicative",
            "facet_fdm_pc_fieldsplit_diag_use_amat": False,
            "facet_fdm_pc_fieldsplit_off_diag_use_amat": False,
            "facet_fdm_fieldsplit_ksp_type": "preonly",
            "facet_fdm_fieldsplit_0_pc_type": "bjacobi",
            "facet_fdm_fieldsplit_0_pc_type_sub_pc_type": "icc",
            "facet_fdm_fieldsplit_1": sp_bddc,
        }
    else:
        sp = {
            "pc_type": "python",
            "pc_python_type": "firedrake.FDMPC",
            "fdm_pc_use_amat": False,
            "fdm_mat_is_allow_repeated": cellwise,
            "fdm": sp_bddc,
        }

    sp.update({
        "ksp_type": "cg",
        "ksp_max_it": 20,
        "ksp_norm_type": "natural",
        "ksp_converged_reason": None,
        "ksp_rtol": rtol,
        "ksp_atol": atol,
    })
    if variant == "fdm":
        sp["mat_type"] = "matfree"
    return sp


def solve_riesz_map(rg, mesh, family, degree, variant, bcs, cellwise=False, subdomain_size=None, condense=False, vector=False, threshold=None, elasticity=False):
    """Solve the riesz map for a random manufactured solution and return the
       square root of the estimated condition number."""
    dirichlet_ids = []
    if bcs:
        dirichlet_ids = ["on_boundary"]
        if hasattr(mesh, "extruded") and mesh.extruded:
            dirichlet_ids.extend(["bottom", "top"])

    tdim = mesh.topological_dimension
    if family.endswith("E"):
        family = "RTCE" if tdim == 2 else "NCE"
    if family.endswith("F"):
        family = "RTCF" if tdim == 2 else "NCF"

    fs = VectorFunctionSpace if vector else FunctionSpace

    V = fs(mesh, family, degree, variant=variant)
    v = TestFunction(V)
    u = TrialFunction(V)
    d = {
        H1: grad,
        HCurl: curl,
        HDiv: div,
    }[V.ufl_element().sobolev_space]
    formdegree = V.finat_element.formdegree

    if elasticity:
        gamma = Constant(1E4)
        a = (inner(grad(u) + grad(u).T, grad(v)) * dx
             + inner(div(u) * gamma, div(v)) * dx)
    elif formdegree == 0:
        a = inner(d(u), d(v)) * dx
    else:
        a = (inner(u, v) + inner(d(u), d(v))) * dx

    u_exact = rg.uniform(V, -1, 1)
    L = replace(a, {u: u_exact})
    bcs = [DirichletBC(V, u_exact, sub) for sub in dirichlet_ids]

    # Near nullspace
    nsp = None
    adaptive = False
    use_divergence = None
    if elasticity:
        adaptive = True
        use_divergence = True  # use divergence mat trick to compute no-net flux coarse space
    elif formdegree == 0:
        b = np.zeros(V.value_shape)
        expr = Constant(b)
        basis = []
        for i in np.ndindex(V.value_shape):
            b[...] = 0
            b[i] = 1
            expr.assign(b)
            basis.append(Function(V).interpolate(expr))
        nsp = VectorSpaceBasis(basis)
        nsp.orthonormalize()

    appctx = {}
    if threshold is not None:
        appctx["primal_markers"] = get_primal_markers(mesh, threshold=threshold)

    uh = Function(V, name="solution")
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    rtol = 1E-8
    sp = solver_parameters(cellwise=cellwise, subdomain_size=subdomain_size, condense=condense,
                           variant=variant, rtol=rtol,
                           use_divergence=use_divergence, adaptive=adaptive)
    sp.setdefault("ksp_view_singularvalues", None)
    solver = LinearVariationalSolver(problem, near_nullspace=nsp,
                                     solver_parameters=sp, appctx=appctx)
    solver.solve()
    uerr = Function(V).assign(uh - u_exact)
    assert (assemble(a(uerr, uerr)) / assemble(a(u_exact, u_exact))) ** 0.5 < rtol

    ew = solver.snes.ksp.computeEigenvalues().real
    kappa = 1.0
    if len(ew):
        assert np.isclose(min(ew), 1.0, rtol=1.e-2)
        kappa = max(abs(ew)) / min(abs(ew))
    return kappa ** 0.5


def tensor_mesh(x, extruded=False, **kwargs):
    base = TensorRectangleMesh(x, x, quadrilateral=True, **kwargs)
    if extruded:
        mesh = ExtrudedMesh(base, len(x)-1, layer_height=np.diff(x))
    else:
        mesh = base
    return mesh


def corner_refined_mesh(nx, ratio=0.5, extruded=False, **kwargs):
    t = 1-np.logspace(-nx, -1, nx, base=1/ratio)
    t /= t[0]
    x = np.concatenate([-t, [0], np.flip(t)])
    return tensor_mesh(x, extruded=extruded, **kwargs)


def cell_aspect_ratio(mesh):
    """Compute the aspect ratio of each cell"""
    J = Jacobian(mesh)
    G = J.T * J
    hs = tuple(abs(G[i, i]**0.5) for i in range(G.ufl_shape[0]))
    hmax = reduce(max_value, hs)
    hmin = reduce(min_value, hs)

    DG0 = FunctionSpace(mesh, "DG", 0)
    ratio = Function(DG0).interpolate(hmax / hmin)
    return ratio


def get_primal_markers(mesh, threshold=2**15):
    """Cell marker for cells with high aspect ratio"""
    threshold = Constant(threshold)
    marker = cell_aspect_ratio(mesh)
    marker.interpolate(conditional(ge(marker, threshold), 1, 0))
    return marker


@pytest.fixture(params=(2, 3), ids=("square", "cube"))
def mh(request):
    dim = request.param
    nx = 4
    base = UnitSquareMesh(nx, nx, quadrilateral=True)
    mh = MeshHierarchy(base, 1)
    if dim == 3:
        mh = ExtrudedMeshHierarchy(mh, height=1, base_layer=nx)
    return mh


@pytest.mark.parallel
@pytest.mark.parametrize("degree", range(1, 3))
@pytest.mark.parametrize("variant", ("spectral", "fdm"))
def test_vertex_dofs(mh, variant, degree):
    """Check that we extract the right number of vertex dofs from a high order Lagrange space."""
    from firedrake.preconditioners.bddc import get_restricted_dofs
    mesh = mh[-1]
    P1 = FunctionSpace(mesh, "Lagrange", 1, variant=variant)
    V0 = FunctionSpace(mesh, "Lagrange", degree, variant=variant)
    v = get_restricted_dofs(V0, "vertex")
    assert v.getSizes() == P1.dof_dset.layout_vec.getSizes()


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family,degree", [("Q", 4), ("E", 3), ("F", 3)])
@pytest.mark.parametrize("condense", (False, True))
def test_bddc_cellwise_fdm(rg, mh, family, degree, condense):
    """Test h-independence of condition number by measuring iteration counts"""
    variant = "fdm"
    bcs = True
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=True, condense=condense) for m in mh]
    assert (np.diff(sqrt_kappa) <= 0.1).all(), str(sqrt_kappa)


@pytest.mark.skipcomplex  # max_value does not work in complex mode
@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family,degree", [("Q", 4)])
def test_bddc_cellwise_high_aspect_ratio(rg, family, degree):
    """Test that marking high aspect ratio cells leads to robust iteration counts"""
    variant = "fdm"
    bcs = True
    mh = [corner_refined_mesh(nx) for nx in (10, 12)]
    # For these meshes it is better to set adaptive BDDC parameters,
    # but here we just test the appctx["primal_markers"] interface
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=True, threshold=2**6) for m in mh]
    assert (np.diff(sqrt_kappa) <= 0.1).all(), str(sqrt_kappa)


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree", [("Q", 4)])
@pytest.mark.parametrize("vector", (False, True), ids=("scalar", "vector"))
def test_bddc_aij_quad(rg, mh, family, degree, vector):
    """Test h-dependence of condition number by measuring iteration counts"""
    variant = None
    bcs = True
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, vector=vector) for m in mh]
    assert (np.diff(sqrt_kappa) <= 0.5).all(), str(sqrt_kappa)


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree,cellwise", [("CG", 3, False), ("CG", 3, True), ("N1curl", 3, False), ("N1div", 3, False)])
def test_bddc_aij_simplex(rg, family, degree, cellwise):
    """Test h-dependence of condition number by measuring iteration counts"""
    variant = None
    bcs = True
    base = UnitCubeMesh(2, 2, 2)
    meshes = MeshHierarchy(base, 2)
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=cellwise) for m in meshes]
    assert (np.diff(sqrt_kappa) <= 0.5).all(), str(sqrt_kappa)


@pytest.mark.skipcomplex(
    reason="Adaptive BDDC's sub-Schur factorization assumes SPD matrices, unsupported for complex Hermitian systems"
)
@pytest.mark.parallel(3)
@pytest.mark.parametrize("family,degree,cellwise", [("CG", 2, False), ("GN", 1, False), ("MTW", 1, False)])
def test_bddc_elasticity_aij_simplex(rg, family, degree, cellwise):
    """Test h-dependence of condition number by measuring iteration counts"""
    base = UnitSquareMesh(2, 2)
    meshes = MeshHierarchy(base, 2)
    dim = base.topological_dimension
    vector = (family == "CG")
    variant = "alfeld" if family == "CG" and degree < 2*dim else None
    bcs = True
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=cellwise, vector=vector, elasticity=True) for m in meshes]
    assert (np.diff(sqrt_kappa) <= 1.0).all(), str(sqrt_kappa)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("cellwise", (True, False))
@pytest.mark.parametrize("local_mat_type", ("aij", "matfree"))
@pytest.mark.parametrize("with_bcs", (False, True), ids=("nobcs", "bcs"))
def test_create_matis(local_mat_type, cellwise, with_bcs):
    from firedrake.preconditioners.bddc import create_matis
    if with_bcs and local_mat_type == "matfree":
        pytest.skip("A matrix-free subdomain matrix cannot scale its Dirichlet diagonal")
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx
    bcs = [DirichletBC(V, 0, "on_boundary")] if with_bcs else []
    A = assemble(a, bcs=bcs, mat_type="matfree").petscmat

    A, assembler = create_matis(A, local_mat_type, cellwise=cellwise)
    B = assemble(a, bcs=bcs, mat_type=local_mat_type).petscmat
    if local_mat_type == "matfree":
        Ax, x = A.createVecs()
        Bx, _ = B.createVecs()
        x.setRandom()
        A.mult(x, Ax)
        B.mult(x, Bx)
        assert np.allclose(Ax.array, Bx.array)
    else:
        A.convert("aij")
        B.axpy(-1, A)
        assert np.isclose(B.norm(PETSc.NormType.FROBENIUS), 0)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("target_size", (1, 4, 10**6))
@pytest.mark.parametrize("with_bcs", (False, True), ids=("nobcs", "bcs"))
def test_matis_subdomain_size(target_size, with_bcs):
    """The assembled operator must not depend on how we decompose it.

    For any subdomains, the MatIS is a partial assembly of the same cell-broken
    matrix.  A full assembly of the MatIS must therefore give the operator.
    """
    from firedrake.preconditioners.matis import create_matis, local_subdomains
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    bcs = [DirichletBC(V, 0, "on_boundary")] if with_bcs else []
    ref = assemble(a, bcs=bcs).petscmat

    subdomains = local_subdomains(mesh, target_size)
    A, update = create_matis(a, "aij", bcs=bcs, subdomains=subdomains)
    assert A.getISAllowRepeated()
    for _ in range(2):
        D = A.convert("aij", PETSc.Mat())
        D.axpy(-1, ref)
        assert np.isclose(D.norm(PETSc.NormType.FROBENIUS), 0, atol=1E-12)
        # Reassembly must land on the same matrix
        update()


@pytest.mark.parallel([1, 3])
def test_partition_cells():
    """Each subdomain must be non-empty and connected, and the ids have no gaps."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from firedrake.preconditioners.matis import dual_graph, local_mesh, partition_cells

    mesh = UnitSquareMesh(6, 6)
    lmesh = local_mesh(mesh, ignore_halo=True)
    if lmesh is None:
        lmesh = mesh
    ncells = lmesh.cell_set.size
    xadj, adjncy = dual_graph(lmesh)
    graph = csr_matrix((np.ones(adjncy.size), adjncy, xadj), shape=(ncells, ncells))

    for target in (1, 3, ncells, 10**6):
        ids = partition_cells(lmesh, target).dat.data_ro.astype(int)
        assert ids.size == ncells
        nsub = ids.max() + 1
        assert np.array_equal(np.unique(ids), np.arange(nsub))
        assert 1 <= nsub <= ncells
        if target == 1:
            assert nsub == ncells
        if target >= ncells:
            assert nsub == 1
        for k in range(nsub):
            cells = np.flatnonzero(ids == k)
            assert connected_components(graph[cells][:, cells], directed=False)[0] == 1


@pytest.mark.parallel([1, 3])
def test_matis_subdomain_blocks():
    """The subdomain sizes must add up to the size of the subdomain matrix."""
    from firedrake.preconditioners.matis import local_neumann_matrix, local_subdomains
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 2)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx

    subdomains = local_subdomains(mesh, 4)
    nsub = int(subdomains.dat.data_ro.max()) + 1
    local = local_neumann_matrix(a, "aij", subdomains=subdomains)

    assert local.subdomain_sizes.size == nsub
    assert (local.subdomain_sizes > 0).all()
    assert local.subdomain_sizes.sum() == local.mat.getSize()[0]
    assert local.rmap.getSize() == local.mat.getSize()[0]


def test_bddc_subdomain_size_view(tmp_path):
    """PCBDDC must use the subdomains that we declare, and not choose its own."""
    from firedrake.preconditioners.matis import local_subdomains
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(1.0), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")

    sp = bddc_params(mat_type="matfree", subdomain_size=8)
    sp.update({"ksp_type": "cg", "ksp_rtol": 1E-9, "ksp_max_it": 50})

    problem = LinearVariationalProblem(a, L, Function(V), bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    assert solver.snes.ksp.getConvergedReason() > 0

    path = str(tmp_path / "pcview.txt")
    viewer = PETSc.Viewer().createASCII(path, comm=mesh.comm)
    solver.snes.ksp.pc.view(viewer)
    viewer.destroy()
    with open(path) as fh:
        totals = [line for line in fh if "Total subdomains" in line]
    assert totals, "PCBDDC did not report a subdomain count"
    total = int(totals[0].split(":")[1])

    expected = int(local_subdomains(mesh, 8).dat.data_ro.max()) + 1
    assert total == expected > 1


@pytest.mark.parallel([1, 3])
def test_bddc_subdomain_size_convergence(rg):
    """Measure the effect of h on the condition number, through the iteration count.

    The subdomain size is a fixed number of cells.  Refinement thus makes the
    subdomains smaller together with the mesh, and the condition number must
    become constant.  The first level of the hierarchy is 8x8, because a
    coarser level is not yet in the asymptotic regime.
    """
    base = UnitSquareMesh(8, 8)
    meshes = MeshHierarchy(base, 2)
    sqrt_kappa = [solve_riesz_map(rg, m, "CG", 2, None, True, subdomain_size=8)
                  for m in meshes]
    assert (np.diff(sqrt_kappa) <= 0.5).all(), str(sqrt_kappa)


def test_matis_subdomain_errors():
    from firedrake.preconditioners.matis import (
        local_neumann_matrix, local_subdomains, partition_cells)
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx
    subdomains = local_subdomains(mesh, 4)

    with pytest.raises(ValueError):
        local_neumann_matrix(a, "aij", cellwise=True, subdomains=subdomains)
    with pytest.raises(NotImplementedError):
        local_neumann_matrix(a, "matfree", subdomains=subdomains)
    with pytest.raises(ValueError):
        partition_cells(mesh, 0)


@pytest.mark.parametrize("extra,error", [
    ({"bddc_cellwise": True}, ValueError),
    ({"bddc_matfree": True}, NotImplementedError),
    ({"mat_type": "is"}, ValueError),
])
def test_bddc_subdomain_size_rejected(extra, error):
    """An unsupported combination must raise, and not give a wrong answer."""
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(1.0), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")

    sp = bddc_params(mat_type="matfree", subdomain_size=4)
    sp.update({"ksp_type": "cg", "ksp_max_it": 10})
    sp.update(extra)

    problem = LinearVariationalProblem(a, L, Function(V), bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=sp)
    # PETSc wraps the error raised while setting up a Python PC
    with pytest.raises(PETSc.Error) as excinfo:
        solver.solve()
    assert isinstance(excinfo.value.__cause__, error)
