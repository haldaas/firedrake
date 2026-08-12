"""Subdomain matrices and ``Mat`` of type ``is`` for domain decomposition.

A ``MATIS`` stores an operator in globally unassembled form: one sequential
matrix per subdomain together with a local-to-global map.  Firedrake builds the
subdomain matrix by assembling the form on a serial submesh of the cells owned
by each MPI process, so the default decomposition has exactly one subdomain per
process.

Two refinements are supported, selected by the arguments of
:func:`local_neumann_matrix`:

``cellwise``
    every cell is its own subdomain.  The local space is broken across all
    cells, so the subdomain matrix is block diagonal with one block per cell
    and the local-to-global map repeats global indices;

``subdomains``
    a partition of the cells owned by the process, giving a chosen number of
    subdomains per process.  :func:`partition_cells` builds one for a target
    subdomain size in cells.

Both are the same construction with a different numbering.  The subdomain
matrix is always assembled cell-broken and then *collapsed* onto the requested
subdomains: the cell-broken matrix is wrapped in a ``COMM_SELF`` ``MATIS``
whose local-to-global map sends each cell-broken degree of freedom to its
position in a numbering that lists the subdomains one after another.  With
repeated entries disallowed PETSc sums the duplicates, and converting the
result to ``aij`` yields a matrix whose degrees of freedom are grouped by
subdomain.

Grouping by subdomain is what ``PCBDDC`` requires.  It reads the subdomains
from the variable block sizes of the subdomain matrix as contiguous ranges of
the local numbering, not from the local-to-global map, so the map alone cannot
express more than one subdomain per process.

The obvious alternative, building one submesh per subdomain and assembling on
each, was rejected.  Firedrake has no disjoint-union mesh, so subdomains that
share an interface cannot be assembled together, and ``n`` submeshes cost ``n``
global kernels, function spaces and sparsities.  The collapse needs one
assembly and an index map.
"""

from collections import namedtuple
from itertools import repeat
from functools import cached_property

import numpy

from finat.ufl import BrokenElement
from ufl import Form, replace

from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import Submesh
from firedrake.petsc import PETSc, DEFAULT_PARTITIONER
from firedrake.preconditioners.fdm import broken_function
from pyop2.mpi import COMM_SELF

__all__ = ("partition_cells", "local_subdomains", "local_neumann_matrix", "create_matis")


def local_mesh(mesh, ignore_halo=True):
    """Return a serial submesh of the cells visible to this process.

    :arg mesh: the distributed mesh.
    :kwarg ignore_halo: if ``True`` the submesh holds only the cells owned by
        this process, giving non-overlapping subdomains.  If ``False`` the halo
        is retained, giving the overlapping subdomains wanted by additive
        Schwarz methods.
    :returns: the submesh, or ``None`` if ``mesh`` is already serial.
    """
    key = ("local_submesh", ignore_halo)
    cache = mesh._shared_data_cache["local_submesh_cache"]
    try:
        return cache[key]
    except KeyError:
        if mesh.comm.size > 1:
            submesh = Submesh(mesh, ignore_halo=ignore_halo, comm=COMM_SELF)
        else:
            submesh = None
        return cache.setdefault(key, submesh)


def local_space(V, cellwise, ignore_halo=True):
    """Return ``V`` reconstructed on the serial submesh, optionally broken."""
    mesh = local_mesh(V.mesh().unique(), ignore_halo)
    element = BrokenElement(V.ufl_element()) if cellwise else None
    return V.reconstruct(mesh=mesh, element=element)


def partition_cells(mesh, target_size):
    """Partition the cells of a serial mesh into connected subdomains.

    :arg mesh: a serial mesh, typically the one returned by :func:`local_mesh`.
    :arg target_size: the target number of cells per subdomain.  The number of
        subdomains is ``round(ncells / target_size)``, at least one and at most
        the number of cells.
    :returns: a DG(0) :class:`~.Function` holding the subdomain id of each cell.

    The partition is computed by PETSc's graph partitioner acting on the dual
    graph of the mesh, so the subdomain size is a target rather than a
    guarantee.  Partitioners routinely return disconnected parts, and a
    disconnected subdomain has a singular Neumann problem, so each part is
    split into its connected components.  The number of subdomains can
    therefore exceed ``round(ncells / target_size)``.
    """
    if mesh.comm.size > 1:
        raise ValueError("partition_cells expects a serial mesh, use local_mesh first")
    if target_size < 1:
        raise ValueError(f"Subdomain size must be at least 1, not {target_size}")

    Q = FunctionSpace(mesh, "DG", 0)
    subdomains = Function(Q, dtype=PETSc.IntType, name="cell_subdomains")

    ncells = mesh.cell_set.size
    nsub = min(max(1, round(ncells / target_size)), ncells)
    if nsub <= 1:
        # A single subdomain holding every cell
        return subdomains
    if nsub == ncells:
        # One cell per subdomain: cellwise, and no partitioner can improve on it
        subdomains.dat.data_wo[...] = numpy.arange(ncells, dtype=PETSc.IntType)
        return subdomains

    if DEFAULT_PARTITIONER == "simple":
        raise ValueError(
            "Splitting a process into several subdomains needs a graph partitioner, "
            "but PETSc was built without any of parmetis, ptscotch or chaco. "
            "Reconfigure PETSc, or leave the subdomain size unset to get one "
            "subdomain per process."
        )

    xadj, adjncy = dual_graph(mesh)
    ids = apply_partitioner(xadj, adjncy, nsub)
    ids, nsub = split_disconnected(xadj, adjncy, ids, nsub)
    subdomains.dat.data_wo[...] = ids
    return subdomains


def local_subdomains(mesh, target_size, ignore_halo=True):
    """Partition the cells visible to each process into subdomains.

    A convenience wrapper composing :func:`local_mesh` and
    :func:`partition_cells`, so that callers need not know whether the mesh is
    distributed.

    :arg mesh: the mesh, distributed or not.
    :arg target_size: the target number of cells per subdomain.
    :returns: a DG(0) :class:`~.Function` on the serial submesh.
    """
    submesh = local_mesh(mesh, ignore_halo)
    if submesh is None:
        submesh = mesh
    return partition_cells(submesh, target_size)


def dual_graph(mesh):
    """Return the CSR cell adjacency graph of a serial mesh."""
    ncells = mesh.cell_set.size
    facet_cell = mesh.interior_facets.facet_cell.reshape(-1, 2)
    # Keep only facets between two cells of the mesh proper
    facet_cell = facet_cell[(facet_cell < ncells).all(axis=1)]

    src = numpy.concatenate([facet_cell[:, 0], facet_cell[:, 1]])
    dst = numpy.concatenate([facet_cell[:, 1], facet_cell[:, 0]])
    order = numpy.argsort(src, kind="stable")

    xadj = numpy.zeros(ncells + 1, dtype=PETSc.IntType)
    numpy.cumsum(numpy.bincount(src, minlength=ncells), out=xadj[1:])
    return xadj, dst[order].astype(PETSc.IntType)


def apply_partitioner(xadj, adjncy, nsub):
    """Split a CSR graph into ``nsub`` parts, returning the part id of each vertex."""
    nvtx = xadj.size - 1
    values = numpy.ones(adjncy.size, dtype=PETSc.ScalarType)
    adj = PETSc.Mat().createAIJWithArrays((nvtx, nvtx), (xadj, adjncy, values),
                                          comm=COMM_SELF)
    adj.assemble()

    # PetscPartitionerPartition and MatPartitioningSetNParts have no petsc4py
    # bindings, so the part count goes through the options database.
    prefix = "firedrake_local_partition_"
    opts = PETSc.Options(prefix)
    opts["mat_partitioning_nparts"] = nsub
    try:
        part = PETSc.MatPartitioning().create(comm=COMM_SELF)
        part.setOptionsPrefix(prefix)
        part.setAdjacency(adj)
        part.setType(DEFAULT_PARTITIONER)
        part.setFromOptions()
        iset = PETSc.IS()
        part.apply(iset)
        ids = iset.getIndices().astype(PETSc.IntType)
        iset.destroy()
        part.destroy()
    finally:
        del opts["mat_partitioning_nparts"]
        adj.destroy()
    return ids


def split_disconnected(xadj, adjncy, ids, nsub):
    """Split every disconnected subdomain into its connected components.

    Graph partitioners routinely return disconnected parts, and a disconnected
    subdomain has a singular Neumann problem.  Each component is promoted to a
    subdomain of its own and empty subdomains are dropped, so the number of
    subdomains returned may differ from the number requested.

    :returns: ``(ids, nsub)``, renumbered contiguously from zero.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    nvtx = ids.size
    graph = csr_matrix((numpy.ones(adjncy.size, dtype=numpy.int8), adjncy, xadj),
                       shape=(nvtx, nvtx))
    out = numpy.empty_like(ids)
    nout = 0
    for k in range(nsub):
        cells = numpy.flatnonzero(ids == k)
        if cells.size == 0:
            continue
        ncomp, labels = connected_components(graph[cells][:, cells], directed=False)
        out[cells] = nout + labels
        nout += ncomp
    return out, nout


def subdomain_numbering(V, subdomains, ignore_halo=True):
    """Relate the cell-broken degrees of freedom of ``V`` to a subdomain numbering.

    :arg V: the global function space.
    :arg subdomains: a DG(0) :class:`~.Function` on the serial submesh holding
        the subdomain id of each cell, as returned by :func:`partition_cells`.
    :returns: ``(collapse, size, indices, subdomain_sizes)``.  ``collapse``
        sends each cell-broken degree of freedom to its position in a numbering
        that lists the subdomains one after another, ``size`` is the length of
        that numbering, ``indices`` sends it to the global numbering of ``V``,
        and ``subdomain_sizes`` counts the degrees of freedom of each subdomain.

    The subdomains occupy contiguous ranges of the new numbering, which is what
    ``PCBDDC`` needs in order to read them back from the variable block sizes.
    """
    ids = subdomains.dat.data_ro.astype(PETSc.IntType)
    nsub = int(ids.max()) + 1 if ids.size else 0

    Vsub = local_space(V, False, ignore_halo)
    Wsub = local_space(V, True, ignore_halo)
    vmap = Vsub.cell_node_map().values
    wmap = Wsub.cell_node_map().values
    bs = Vsub.block_size

    # The broken node map is a bijection, and every broken node belongs to
    # exactly one cell and hence to exactly one subdomain.
    broken_node = numpy.empty(Wsub.node_count, dtype=PETSc.IntType)
    broken_node[wmap] = vmap
    broken_subdomain = numpy.empty(Wsub.node_count, dtype=PETSc.IntType)
    broken_subdomain[wmap] = ids[:, None]

    collapse = numpy.empty(Wsub.node_count, dtype=PETSc.IntType)
    subdomain_sizes = numpy.empty(nsub, dtype=PETSc.IntType)
    nodes = []
    offset = 0
    for k in range(nsub):
        nodes_k = numpy.unique(vmap[ids == k])
        here = broken_subdomain == k
        collapse[here] = offset + numpy.searchsorted(nodes_k, broken_node[here])
        nodes.append(nodes_k)
        offset += nodes_k.size
        subdomain_sizes[k] = nodes_k.size * bs
    nodes = numpy.concatenate(nodes) if nsub else numpy.empty(0, dtype=PETSc.IntType)

    # Expand nodes into degrees of freedom
    components = numpy.arange(bs, dtype=PETSc.IntType)
    collapse = (collapse[:, None] * bs + components).ravel()

    u = Function(V)
    shp = u.dat.data_ro.shape
    u.dat.data_wo[...] = numpy.arange(*V.dof_dset.layout_vec.getOwnershipRange()).reshape(shp)
    usub = Function(Vsub).assign(u)
    indices = usub.dat.data_ro.astype(PETSc.IntType).reshape(-1, bs)[nodes].ravel()
    return collapse, offset * bs, indices, subdomain_sizes


def restriction_matrix(collapse, ncols):
    """Return the boolean matrix sending cell-broken degrees of freedom to subdomain ones.

    Row ``i`` holds a single one in column ``collapse[i]``, so that
    ``R.T @ A @ R`` sums the cell-broken contributions of each subdomain
    degree of freedom.
    """
    nrows = collapse.size
    indptr = numpy.arange(nrows + 1, dtype=PETSc.IntType)
    values = numpy.ones(nrows, dtype=PETSc.ScalarType)
    R = PETSc.Mat().createAIJWithArrays((nrows, ncols), (indptr, collapse, values),
                                        comm=COMM_SELF)
    R.assemble()
    return R


def gather(vec, indices, comm):
    """Read the entries of a distributed ``Vec`` at the given global indices."""
    iset = PETSc.IS().createGeneral(indices, comm=comm)
    out = PETSc.Vec().createSeq(indices.size, comm=COMM_SELF)
    scatter = PETSc.Scatter().create(vec, iset, out, None)
    scatter.scatter(vec, out, addv=PETSc.InsertMode.INSERT_VALUES,
                    mode=PETSc.ScatterMode.FORWARD)
    values = out.getArray().copy()
    scatter.destroy()
    out.destroy()
    iset.destroy()
    return values


def dof_multiplicity(V, indices, comm):
    """Count how many subdomains hold each degree of freedom of the subdomain matrix.

    :arg indices: the local-to-global map of the subdomain matrix, which lists
        a global degree of freedom once for every subdomain holding it.
    :returns: the multiplicity of each entry of ``indices``.
    """
    vec = V.dof_dset.layout_vec.duplicate()
    vec.zeroEntries()
    vec.setValues(indices, numpy.ones(indices.size, dtype=PETSc.ScalarType),
                  addv=PETSc.InsertMode.ADD_VALUES)
    vec.assemble()
    counts = gather(vec, indices, comm)
    vec.destroy()
    return counts


def bc_diagonal_scaling(mat, V, bcs, rindices, cindices, comm):
    """Return a callable putting the right value on the Dirichlet diagonal.

    The assembler writes a one on every copy of a Dirichlet degree of freedom,
    once per subdomain holding it, so the globally assembled operator would
    carry the number of copies on its diagonal rather than a one.  PETSc
    divides by the multiplicity when it applies boundary conditions to a
    ``MATIS`` itself, in ``MatISZeroRowsColumnsLocal``; the subdomain matrix
    here is assembled outside PETSc, so the same scaling is applied by hand.

    The returned callable must be run after every reassembly, which overwrites
    the diagonal again.
    """
    def noop():
        pass

    if not bcs or mat.getType() == "python":
        return noop
    if rindices.size != cindices.size or not numpy.array_equal(rindices, cindices):
        # A rectangular operator has no diagonal to correct
        return noop

    marker = Function(V)
    for bc in bcs:
        bc.set(marker, 1)
    with marker.dat.vec_ro as mvec:
        marked = gather(mvec, rindices, comm)

    rows = numpy.flatnonzero(marked > 0.5).astype(PETSc.IntType)
    if rows.size == 0:
        return noop
    values = 1.0 / dof_multiplicity(V, rindices, comm)[rows]

    def rescale():
        diagonal = mat.getDiagonal()
        array = diagonal.getArray()
        array[rows] = values
        mat.setDiagonal(diagonal)
        diagonal.destroy()
    return rescale


class BrokenDirichletBC(DirichletBC):
    """A Dirichlet condition transferred onto the cell-broken space."""

    def __init__(self, bc):
        self.bc = bc
        V = bc.function_space().broken_space()
        g = bc._original_arg
        super().__init__(V, g, bc.sub_domain)

    @cached_property
    def nodes(self):
        u = Function(self.bc.function_space())
        self.bc.set(u, 1)
        u = broken_function(u.function_space(), val=u.dat)
        return numpy.flatnonzero(u.dat.data)


LocalNeumannMatrix = namedtuple(
    "LocalNeumannMatrix",
    ("mat", "rmap", "cmap", "subdomain_sizes", "comm", "sizes", "update"))
"""The subdomain matrix of a form, and the maps relating it to the global problem.

``mat``
    the sequential subdomain matrix, with degrees of freedom grouped by subdomain;
``rmap``, ``cmap``
    maps from its degrees of freedom to the global numbering of the test and
    trial spaces, repeating global indices when there is more than one subdomain;
``subdomain_sizes``
    the number of degrees of freedom of each subdomain, or ``None`` when there
    is a single subdomain per process;
``comm``, ``sizes``
    the communicator and global sizes of the operator;
``update``
    reassembles ``mat`` in place.
"""


def local_neumann_matrix(a, local_mat_type, cellwise=False, bcs=(),
                         ignore_halo=True, subdomains=None):
    """Assemble the subdomain (Neumann) matrix of a form on the serial submesh.

    :arg a: a :class:`~ufl.Form`, or a ``python`` :class:`PETSc.Mat` wrapping one.
    :arg local_mat_type: the ``Mat`` type of the subdomain matrix.
    :arg cellwise: break the local space across cells, making every cell a
        subdomain.
    :arg bcs: Dirichlet conditions, ignored if ``a`` is a ``Mat``, which
        carries its own.
    :kwarg ignore_halo: whether the submesh excludes the halo.  ``True`` gives
        the non-overlapping subdomains wanted by ``PCBDDC``; ``False`` gives
        overlapping ones.
    :kwarg subdomains: a DG(0) :class:`~.Function` partitioning the cells of the
        submesh, giving several subdomains per process.  Mutually exclusive
        with ``cellwise``.
    :returns: a :class:`LocalNeumannMatrix`.
    """
    from firedrake.assemble import get_assembler

    if subdomains is not None:
        if cellwise:
            raise ValueError("Pass either cellwise or subdomains, not both")
        if local_mat_type == "matfree":
            raise NotImplementedError(
                "Several subdomains per process need an assembled subdomain matrix, "
                "which a matrix-free local matrix cannot provide")

    # Several subdomains per process are built by summing the cell-broken
    # matrix onto subdomain degrees of freedom, so the local space is broken
    # in both cases.
    broken = cellwise or subdomains is not None

    def local_argument(arg, broken):
        return arg.reconstruct(function_space=local_space(arg.function_space(), broken, ignore_halo))

    def local_integral(it):
        extra_domain_integral_type_map = dict(it.extra_domain_integral_type_map())
        extra_domain_integral_type_map[it.ufl_domain()] = it.integral_type()
        return it.reconstruct(domain=local_mesh(it.ufl_domain(), ignore_halo),
                              extra_domain_integral_type_map=extra_domain_integral_type_map)

    def local_bc(bc, broken):
        V = bc.function_space()
        Vsub = local_space(V, False, ignore_halo)
        sub_domain = list(bc.sub_domain)
        if "on_boundary" in sub_domain:
            sub_domain.remove("on_boundary")
            sub_domain.extend(V.mesh().unique().exterior_facets.unique_markers)

        valid_markers = Vsub.mesh().unique().exterior_facets.unique_markers
        sub_domain = list(set(sub_domain) & set(valid_markers))
        bc = bc.reconstruct(V=Vsub, g=0, sub_domain=sub_domain)
        if broken:
            bc = BrokenDirichletBC(bc)
        return bc

    def local_to_global_indices(V, cellwise):
        u = Function(V)
        shp = u.dat.data_ro.shape
        u.dat.data_wo[...] = numpy.arange(*V.dof_dset.layout_vec.getOwnershipRange()).reshape(shp)

        Vsub = local_space(V, False, ignore_halo)
        usub = Function(Vsub).assign(u)
        if cellwise:
            usub = broken_function(usub.function_space(), val=usub.dat)
        return usub.dat.data_ro.astype(PETSc.IntType)

    if isinstance(a, Form):
        form = a
        args = a.arguments()
        comm = args[0].function_space().comm
        sizes = tuple(arg.function_space().dof_dset.layout_vec.getSizes() for arg in args)
    elif isinstance(a, PETSc.Mat):
        assert a.type == "python"
        ctx = a.getPythonContext()
        form = ctx.a
        bcs = ctx.bcs
        comm = a.comm
        sizes = a.getSizes()

    local_form = replace(form, {arg: local_argument(arg, broken) for arg in form.arguments()})
    local_form = Form(list(map(local_integral, local_form.integrals())))
    local_bcs = tuple(map(local_bc, bcs, repeat(broken)))

    assembler = get_assembler(local_form, bcs=local_bcs, mat_type=local_mat_type)
    tensor = assembler.assemble()

    Vrow, Vcol = (arg.function_space() for arg in form.arguments())
    if subdomains is None:
        rindices = local_to_global_indices(Vrow, cellwise)
        cindices = local_to_global_indices(Vcol, cellwise)
        subdomain_sizes = None
        mat = tensor.petscmat

        def collapse():
            pass
    else:
        rcollapse, nrows, rindices, subdomain_sizes = subdomain_numbering(Vrow, subdomains, ignore_halo)
        ccollapse, ncols, cindices, _ = subdomain_numbering(Vcol, subdomains, ignore_halo)

        Rrow = restriction_matrix(rcollapse, nrows)
        square = nrows == ncols and numpy.array_equal(rcollapse, ccollapse)
        Rcol = Rrow if square else restriction_matrix(ccollapse, ncols)

        def triple_product(result=None):
            # Reuses the symbolic phase of the product when result is given, so
            # the cell-broken matrix must keep the same sparsity across updates.
            if square:
                return tensor.petscmat.ptap(Rrow, result)
            return Rrow.transposeMatMult(tensor.petscmat.matMult(Rcol), result)

        mat = triple_product()

        def collapse():
            triple_product(mat)

    rmap = PETSc.LGMap().create(rindices, comm=comm)
    cmap = PETSc.LGMap().create(cindices, comm=comm)
    rescale = bc_diagonal_scaling(mat, Vrow, bcs, rindices, cindices, comm)
    rescale()

    def update():
        assembler.assemble(tensor=tensor)
        collapse()
        rescale()

    return LocalNeumannMatrix(mat, rmap, cmap, subdomain_sizes, comm, sizes, update)


def create_matis(a, local_mat_type, cellwise=False, bcs=(), subdomains=None):
    """Assemble a form as a ``Mat`` of type ``is``.

    :returns: ``(Amatis, update)``, where calling ``update`` reassembles it.
    """
    local = local_neumann_matrix(a, local_mat_type, cellwise=cellwise, bcs=bcs,
                                 subdomains=subdomains)

    Amatis = PETSc.Mat().createIS(local.sizes, comm=local.comm)
    Amatis.setISAllowRepeated(cellwise or subdomains is not None)
    Amatis.setLGMap(local.rmap, local.cmap)
    Amatis.setISLocalMat(local.mat)
    Amatis.setUp()
    Amatis.assemble()
    if local.subdomain_sizes is not None:
        # PCBDDC reads the subdomains back from these as contiguous ranges
        Amatis.getISLocalMat().setVariableBlockSizes(local.subdomain_sizes)

    def update():
        local.update()
        Amatis.assemble()
    return Amatis, update
