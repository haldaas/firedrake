"""Subdomain matrices for domain decomposition methods.

A ``Mat`` of type ``is`` holds an operator in unassembled form.  It stores one
sequential subdomain matrix per MPI process, and a map from the degrees of
freedom of that matrix to the global numbering.  The operator is the sum of the
subdomain matrices.

This module assembles the subdomain matrix.  It builds a serial submesh of the
cells that the process holds, and assembles the form on that submesh.  The
default decomposition therefore has one subdomain per process.

:func:`local_neumann_matrix` builds two other decompositions:

``cellwise=True``
    Each cell is a subdomain.  The local space is broken across the cells, so
    the subdomain matrix is block diagonal with one block per cell, and the
    local-to-global map repeats global indices.

``subdomains=<DG(0) Function>``
    The cells of the process are divided into several subdomains.  Use
    :func:`partition_cells` to compute such a division for a target subdomain
    size in cells.

Both decompositions assemble the same cell-broken matrix ``A``.  To get several
subdomains per process, the module then computes the product ``R.T @ A @ R``.
The boolean matrix ``R`` comes from :func:`restriction_matrix`.  For each
subdomain, it adds together the cell-broken copies of each degree of freedom of
that subdomain.  The product is a matrix whose degrees of freedom are grouped by
subdomain: each subdomain occupies a contiguous range of rows.

``PCBDDC`` needs this grouping.  It reads the subdomains from the variable
block sizes of the subdomain matrix, as contiguous ranges of the local
numbering.  It does not read them from the local-to-global map, and that map
alone cannot express more than one subdomain per process.

One submesh per subdomain is an alternative to the product above, but it is not
used here.  Firedrake has no disjoint-union mesh, so it cannot assemble two
subdomains that share an interface together.  Also, ``n`` submeshes cost ``n``
global kernels, ``n`` function spaces and ``n`` sparsities.  The product needs
one assembly and one index map.

The subdomains do not overlap, because the submesh holds only the cells that
the process owns.  Pass ``ignore_halo=False`` to keep the halo cells and get
the overlapping subdomains that ``PCHPDDM`` needs.
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
    """Return a serial submesh of the cells that this process holds.

    Parameters
    ----------
    mesh : MeshGeometry
        The distributed mesh.
    ignore_halo : bool
        If ``True``, the submesh holds only the cells that the process owns.
        The subdomains then do not overlap.  If ``False``, the submesh also
        holds the halo cells.  The subdomains then overlap, as additive
        Schwarz methods need.

    Returns
    -------
    MeshGeometry or None
        The submesh, or ``None`` if ``mesh`` is already serial.

    Notes
    -----
    The submesh is cached on ``mesh``, because it is costly to build and
    several function spaces reconstruct themselves on it.
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
    """Return ``V`` reconstructed on the serial submesh.

    Parameters
    ----------
    V : FunctionSpace
        The global function space.
    cellwise : bool
        If ``True``, break the element, so that the degrees of freedom of each
        cell are independent of those of its neighbours.
    ignore_halo : bool
        Whether the submesh excludes the halo, as in :func:`local_mesh`.

    Returns
    -------
    FunctionSpace
        The space on the submesh.  If ``V`` is already serial, the submesh is
        ``None`` and ``V`` keeps its own mesh.
    """
    mesh = local_mesh(V.mesh().unique(), ignore_halo)
    element = BrokenElement(V.ufl_element()) if cellwise else None
    return V.reconstruct(mesh=mesh, element=element)


def partition_cells(mesh, target_size):
    """Divide the cells of a serial mesh into connected subdomains.

    Parameters
    ----------
    mesh : MeshGeometry
        A serial mesh, usually the one that :func:`local_mesh` returns.
    target_size : int
        The wanted number of cells per subdomain.  The number of subdomains is
        ``round(ncells / target_size)``, but not less than one and not more
        than the number of cells.

    Returns
    -------
    Function
        A DG(0) :class:`~.Function` that holds the subdomain id of each cell.

    Raises
    ------
    ValueError
        If ``mesh`` is not serial, or if ``target_size`` is less than one, or
        if PETSc has no graph partitioner but one is necessary.

    Notes
    -----
    A PETSc graph partitioner divides the dual graph of the mesh, so the
    subdomain size is a target and not a guarantee.  A partitioner can also
    return a part that is not connected, and a subdomain that is not connected
    has a singular Neumann problem.  Each such part is thus divided into its
    connected components.  The number of subdomains can therefore be more than
    ``round(ncells / target_size)``.
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
        # One subdomain that holds every cell
        return subdomains
    if nsub == ncells:
        # One cell per subdomain.  This is the cellwise case, and no
        # partitioner can improve on it.
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
    """Divide the cells that each process holds into subdomains.

    This function calls :func:`local_mesh` and then :func:`partition_cells`.
    The caller thus does not have to know whether the mesh is distributed.

    Parameters
    ----------
    mesh : MeshGeometry
        The mesh, distributed or not.
    target_size : int
        The wanted number of cells per subdomain.
    ignore_halo : bool
        Whether the submesh excludes the halo, as in :func:`local_mesh`.

    Returns
    -------
    Function
        A DG(0) :class:`~.Function` on the serial submesh.
    """
    submesh = local_mesh(mesh, ignore_halo)
    if submesh is None:
        submesh = mesh
    return partition_cells(submesh, target_size)


def dual_graph(mesh):
    """Return the dual graph of a serial mesh in CSR format.

    The vertices of the dual graph are the cells of the mesh.  Two vertices
    are adjacent if the two cells have a common facet.

    Parameters
    ----------
    mesh : MeshGeometry
        A serial mesh.

    Returns
    -------
    xadj : numpy.ndarray
        The start of the adjacency list of each cell, with ``ncells + 1``
        entries.
    adjncy : numpy.ndarray
        The cells that are adjacent to each cell, one list after another.
    """
    ncells = mesh.cell_set.size
    facet_cell = mesh.interior_facets.facet_cell.reshape(-1, 2)
    # Keep only the facets between two cells of the mesh itself
    facet_cell = facet_cell[(facet_cell < ncells).all(axis=1)]

    src = numpy.concatenate([facet_cell[:, 0], facet_cell[:, 1]])
    dst = numpy.concatenate([facet_cell[:, 1], facet_cell[:, 0]])
    order = numpy.argsort(src, kind="stable")

    xadj = numpy.zeros(ncells + 1, dtype=PETSc.IntType)
    numpy.cumsum(numpy.bincount(src, minlength=ncells), out=xadj[1:])
    return xadj, dst[order].astype(PETSc.IntType)


def apply_partitioner(xadj, adjncy, nsub):
    """Divide a CSR graph into ``nsub`` parts.

    Parameters
    ----------
    xadj, adjncy : numpy.ndarray
        The graph in CSR format, as :func:`dual_graph` returns it.
    nsub : int
        The number of parts.

    Returns
    -------
    numpy.ndarray
        The part id of each vertex.  A part can be empty, and a part can also
        be disconnected.
    """
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
    """Divide each disconnected subdomain into its connected components.

    A graph partitioner can return a part that is not connected, and a
    subdomain that is not connected has a singular Neumann problem.  Each
    component thus becomes a subdomain of its own.  Empty subdomains are
    removed.  The number of subdomains that this function returns can
    therefore differ from ``nsub``.

    Parameters
    ----------
    xadj, adjncy : numpy.ndarray
        The graph in CSR format, as :func:`dual_graph` returns it.
    ids : numpy.ndarray
        The subdomain id of each vertex.
    nsub : int
        The number of subdomains in ``ids``.

    Returns
    -------
    ids : numpy.ndarray
        The new subdomain id of each vertex, numbered without gaps from zero.
    nsub : int
        The new number of subdomains.
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

    The subdomain numbering lists the degrees of freedom of the first
    subdomain, then those of the second subdomain, and so on.  A degree of
    freedom on an interface occurs once for each subdomain that holds it.

    Parameters
    ----------
    V : FunctionSpace
        The global function space.
    subdomains : Function
        A DG(0) :class:`~.Function` on the serial submesh that holds the
        subdomain id of each cell, as :func:`partition_cells` returns it.
    ignore_halo : bool
        Whether the submesh excludes the halo, as in :func:`local_mesh`.

    Returns
    -------
    collapse : numpy.ndarray
        The position of each cell-broken degree of freedom in the subdomain
        numbering.
    size : int
        The number of degrees of freedom in the subdomain numbering.
    indices : numpy.ndarray
        The global number of each degree of freedom of the subdomain
        numbering.
    subdomain_sizes : numpy.ndarray
        The number of degrees of freedom of each subdomain.

    Notes
    -----
    Each subdomain occupies a contiguous range of the subdomain numbering.
    ``PCBDDC`` needs this, because it reads the subdomains back from the
    variable block sizes of the subdomain matrix.
    """
    ids = subdomains.dat.data_ro.astype(PETSc.IntType)
    nsub = int(ids.max()) + 1 if ids.size else 0

    Vsub = local_space(V, False, ignore_halo)
    Wsub = local_space(V, True, ignore_halo)
    vmap = Vsub.cell_node_map().values
    wmap = Wsub.cell_node_map().values
    bs = Vsub.block_size

    # The broken node map is a bijection.  Each broken node belongs to one
    # cell, and therefore to one subdomain.
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
    """Return the boolean matrix that maps cell-broken to subdomain degrees of freedom.

    Row ``i`` holds one entry, a one in column ``collapse[i]``.  The product
    ``R.T @ A @ R`` thus adds together the cell-broken contributions to each
    subdomain degree of freedom.

    Parameters
    ----------
    collapse : numpy.ndarray
        The position of each cell-broken degree of freedom in the subdomain
        numbering, as :func:`subdomain_numbering` returns it.
    ncols : int
        The number of degrees of freedom in the subdomain numbering.

    Returns
    -------
    PETSc.Mat
        A sequential ``aij`` matrix with one entry per row.
    """
    nrows = collapse.size
    indptr = numpy.arange(nrows + 1, dtype=PETSc.IntType)
    values = numpy.ones(nrows, dtype=PETSc.ScalarType)
    R = PETSc.Mat().createAIJWithArrays((nrows, ncols), (indptr, collapse, values),
                                        comm=COMM_SELF)
    R.assemble()
    return R


def gather(vec, indices, comm):
    """Read the entries of a distributed ``Vec`` at the given global indices.

    Parameters
    ----------
    vec : PETSc.Vec
        The distributed vector.
    indices : numpy.ndarray
        The global indices to read.  They can repeat, and they can point to
        entries that another process owns.
    comm : mpi4py.MPI.Comm
        The communicator of ``vec``.

    Returns
    -------
    numpy.ndarray
        The value at each entry of ``indices``.
    """
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
    """Count the subdomains that hold each degree of freedom of the subdomain matrix.

    Parameters
    ----------
    V : FunctionSpace
        The global function space.
    indices : numpy.ndarray
        The local-to-global map of the subdomain matrix.  It lists a global
        degree of freedom one time for each subdomain that holds it.
    comm : mpi4py.MPI.Comm
        The communicator of the operator.

    Returns
    -------
    numpy.ndarray
        The number of subdomains that hold each entry of ``indices``.  The
        count includes the subdomains of the other processes.
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
    """Return a callable that writes the correct Dirichlet diagonal.

    The assembler writes a one on each copy of a Dirichlet degree of freedom,
    one copy for each subdomain that holds it.  The assembled operator is the
    sum of the subdomain matrices, so its diagonal would hold the number of
    copies and not a one.  The callable divides each such entry by that
    number.  PETSc does the same in ``MatISZeroRowsColumnsLocal`` when it
    applies the boundary conditions itself.  Here the subdomain matrix is
    assembled outside PETSc, so this module must do it.

    Parameters
    ----------
    mat : PETSc.Mat
        The subdomain matrix.
    V : FunctionSpace
        The test function space.
    bcs : tuple
        The Dirichlet conditions of the operator.
    rindices, cindices : numpy.ndarray
        The row and column local-to-global maps of ``mat``.
    comm : mpi4py.MPI.Comm
        The communicator of the operator.

    Returns
    -------
    callable
        A function of no arguments that writes the diagonal.  Call it after
        each reassembly, because the assembler writes the ones again.  It does
        nothing if there are no conditions to apply, if ``mat`` is matrix-free,
        or if the operator is rectangular and thus has no diagonal.
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
    """A Dirichlet condition moved onto the cell-broken space.

    The broken space has one degree of freedom for each cell that touches a
    node.  This class constrains all of those copies.

    Parameters
    ----------
    bc : DirichletBC
        The condition on the unbroken space.
    """

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
"""The subdomain matrix of a form, and the maps that relate it to the operator.

Attributes
----------
mat : PETSc.Mat
    The sequential subdomain matrix.  Its degrees of freedom are grouped by
    subdomain.
rmap, cmap : PETSc.LGMap
    The maps from the rows and columns of ``mat`` to the global numbering of
    the test and trial spaces.  A global index repeats when the process holds
    more than one subdomain.
subdomain_sizes : numpy.ndarray or None
    The number of degrees of freedom of each subdomain, or ``None`` if the
    process holds one subdomain.
comm : mpi4py.MPI.Comm
    The communicator of the operator.
sizes : tuple
    The global sizes of the operator.
update : callable
    A function of no arguments that assembles ``mat`` again, in place.
"""


def local_neumann_matrix(a, local_mat_type, cellwise=False, bcs=(),
                         ignore_halo=True, subdomains=None,
                         scale_bc_diagonal=True):
    """Assemble the subdomain matrix of a form on the serial submesh.

    The subdomain matrix is a Neumann matrix: the form is integrated over the
    cells of the subdomain only, and nothing constrains the interface with the
    other subdomains.

    Parameters
    ----------
    a : ufl.Form or PETSc.Mat
        The bilinear form, or a ``python`` :class:`PETSc.Mat` that holds one.
    local_mat_type : str
        The ``Mat`` type of the subdomain matrix.
    cellwise : bool
        If ``True``, break the local space across the cells, so that each cell
        is a subdomain.
    bcs : tuple
        The Dirichlet conditions.  This argument is ignored if ``a`` is a
        ``Mat``, because the ``Mat`` holds its own conditions.
    ignore_halo : bool
        Whether the submesh excludes the halo.  ``True`` gives the subdomains
        without overlap that ``PCBDDC`` needs.  ``False`` gives the
        overlapping subdomains that ``PCHPDDM`` needs.
    subdomains : Function or None
        A DG(0) :class:`~.Function` that divides the cells of the submesh into
        several subdomains.  Do not use it together with ``cellwise``.
    scale_bc_diagonal : bool
        Whether to divide the Dirichlet diagonal by the number of subdomains
        that hold each degree of freedom, see :func:`bc_diagonal_scaling`.
        Use ``True`` if the subdomain matrices are added together to give the
        operator, as a ``Mat`` of type ``is`` does.  Use ``False`` for a
        method such as ``PCHPDDM``, which applies its own partition of unity
        and needs a one on that diagonal.

    Returns
    -------
    LocalNeumannMatrix
        The subdomain matrix, and the maps that relate it to the operator.

    Raises
    ------
    ValueError
        If both ``cellwise`` and ``subdomains`` are given.
    NotImplementedError
        If ``subdomains`` is given and ``local_mat_type`` is ``"matfree"``.
    """
    from firedrake.assemble import get_assembler

    if subdomains is not None:
        if cellwise:
            raise ValueError("Pass either cellwise or subdomains, not both")
        if local_mat_type == "matfree":
            raise NotImplementedError(
                "Several subdomains per process need an assembled subdomain matrix, "
                "which a matrix-free local matrix cannot provide")

    # Several subdomains per process come from the same cell-broken matrix,
    # added together onto the subdomain degrees of freedom.  The local space is
    # thus broken in both cases.
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
        if not sub_domain:
            # This subdomain does not touch the constrained boundary, so it
            # has no condition.  A DirichletBC with no markers is a different
            # thing: it has no nodes to concatenate, and it raises.
            return None
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
    local_bcs = tuple(filter(None, map(local_bc, bcs, repeat(broken))))

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
            # A given result reuses the symbolic phase of the product.  The
            # cell-broken matrix must therefore keep the same sparsity when it
            # is assembled again.
            if square:
                return tensor.petscmat.ptap(Rrow, result)
            return Rrow.transposeMatMult(tensor.petscmat.matMult(Rcol), result)

        mat = triple_product()

        def collapse():
            triple_product(mat)

    rmap = PETSc.LGMap().create(rindices, comm=comm)
    cmap = PETSc.LGMap().create(cindices, comm=comm)
    if scale_bc_diagonal:
        rescale = bc_diagonal_scaling(mat, Vrow, bcs, rindices, cindices, comm)
    else:
        def rescale():
            pass
    rescale()

    def update():
        assembler.assemble(tensor=tensor)
        collapse()
        rescale()

    return LocalNeumannMatrix(mat, rmap, cmap, subdomain_sizes, comm, sizes, update)


def create_matis(a, local_mat_type, cellwise=False, bcs=(), subdomains=None):
    """Assemble a form as a ``Mat`` of type ``is``.

    Parameters
    ----------
    a : ufl.Form or PETSc.Mat
        The bilinear form, or a ``python`` :class:`PETSc.Mat` that holds one.
    local_mat_type : str
        The ``Mat`` type of the subdomain matrix.
    cellwise : bool
        If ``True``, each cell is a subdomain.
    bcs : tuple
        The Dirichlet conditions.  This argument is ignored if ``a`` is a
        ``Mat``.
    subdomains : Function or None
        A DG(0) :class:`~.Function` that divides the cells of each process
        into several subdomains.  See :func:`local_neumann_matrix`.

    Returns
    -------
    Amatis : PETSc.Mat
        The operator as a ``Mat`` of type ``is``.
    update : callable
        A function of no arguments that assembles ``Amatis`` again, in place.
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
