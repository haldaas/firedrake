from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space
from firedrake.mesh import DistributedMeshOverlapType
from firedrake.preconditioners.matis import local_neumann_matrix
from firedrake.logging import warning
from pyop2.mpi import COMM_SELF
import numpy

__all__ = ("HPDDMPC",)


class HPDDMPC(PCBase):
    """PC for PETSc PCHPDDM, an overlapping Schwarz method with a GenEO coarse space.

    The subdomains are the cells owned by each MPI process together with their
    halo, so they overlap, and the width of the overlap is the width of the
    mesh halo.  It is chosen when the mesh is built, through the
    ``'overlap_type'`` entry of ``distribution_parameters``, and not by this
    preconditioner.  ``PCHPDDM`` takes exactly one subdomain per process, so
    there is no equivalent of ``'bddc_subdomain_size'``.

    The coarse space is built by solving a generalised eigenvalue problem on
    each subdomain against its Neumann matrix, which this PC assembles by
    reassembling the bilinear form on a serial submesh.  Unlike ``PCBDDC``,
    ``PCHPDDM`` needs no divergence matrix or discrete gradient for H(div) and
    H(curl) problems, and it preconditions an ordinary assembled operator, so
    ``mat_type`` should be ``'aij'``, ``'baij'`` or ``'sbaij'``.

    Internally, this PC creates a PETSc PCHPDDM object that can be controlled by
    the options:
    - ``'hpddm_pc_hpddm_levels_1_eps_nev'`` for the number of eigenvectors
    computed on each subdomain, which is the size of the coarse space per
    subdomain.  Without it, or a threshold, PCHPDDM builds no coarse level at
    all and degenerates to a one-level Schwarz method, so this PC defaults it to
    ``DEFAULT_EPS_NEV``,
    - ``'hpddm_pc_hpddm_levels_1_sub_pc_type'`` for the subdomain solver,
    - ``'hpddm_pc_hpddm_levels_1_st_pc_type'`` for the eigensolver's spectral
    transformation,
    - ``'hpddm_pc_hpddm_coarse_'`` to set the coarse solver KSP,
    - ``'hpddm_pc_hpddm_levels_1_st_share_sub_ksp'`` to solve the eigenproblem
    with the same factorisation as the subdomain solve, instead of building a
    second one,
    - ``'hpddm_pc_hpddm_levels_1_pc_asm_type'`` for the one-level method, which
    this PC defaults to ``'basic'`` rather than PETSc's ``'restrict'``,
    - ``'hpddm_pc_hpddm_coarse_correction'`` for the coarse correction, which
    this PC defaults to ``'balanced'`` rather than PETSc's ``'deflated'``.

    The last two defaults are chosen so that the preconditioner is symmetric and
    can be used under CG: the restricted Schwarz method and the deflated coarse
    correction are both non-symmetric, and CG breaks down on them.  Override
    both to use the PETSc defaults under a method such as GMRES, which
    converges in fewer iterations.

    For a symmetric positive definite operator the coarse problem is too, so
    solve it with ``'cholesky'`` rather than ``'lu'``.  Beyond costing twice the
    work, ``'lu'`` was measured to need several times as many iterations on the
    problems tested here, converging to the same answer but more slowly; PETSc's
    default coarse solver does not show this.

    Not supported on extruded meshes, because the subdomain matrix is assembled
    on a submesh and a submesh of an extruded mesh cannot be built.
    """

    _prefix = "hpddm_"

    DEFAULT_EPS_NEV = 10
    """Eigenvectors per subdomain, used when the user sets no ``eps_nev``."""

    def initialize(self, pc):
        prefix = (pc.getOptionsPrefix() or "") + self._prefix

        dm = pc.getDM()
        V = get_function_space(dm)
        mesh = V.mesh().unique()

        # Create new PC object as HPDDM type.  The type must be set before
        # setHPDDMAuxiliaryMat, which is dispatched through PetscTryMethod and
        # would silently do nothing on a PC of another type.
        hpddmpc = PETSc.PC().create(comm=pc.comm)
        hpddmpc.incrementTabLevel(1, parent=pc)
        hpddmpc.setOptionsPrefix(prefix)
        hpddmpc.setType(PETSc.PC.Type.HPDDM)

        A, P = pc.getOperators()
        if not P.type.endswith("aij"):
            raise ValueError(
                f"PCHPDDM needs an assembled preconditioning matrix, not '{P.type}'. "
                "Set 'mat_type' to 'aij', 'baij' or 'sbaij'.")
        if mesh.extruded:
            raise NotImplementedError(
                "HPDDMPC does not work on extruded meshes, because the subdomain "
                "matrix is assembled on a submesh, which cannot be built from an "
                "extruded mesh.")
        validate_overlap(mesh)

        # The subdomain matrix is the Neumann matrix of the overlapping
        # subdomain, so the halo is kept.  Its Dirichlet diagonal must not be
        # scaled by the dof multiplicity: PCHPDDM does not sum the subdomain
        # matrices to form the operator, it applies its own partition of unity.
        a, bcs = self.form(pc)
        local = local_neumann_matrix(a, "aij", bcs=bcs, ignore_halo=False,
                                     scale_bc_diagonal=False)

        # PCHPDDM wants the overlapping subdomain as an IS in the global
        # numbering of P, which is the local-to-global map of the subdomain
        # matrix.  The submesh numbers its degrees of freedom in its own order,
        # so the map is unsorted, and PCASM sorts the IS it is given without
        # telling PCHPDDM to permute an auxiliary matrix that is the Neumann
        # matrix.  Sort it here and permute the matrix to match, so that the
        # two agree whatever PCHPDDM does with them.
        indices = local.rmap.getIndices()
        self.perm = PETSc.IS().createGeneral(
            numpy.argsort(indices).astype(PETSc.IntType), comm=COMM_SELF)
        iset = PETSc.IS().createGeneral(indices[self.perm.getIndices()],
                                        comm=COMM_SELF)
        iset.setBlockSize(V.block_size)

        self.aux = self.permuted(local.mat)
        hpddmpc.setOperators(A, P)
        hpddmpc.setHPDDMAuxiliaryMat(iset, self.aux)
        # The auxiliary matrix we just supplied really is the local Neumann
        # matrix, so say so.  This only records the fact: it saves PCHPDDM
        # extracting submatrices to build the eigenproblem, and it is what
        # lets the subdomain factorisation be shared with the eigensolver's
        # spectral transformation.  The fine-level solve uses the
        # corresponding block of P either way.
        hpddmpc.setHPDDMHasNeumannMat(True)

        # we may inject some options, we remove them after setting the PC up
        opts = PETSc.Options(hpddmpc.getOptionsPrefix())
        defaults = {}

        def default(key, value):
            if key not in opts:
                defaults[key] = value

        # Without this PCASM picks its own subdomains and ignores the IS above
        default("pc_hpddm_define_subdomains", True)

        # PCASM defaults to the restricted method, which is not symmetric and
        # so cannot be used under CG.  The basic (additive) method is, and it
        # pairs with the balanced coarse correction, which is symmetric where
        # the default deflated one is not.
        default("pc_hpddm_levels_1_pc_asm_type", "basic")
        default("pc_hpddm_coarse_correction", "balanced")

        # Without an eps_nev or a threshold there is no coarse level at all
        if not any(f"pc_hpddm_levels_1_{k}" in opts
                   for k in ("eps_nev", "eps_threshold_absolute",
                             "eps_threshold", "svd_nsv")):
            default("pc_hpddm_levels_1_eps_nev", self.DEFAULT_EPS_NEV)

        self.pc = hpddmpc
        self.iset = iset
        self.local = local
        self.opts = opts
        self.defaults = defaults
        self.set_up_inner_pc()

    def set_up_inner_pc(self):
        """Set the inner PC up with our default options in the database.

        PCHPDDM reads ``pc_hpddm_define_subdomains`` and the ``PCASM`` options
        in ``PCSetUp`` rather than in ``PCSetFromOptions``, so the injected
        defaults have to survive until the PC is set up.  Setting the auxiliary
        matrix resets the PC, so this runs again after every reassembly.
        """
        for key, value in self.defaults.items():
            self.opts[key] = value
        try:
            self.pc.setFromOptions()
            self.pc.setUp()
        finally:
            for key in self.defaults:
                del self.opts[key]

    def permuted(self, mat):
        """Return ``mat`` reordered to match the sorted subdomain index set."""
        return mat.permute(self.perm, self.perm)

    def view(self, pc, viewer=None):
        self.pc.view(viewer=viewer)

    def update(self, pc):
        self.local.update()
        # petsc4py binds no setup callback for the auxiliary matrix, so
        # reassembling it in place would leave PCHPDDM holding a coarse space
        # built from the old values.  Setting it again resets the PC and forces
        # the coarse space to be rebuilt.
        aux = self.permuted(self.local.mat)
        self.pc.setHPDDMAuxiliaryMat(self.iset, aux)
        self.aux.destroy()
        self.aux = aux
        self.set_up_inner_pc()

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)

    def destroy(self, pc):
        for name in ("aux", "iset", "perm", "pc"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()


def validate_overlap(mesh):
    """Warn if the mesh halo is too thin to give overlapping subdomains.

    The subdomains of ``PCHPDDM`` are the cells owned by each process together
    with their halo, so a mesh distributed without an overlap yields subdomains
    that do not overlap at all and a method that degenerates to block Jacobi.

    Parameters
    ----------
    mesh : MeshGeometry
        The mesh the operator is defined on.
    """
    if mesh.comm.size == 1:
        return
    overlap_entity, overlap_depth = mesh._distribution_parameters["overlap_type"]
    if overlap_entity == DistributedMeshOverlapType.NONE or overlap_depth < 1:
        warning("HPDDMPC needs a mesh distributed with an overlap, but this one "
                f"has overlap_type ({overlap_entity}, {overlap_depth}), so the "
                "subdomains do not overlap and the method reduces to block "
                "Jacobi.  Did you forget to set overlap_type in your mesh's "
                "distribution_parameters?")
