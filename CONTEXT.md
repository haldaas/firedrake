# Context

Terms whose meaning in this codebase is not the obvious one, or that mean
different things in different layers.

## subdomain

The word carries two unrelated meanings, one per layer. Both are correct; do
not try to reconcile them.

**In the mesh layer** a subdomain is a *marked region of a mesh*: a set of
cells or facets carrying a marker, as in `Submesh(mesh, subdomain_id=...)`,
`mesh.cell_subset(subdomain_id)`, `SubDomainData`, and the `Cell Sets` and
`Face Sets` labels. It says nothing about parallelism.

**In the domain-decomposition preconditioners** — `firedrake/preconditioners/matis.py`,
`bddc.py`, and anything built on `Mat` of type `is` — a subdomain is a *piece of
the decomposition of the operator*: the unit on which a local solve is done, in
the sense of the domain-decomposition literature. Subdomains need not
correspond to any marked region, and there may be one per cell, one per MPI
process, or several per process.

The domain-decomposition sense is the one to use in new domain-decomposition
code, matching the literature. Reach for "region" if a mesh marker has to be
mentioned nearby.
