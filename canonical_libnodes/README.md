## Canonical Library Nodes and Expansion

While perfoming DSE, the application SDFG is converted to a Canonical Directed Acyclic Graphs (Canonical DAGs) for scheduling.

For doing this we use  the multi-level and data-centric  SDFG intermediate representation.This allow us:
- to programmatically lower high-level operations (e.g., Matmuls) into their canonical representations;
- enable data-access patterns analysis to detect which communications can be streamed or necessitate buffering nodes.




This folder contains a collection of expansions for DaCe library nodes whose expansions can be naturally converted to Canonical DAG.
For this reason we refer at them as "canonical expansions".

For additional details on this process, please refer to the development wiki.


### Content of this folder

The expansions are organized in different subfolders:

- `blas/`: expansions for BLAS numerical routines (e.g., matrix-vector and matrix-matrix multiplications)
- `others`: other numerical routines (e.g., cholesky factorization, and matrix inversion)
- `ml/`: expansions for Machine Learning operator (to be used with DaCeML generated programs)
- `misc/`: utility expansions/library nodes that are used by other expansions (e.g., nodes for broadcasting/gathering data)