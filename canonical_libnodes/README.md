## Canonical Library Nodes and Expansion

While perfoming DSE, the application SDFG is converted to a Canonical Directed Acyclic Graphs (Canonical DAGs) for scheduling.

For doing this we use  the multi-level and data-centric  SDFG intermediate representation.This allow us:
- to programmatically lower high-level operations (e.g., Matmuls) into their canonical representations;
- enable data-access patterns analysis to detect which communications can be streamed or necessitate buffering nodes.




This folder contains a collection of expansions for DaCe library nodes whose expansions can be naturally converted to Canonical DAG.
For this reason we refer at them as "canonical expansions".

For additional details on this process, please refer to the development wiki.
