# Graph signal processing and cellular sheaves  

This repository contains implementations and results related to different experiments made to generalize the known results of graph signal processing on the graph cellular sheaves, a more expressive data structure that expands the domain of definition of signals over a graph.

Considering a graph $G(V,E)$, a cellular sheaf $\mathcal{F}$ on a graph is made up of
+ A vectorial space $\mathcal{F}_v$ for each node $v \in V$,
+ A vectorial space $\mathcal{F}_e$ for each edge $e \in E$,
+ A linear map $\mathcal{F}_{v \triangleleft e} : \mathcal{F}_v \rightarrow \mathcal{F}_e$ for each incidency $v \triangleleft e$, for each node $v \in V$, for each edge $e \in E$.

The most obvious vectorial spaces to derive from this definition are the so called space of cochains: they result from a direct sum of the spaces (namely stalks) defined over the nodes and the edges respectively, so that an element belonging to a space of cochain is just the stack of the signals defined over all the nodes or the edges:

$$C^0(G,\mathcal{F}) = \bigoplus_{v \in V} \mathcal{F}_v$$

$$C^1(G,\mathcal{F}) = \bigoplus_{e \in E} \mathcal{F}_e$$

The block matrix collecting all the linear maps according to a fixed oriented incidency is called the coboundary map $\delta$: the sheaf laplacian can be derived from the coboundary map similarly to how we derive the graph laplacian from the incidency matrix: 

$$ L_{\mathcal{F}} = \delta^T \delta $$

This laplacian encodes both for the linear maps between the stalks and for the sparsity for the underlying graph. Its spectral properties allows for a generalization of the tools of graph signal processing and even more, under proper definitions, for generalizations in higher order topologies (simplicial complexes, cell complexes). 

