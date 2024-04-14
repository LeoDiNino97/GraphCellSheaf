# Graph signal processing and cellular sheaf  

This repository contains implementations and results related to different experiments made to generalize the known results of graph signal processing on the cellular sheaf, a more expressive data structure that expands the domain of definition of signals over a graph.

Considering a graph $G(V,E)$, a cellular sheaf on a graph is made up of
+ A vectorial space $\mathcal{F}_v$ for each node $v \in V$,
+ A vectorial space $\mathcal{F}_e$ for each edge $e \in E$,
+ A linear map $\mathcal{F}_{v \triangleleft e} : \mathcal{F}_v \rightarrow \mathcal{F}_e$ for each incidency $v \triangleleft e$, for each node $v \in V$, for each edge $e \in E$.
