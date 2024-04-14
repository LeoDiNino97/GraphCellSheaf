# Graph signal processing and cellular sheaf  

This repository contains implementations and results related to different experiments made to generalize the known results of graph signal processing on the graph cellular sheaves, a more expressive data structure that expands the domain of definition of signals over a graph.

Considering a graph $G(V,E)$, a cellular sheaf $\mathcal{F}$ on a graph is made up of
+ A vectorial space $\mathcal{F}_v$ for each node $v \in V$,
+ A vectorial space $\mathcal{F}_e$ for each edge $e \in E$,
+ A linear map $\mathcal{F}_{v \triangleleft e} : \mathcal{F}_v \rightarrow \mathcal{F}_e$ for each incidency $v \triangleleft e$, for each node $v \in V$, for each edge $e \in E$.

The most obvious vectorial spaces to derive from this definition are the so called space of cochains: they result from a direct sum of the spaces defined over the nodes and the edges respectively, so that an element belonging to a space of cochain is just the stack of the signals defined over all the nodes or the edges:

\begin{equation}  
  C^0(G,\mathcal{F}) = \bigoplus_{v \in V} \mathcal{F}_v \\ \nonumber
\end{equation}

\begin{equation}
  C^1(G,\mathcal{F}) = \bigoplus_{e \in E} \mathcal{F}_e \\
\end{equation}
