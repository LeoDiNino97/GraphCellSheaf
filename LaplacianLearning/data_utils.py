import numpy as np 
from itertools import combinations

def random_ER_graph(
        V:int
        ) -> list:
    
    '''
    Generate random Erdos-Renyi graph of a given number of nodes with probability slighlty higher than the connection threshold 

    Parameters:
    - V (int): The number of nodes.

    Returns:
    - list: collection of edges  
    '''

    edges = []

    for u in range(V):
        for v in range(u+1, V):
            p = np.random.uniform(0,1,1)
            if p < 1.3*np.log(V)/V:
                edges.append((u,v))

    return edges

def random_sheaf(
        V:int,
        d:int,
        edges:list
        ) -> np.array:
    
    '''
    Generate random sheaf laplacian whose restriction maps are randomly sampled from a gaussian distribution 

    Parameters:
    - V (int): The number of nodes.
    - d (int): Stalks dimension
    - edges (list): list of the edges of the underlying graph

    Returns:
    - np.array: sheaf laplacian
    '''

    E = len(edges)

    # Incidency linear maps

    F = {
        e:{
            e[0]:np.random.randn(d,d),
            e[1]:np.random.randn(d,d)
            } 
            for e in edges
        }                                           

    # Coboundary maps

    B = np.zeros((d*E, d*V))                        

    for i in range(len(edges)):

        # Main loop to populate the coboundary map

        edge = edges[i]

        u = edge[0] 
        v = edge[1] 

        B_u = F[edge][u]
        B_v = F[edge][v]

        B[i*d:(i+1)*d, u*d:(u+1)*d] = B_u           
        B[i*d:(i+1)*d, v*d:(v+1)*d] = - B_v

    L_f = B.T @ B

    return L_f

def synthetic_data(
        N:int, 
        d:int,
        V:int,
        L:np.array
        ) -> np.array:
    '''
    Generate synthetic smooth signals based on a given sheaf laplacian.

    Parameters:
    - N (int): The number of signals to generate.
    - d (int): The stalk dimension.
    - V (int): The number of nodes.
    - L (np.array): A numpy array representing the sheaf laplacian.

    Returns:
    - np.array: A numpy array of shape (V*d, N) containing the synthetic data.    
    '''

    # Generate random signals over the stalks of the vertices
    X = np.random.randn(V*d,N)

    # Retrieve the eigendecomposition of the sheaf laplacian
    Lambda, U = np.linalg.eig(L)

    # Tikhonov regularization based approach
    H = 1/(1 + 10*Lambda)

    # Propect into vertices domain <- filter out <- project into spectrum of laplacian
    Y = U @ np.diag(H) @ U.T @ X

    # Add gaussian noise
    Y += np.random.normal(0, 10e-2, size=Y.shape)

    return Y

# ___________________________________________________________________________________________________
# ______________________________________METRICS FOR EVALUATION_______________________________________
# ___________________________________________________________________________________________________

def reconstructed_laplacian_metrics(
        V:int,
        edges:list,
        d:int,
        maps:dict,
        X:np.array,
        L_f:np.array,
        ) -> dict:
    
    '''
    Retrieve the first E edges based on their expressed energy and compute metrics of similarities with the original laplacian.

    Parameters:
    - V (int): number of nodes.
    - edges (list): edges of the underlying graph
    - d (int): stalks dimensions
    - maps (dict) dictionary of restriction maps 
    - X (np.array): dataset of shape (V*d, N) of smooth signals
    - L_f (np.array): groundthruth for the sheaf laplacian
    
    Returns:
    - dict: dictionary containing three metrics as chosen by Hansen: average entrywise L2 and L1 reconstruction error, precision in recovering the graph underlying the sheaf
    '''

    E = len(edges)

    all_edges = list(combinations(range(V), 2))

    energies = {
        e : 0
        for e in all_edges
        }
    
    for e in all_edges:
        u = e[0]
        v = e[1]

        X_u = X[u*d:(u+1)*d,:]
        X_v = X[v*d:(v+1)*d,:]


        F_u = maps[e][u]
        F_v = maps[e][u]

        L = 0

        for i in range(X.shape[1]):
            x_u = X_u[:,i]
            x_v = X_v[:,i]
            L += np.linalg.norm(F_u @ x_u - F_v @ x_v)
        
        energies[e] = L

    retrieved = sorted(energies.items(), key=lambda x:x[1])[:E]

    B_hat = np.zeros((d*E, d*V))

    for i in range(E):
        edge = retrieved[i][0]

        u = edge[0] 
        v = edge[1] 

        B_u = maps[edge][u]
        B_v = maps[edge][v]

        B_hat[i*d:(i+1)*d, u*d:(u+1)*d] = B_u
        B_hat[i*d:(i+1)*d, v*d:(v+1)*d] = - B_v

    L_f_hat = B_hat.T @ B_hat

    return {
        "AvgEntryWiseED_L2" : np.sqrt(np.sum((L_f - L_f_hat)**2)) / L_f.size,
        "AvgEntryWiseED_L1" : np.sqrt(np.sum(np.abs(L_f - L_f_hat))) / L_f.size,
        "SparsityAccuracy" : len(set(list(map(lambda x: x[0], retrieved))).intersection(set(edges))) / E
        }
