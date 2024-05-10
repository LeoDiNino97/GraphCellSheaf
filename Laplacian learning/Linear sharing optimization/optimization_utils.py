import numpy as np 
from itertools import combinations

def initialization(
        V:int,
        d:int
        ) -> dict:
    '''
    Initialize a dictionary with all the restriction maps of all possible edges.

    Parameters:
    - V (int): The number of nodes.
    - d (int): The stalk dimension.

    Returns:
    - dict: A dictionary containing the initialized restriction maps.    
    '''

    all_edges = list(combinations(range(V),2))
    maps = {
        e: np.eye(d*V)
        for e in all_edges
    }

    return maps

def edges_map(
        V:int
        ) -> dict:
    
    '''
    Initialize a dictionary with all the possible edges between V nodes.

    Parameters:
    - V (int): The number of nodes.

    Returns:
    - dict: A dictionary containing the edges and a progressive integer as identifier.    
    '''

    all_edges = list(combinations(range(V),2))

    edges_dict = {
        all_edges[i]: i
        for i in range(len(all_edges))
    }

    return edges_dict

#___________________________________________________________________________________
#________________________________LOCAL UPDATE_______________________________________
#___________________________________________________________________________________

def local_updates(
        edge:tuple,
        d:int,
        X:np.array,
        L_hat:np.array,
        BB:np.array,
        BB_hat:np.array,
        M:np.array,
        rho:float,
        ) -> tuple:
    
    '''
    Optimize the local blocks on a certain edge in a closed form (strongly convex problem in each block) 

    Parameters:


    Returns:
    - np.array: An array containing the block defined on the edge of interest    
    '''
    u = edge[0]
    v = edge[1]

    X_ = np.zeros_like(X)

    X_[u*d:(u+1)*d,:] = X[u*d:(u+1)*d,:]
    X_[v*d:(v+1)*d,:] = X[v*d:(v+1)*d,:]

    return 1/rho * (BB + L_hat - BB_hat - M - X_ @ X_.T)


#__________________________________________________________________________________________
#_________________________________CENTRAL AGGREGATION______________________________________
#___________________________________________________________________________________________

def global_update_L(
        mu:float,
        lambda_:float,
        rho:float,
        E:int,
        BB:np.array,
        M:np.array
        ) -> np.array:
    
    '''
    Global update of the estimated laplacian through a proximal mapping

    Parameter:
    - lambda_ (float): Penalization factor
    - rho (float): coefficient of the augmentation term in the Lagrangian equation of the problem 
    - E (int): total number of edges
    - BB (np.array): aggregator of local-to-central messages
    - M (np.array): shared multiplier

    Returns:
    - np.array: globally updated sheaf laplacian
    '''
    
    # Computing the mu coefficient

    mu = mu * E / rho

    # Computing the gamma coefficient

    gamma = lambda_/(rho*E)

    # Computing the eigendecomposition of the term to be proximalized

    s, U = np.linalg.eig(BB + M)

    # Computing the eigenvalues of the proximal operator

    z = 1/(2*(2*mu*gamma + 1))*(s + np.sqrt(s**2 + 4*gamma*(2*gamma*mu + 1)))

    # Rebuilding the proximal operator

    L = U @ np.diag(z) @ U.T

    return L

def global_update_M(
        M:np.array,
        BB:np.array,
        L:np.array
        ) -> np.array:
    
    '''
    Global update of the shared multiplier

    Parameter:
    - M (np.array): shared multiplier
    - BB (np.array): aggregator of local-to-central messages
    - L (np.array): current estimate of the global variable for the laplacian

    Returns:
    - np.array: globally updated multiplier
    '''

    return M + BB - L 


    


