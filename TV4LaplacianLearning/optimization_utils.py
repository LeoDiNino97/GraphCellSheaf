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
        e:{
            e[0] : np.eye(d),
            e[1] : np.eye(d)
        }
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

def local_inexact_SCA(
        X_u:np.array,
        X_v:np.array,
        F_u_0:np.array,
        F_u_k:np.array,
        F_v_0:np.array,
        F_v_k:np.array,
        L_uu:np.array,
        L_uv:np.array,
        L_vv:np.array,
        BB_uu:np.array,
        BB_uv:np.array,
        BB_vv:np.array,
        M_uu:np.array,
        M_uv:np.array,
        M_vv:np.array,
        rho:float,
        LR:float,
        gamma:float,
        T:int,
        t:int
        ) -> tuple:
    
    '''
    This routine performs an inexact successive convex approximation to optimize the local restriction maps
    on a certain edge calling a subroutine performing a gradient based procedure. 

    Parameters:
    - X_u (np.array): Signals on node u 
    - X_v (np.array): Signals on node v
    - F_u_0 (np.array): Initialization for map F_u
    - F_u_k (np.array): Buffered previous iteration result for F_u
    - F_v_0 (np.array): Initialization for map F_v
    - F_v_k (np.array): Buffered previous iteration result for F_u
    - L_uu (np.array): (u,u) block in the global variable for the sheaf laplacian 
    - L_uv (np.array): (u,v) block in the global variable for the sheaf laplacian
    - L_vv (np.array): (v,v) block in the global variable for the sheaf laplacian
    - BB_uu (np.array): (u,u) block in the global aggregator for local to global messages
    - BB_uv (np.array): (u,v) block in the global aggregator for local to global messages
    - BB_vv (np.array): (v,v) block in the global aggregator for local to global messages 
    - M_uu (np.array): (u,u) block in the shared multiplier 
    - M_uv (np.array): (u,v) block in the shared multiplier  
    - M_vv (np.array): (v,v) block in the shared multiplier 
    - rho (float): coefficient of the augmentation term in the Lagrangian equation of the problem 
    - LR (float): learning rate for the inner gradient descent subroutine
    - gamma (float): convex smoothing parameter for the outer SCA routine
    - T (int): max number of iterations for outer routine (SCA)
    - t (int): max number of iterations for inner routine (gradient based)

    Returns:
    - tuple: A tuple containing the two restriction maps on the edge of interest    
    '''

    # Initialization

    F_u = F_u_0
    F_v = F_v_0

    for _ in range(T):

        # Calling inexact solvers

        F_u_hat = gradient_descent_U(X_u, X_v, 
                                     F_u, F_u_k, F_v, F_v_k, 
                                     L_uu, L_uv, 
                                     BB_uu, BB_uv, 
                                     M_uu, M_uv,
                                     rho, LR, t)
        
        F_v_hat = gradient_descent_V(X_u, X_v, 
                                     F_u, F_u_k, F_v, F_v_k, 
                                     L_uv, L_vv, 
                                     BB_uv, BB_vv, 
                                     M_uv, M_vv,
                                     rho, LR, t)

        # Convex smoothing
        F_u = F_u + gamma*(F_u_hat - F_u)
        F_v = F_v + gamma*(F_v_hat - F_v)

        gamma *= 0.9

    return (F_u, F_v)


def gradient_descent_U(
        X_u:np.array,
        X_v:np.array,
        F_u_0:np.array,
        F_u_k:np.array,
        F_v_0:np.array,
        F_v_k:np.array,
        L_uu:np.array,
        L_uv:np.array,
        BB_uu:np.array,
        BB_uv:np.array,
        M_uu:np.array,
        M_uv:np.array,
        rho:float,
        LR:float,
        t:int
        ) -> np.array:
    
    '''
    This subroutine performs the gradient based procedure on block F_u in the local optimization step. 

    Parameters:
    - X_u (np.array): Signals on node u 
    - X_v (np.array): Signals on node v
    - F_u_0 (np.array): Initialization for map F_u
    - F_u_k (np.array): Buffered previous iteration result for F_u
    - F_v_0 (np.array): Initialization for map F_v
    - F_v_k (np.array): Buffered previous iteration result for F_u
    - L_uu (np.array): (u,u) block in the global variable for the sheaf laplacian 
    - L_uv (np.array): (u,v) block in the global variable for the sheaf laplacian
    - BB_uu (np.array): (u,u) block in the global aggregator for local to global messages
    - BB_uv (np.array): (u,v) block in the global aggregator for local to global messages
    - M_uu (np.array): (u,u) block in the shared multiplier 
    - M_uv (np.array): (u,v) block in the shared multiplier  
    - rho (float): coefficient of the augmentation term in the Lagrangian equation of the problem 
    - LR (float): learning rate for the inner gradient descent subroutine
    - t (int): max number of iterations for inner routine (gradient based)

    Returns:
    - np.array: The restriction map F_u  
    '''    
    
    # External initialization 

    F_u = F_u_0
    F_v = F_v_0

    # This quantities in the gradients of the block losses are fixed
     
    l_uu = - F_u_k.T @ F_u_k + BB_uu - L_uu + M_uu
    l_uv = + F_u_k.T @ F_v_k + BB_uv - L_uv + M_uv

    for _ in range(t):
        l_uu_ = F_u.T @ F_u + l_uu
        l_uv_ = - F_u.T @ F_v + l_uv

        grad_u = ( (F_u @ X_u - F_v @ X_v) @ X_u.T               # Gradient of the loss on the local communication
                  + rho * (F_u @ l_uu_ - 2*F_v @ l_uv_) )        # Gradient of the block-wise losses

        F_u -= LR * grad_u

    return F_u

def gradient_descent_V(
        X_u:np.array,
        X_v:np.array,
        F_u_0:np.array,
        F_u_k:np.array,
        F_v_0:np.array,
        F_v_k:np.array,
        L_uv:np.array,
        L_vv:np.array,
        BB_uv:np.array,
        BB_vv:np.array,
        M_uv:np.array,
        M_vv:np.array,
        rho:float,
        LR:float,
        t:int
        ) -> np.array:
    
    '''
    This subroutine performs the gradient based procedure on block F_v in the local optimization step. 

    Parameters:
    - X_u (np.array): Signals on node u 
    - X_v (np.array): Signals on node v
    - F_u_0 (np.array): Initialization for map F_u
    - F_u_k (np.array): Buffered previous iteration result for F_u
    - F_v_0 (np.array): Initialization for map F_v
    - F_v_k (np.array): Buffered previous iteration result for F_u
    - L_uv (np.array): (u,v) block in the global variable for the sheaf laplacian 
    - L_vv (np.array): (v,v) block in the global variable for the sheaf laplacian
    - BB_uv (np.array): (u,v) block in the global aggregator for local to global messages
    - BB_vv (np.array): (v,v) block in the global aggregator for local to global messages
    - M_uv (np.array): (u,v) block in the shared multiplier 
    - M_vv (np.array): (v,v) block in the shared multiplier  
    - rho (float): coefficient of the augmentation term in the Lagrangian equation of the problem 
    - LR (float): learning rate for the inner gradient descent subroutine
    - t (int): max number of iterations for inner routine (gradient based)

    Returns:
    - np.array: The restriction map F_v
    '''    

    # External initializations

    F_u = F_u_0
    F_v = F_v_0

    # This quantities in the gradients of the block losses are fixed

    l_vv = - F_v_k.T @ F_v_k + BB_vv - L_vv + M_vv
    l_uv = + F_u_k.T @ F_v_k + BB_uv - L_uv + M_uv

    for _ in range(t):
        l_vv_ = F_v.T @ F_v + l_vv
        l_uv_ = - F_u.T @ F_v + l_uv

        grad_v = ( - (F_u @ X_u - F_v @ X_v) @ X_v.T       # Gradient of the loss on the local communication
                  + rho * (F_v @ l_vv_ - 2*F_u @ l_uv_) )  # Gradient of the block-wise losses
                                                           

        F_v -= LR * grad_v

    return F_v

def local_to_global(
        F_u:np.array,
        F_v:np.array,
        e:tuple,
        d:int,
        V:int,
        edges_dict:dict
        ) -> np.array:
    
    '''
    This subroutine performs the message passing from the local agents to the central aggregator 

    Parameters:
    - F_u (np.array): Restriction map F_u
    - F_v (np.array): Restriction map F_v
    - e (tuple): The edge where (u,v) are incident on
    - d (int): Dimension of the vertices stalks
    - V (int): Number of vertices
    - edges_dict (dict): Dictionary containing the identifiers for the edges

    Returns:
    - np.array: The local-to-global message
    '''        

    E = len(list(combinations(range(V),2)))
    B_e = np.zeros((d*E, d*V))

    u = e[0]
    v = e[1]

    id = edges_dict[e]

    B_e[id*d:(id+1)*d, u*d:(u+1)*d] = F_u
    B_e[id*d:(id+1)*d, v*d:(v+1)*d] = F_v

    return B_e.T @ B_e


#__________________________________________________________________________________________
#_________________________________CENTRAL AGGREGATION______________________________________
#___________________________________________________________________________________________

def global_update_L(
        lambda_:float,
        rho:float,
        E:int,
        BB:np.array,
        M:np.array
        ) -> np.array:
    
    # Computing the eigendecomposition of the term to be proximalized

    s, U = np.linalg.eig(BB + M)

    # Computing the eigenvalues of the proximal operator

    z = 0.5*(s + np.sqrt(s**2 + 4*lambda_/(rho*E)))

    # Rebuilding the proximal operator

    L = U @ np.diag(z) @ U.T

    return L

def global_update_M(
        M:np.array,
        BB:np.array,
        L:np.array
        ) -> np.array:
    
    return M + BB - L 

def global_to_local(
        d:int,
        edge:tuple,
        L:np.array,
        BB:np.array,
        M:np.array
        ) -> dict:
    
    u = edge[0]
    v = edge[1]

    # Defining the block to be sent for the message passing 

    BB_uu = BB[u*d:(u+1)*d,u*d:(u+1)*d]
    BB_uv = BB[u*d:(u+1)*d,v*d:(v+1)*d]
    BB_vv = BB[v*d:(v+1)*d,v*d:(v+1)*d]

    L_uu = L[u*d:(u+1)*d,u*d:(u+1)*d]
    L_uv = L[u*d:(u+1)*d,v*d:(v+1)*d]
    L_vv = L[v*d:(v+1)*d,v*d:(v+1)*d]

    M_uu = M[u*d:(u+1)*d,u*d:(u+1)*d]
    M_uv = M[u*d:(u+1)*d,v*d:(v+1)*d]
    M_vv = M[v*d:(v+1)*d,v*d:(v+1)*d]

    # Mapping the block to be sent
    D = {
        'BB_uu':BB_uu,
        'BB_uv':BB_uv,
        'BB_vv':BB_vv,
        'L_uu':L_uu,
        'L_uv':L_uv,
        'L_vv':L_vv,
        'M_uu':M_uu,
        'M_uv':M_uv,
        'M_vv':M_vv
    }

    return D


    


