import numpy as np 

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


def random_ER_graph(
        V:int
        ) -> list:
    
    edges = []

    for u in range(V):
        for v in range(u, V):
            p = np.random.uniform(0,1,1)
            if p < 1.1*np.log(V)/V:
                edges.append((u,v))

    return edges

def random_sheaf(
        V:int,
        d:int,
        edges:list
        ) -> np.array:
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