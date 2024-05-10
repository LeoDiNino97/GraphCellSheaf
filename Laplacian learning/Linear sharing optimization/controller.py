import numpy as np
import optimization_utils as opt
from tqdm import tqdm

def learning(
        V:int,
        d:int,
        X:np.array,
        rho:float,
        lambda_:float,
        mu:float,
        max_iter:int
    ):
    
    # Initialization 

    edges_dict = opt.edges_map(V)
    edges = list(edges_dict.keys())
    E = len(edges)

    # Every local agent has its own buffer for previous iteration
    buffer_A = {
        e: np.zeros((d*V,d*V))
        for e in edges
    }

    for e in edges:
        u = e[0]
        v = e[0]

        buffer_A[e][u*d:(u+1)*d,u*d:(u+1)*d] = np.eye(d)
        buffer_A[e][v*d:(v+1)*d,v*d:(v+1)*d] = np.eye(d)

        buffer_A[e][u*d:(u+1)*d,v*d:(v+1)*d] = - np.eye(d)
        buffer_A[e][v*d:(v+1)*d,u*d:(u+1)*d] = - np.eye(d)

    # Every local agent has its own buffer for receiving global messages

    buffer_B = {
        e: (np.ones(d*V),np.ones(d*V))
        for e in edges
    }

    # Initialization of the shared multiplier

    M = np.eye(d*V)

    for _ in tqdm(range(max_iter)):

        # Initializing the central aggregator

        BB_ = np.zeros((V*d, V*d))

        # Inner loop for the local updates

        for e in edges:

            u = e[0]
            v = e[1]

            # Local update
            BB = buffer_A[e]
            L_hat = buffer_B[e][1]
            BB_hat = buffer_B[e][0]

            BB_e = opt.local_updates(e, d, X, L_hat, BB, BB_hat, M, rho)

            # Update local buffer
            BB_e_ = np.zeros_like(BB_e)

            BB_e_[u*d:(u+1)*d,u*d:(u+1)*d] = BB_e[u*d:(u+1)*d,u*d:(u+1)*d]
            BB_e_[v*d:(v+1)*d,v*d:(v+1)*d] = BB_e[v*d:(v+1)*d,v*d:(v+1)*d]

            BB_e_[u*d:(u+1)*d,v*d:(v+1)*d] = BB_e[u*d:(u+1)*d,v*d:(v+1)*d]
            BB_e_[v*d:(v+1)*d,u*d:(u+1)*d] = BB_e[v*d:(v+1)*d,u*d:(u+1)*d]

            buffer_A[e] = BB_e_

            # Send the message to the central aggregator

            BB_ += BB_e

        # Central updates
        BB_ /= E
        L = opt.global_update_L_CVXPY(mu, lambda_, rho, E, BB_, M)
        M = opt.global_update_M(M, BB_, L)

        # Send the messages from the central aggregator to local agents

        for e in edges:
            buffer_B[e] = (BB_, L)

    # We provide as output all the computed restriction maps 

    return buffer_A


