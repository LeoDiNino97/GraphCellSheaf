import numpy as np
import optimization_utils as opt

def learning(
        V:int,
        d:int,
        X:np.array,
        LR:np.array,
        rho:float,
        lambda_:float,
        gamma:float,
        T:int,
        t:int,
        max_iter:int
    ):
    
    # Initialization 

    maps_0 = opt.initialization(V,d)
    edges_dict = opt.edges_map(V)
    edges = list(edges_dict.keys())
    E = len(edges)

    # Every local agent has its own buffer for previous iteration

    buffer_A = {
        e: {
            e[0] : np.eye(d),
            e[1] : np.eye(d)
        }
        for e in edges
    }
    
    # Every local agent has its own buffer for receiving global messages

    buffer_B = {
        e: {
            'BB_uu': np.eye(d),
            'BB_uv': np.eye(d),
            'BB_vv': np.eye(d),
            'L_uu': np.eye(d),
            'L_uv': np.eye(d),
            'L_vv': np.eye(d),
            'M_uu': np.eye(d),
            'M_uv': np.eye(d),
            'M_vv': np.eye(d)
        }
    }

    # Initialization of the shared multiplier

    M = np.eye(V*d)

    for _ in max_iter:

        # Initializing the central aggregator

        BB_ = np.zeros((V*d, V*d))

        # Inner loop for the local updates

        for e in edges:
            u = e[0]
            v = e[1]

            # Data on the nodes of interest

            X_u = X[u*d:(u+1)*d,:]
            X_v = X[u*d:(u+1)*d,:]
            
            # We use the same initialization for both the inner SCA and the outer ADMM
            
            F_u_0 = maps_0[e][u]
            F_v_0 = maps_0[e][u]

            # Retrieve previous iteration from the buffer

            F_u_k = buffer_A[e][u]
            F_v_k = buffer_A[e][v]

            # Received messages from central aggregator

            L_uu = buffer_B[e]['L_uu']
            L_uv = buffer_B[e]['L_uv']
            L_vv = buffer_B[e]['L_vv']

            BB_uu = buffer_B[e]['BB_uu']
            BB_uv = buffer_B[e]['BB_uv']
            BB_vv = buffer_B[e]['BB_vv']

            M_uu = buffer_B[e]['M_uu']
            M_uv = buffer_B[e]['M_uv']
            M_vv = buffer_B[e]['M_vv']

            # Restriction maps learning

            F_u, F_v = opt.local_inexact_SCA(X_u, X_v,
                                            F_u_0, F_u_k,
                                            F_v_0, F_v_k,
                                            L_uu, L_uv, L_vv,
                                            BB_uu, BB_uv, BB_vv,
                                            M_uu, M_uv, M_vv,
                                            rho, LR, gamma,
                                            T, t)

            # Update local buffer
            buffer_A[e][u] = F_u
            buffer_A[e][v] = F_v

            # Send the message to the central aggregator

            BB_ += opt.local_to_global(F_u, F_v, e, d, V, edges_dict)

        # Central updates

        L = opt.global_update_L(lambda_, rho, E, BB_, M)
        M = opt.global_update_M(M, BB_, L)

        # Send the messages from the central aggregator to local agents

        for e in edges:
            buffer_B[e] = opt.global_to_local(d, e, L, BB_, M)

    # We provide in output all the computed restriction maps 
    
    return buffer_A


