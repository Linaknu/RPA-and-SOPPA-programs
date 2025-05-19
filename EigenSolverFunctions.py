import numpy as np
from eigensolver_data import *
from MatrixFunctions import *  



def get_indices_of_k_smallest(arr, k): #Find k lowest values (and indices) in 2d array and sort them
    idx = np.argpartition(arr.ravel(), k)
    idx_k_lowest = tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k,0), max(k, 0))])
    return idx_k_lowest[0][np.argsort(arr[idx_k_lowest])],idx_k_lowest[1][np.argsort(arr[idx_k_lowest])]


#Make zeroth order diagonal

def Zeroth_order_positive_eigenvalues(occ, vir, ri, mo_e, k):

    Zeroth_order_diagonal_positive = np.full((Length_of_D_S1,3),np.nan)

    idx_ai = 0
    idx_S1_aibj = 0
    idx_S2_aibj = 0
    for i in range(occ):
        for a in range(vir):
            Zeroth_order_diagonal_positive[idx_ai,0] = A(i,a,i,a,A_order=0)
            for j in range(i,occ):
                for b in range(a, vir):

                    Zeroth_order_diagonal_positive[idx_S1_aibj,1] = D_func_S1S2(i,a,j,b)

                    if i != j: 
                        if a != b:
                            Zeroth_order_diagonal_positive[idx_S2_aibj,2] = D_func_S1S2(i,a,j,b)
                            idx_S2_aibj += 1
                    idx_S1_aibj += 1
            idx_ai += 1  

    np.set_printoptions(formatter={'float': '{: 0.12f}'.format})

    idx_0, idx_1 = get_indices_of_k_smallest(Zeroth_order_diagonal_positive, k)
    
    #second axis is: A, DS1, DS2
    return idx_0, idx_1

#Generates the initial k trial vectors
def initial_k_trialvec(k):    

    idx_0, idx_1 = Zeroth_order_positive_eigenvalues(occ, vir, ri, mo_e, k)

    b_E = np.zeros((Length_of_A,k))         ;   b_D = np.zeros((Length_of_A,k))
    b_EE_S1 = np.zeros((Length_of_D_S1,k))  ;   b_EE_S2 = np.zeros((Length_of_D_S2,k))
    b_DD_S1 = np.zeros((Length_of_D_S1,k))  ;   b_DD_S2 = np.zeros((Length_of_D_S2,k))

    for i in range(k):
        if idx_1[i] == 0:
            b_E[idx_0[i],i] = 1
        if idx_1[i] == 1:
            b_EE_S1[idx_0[i],i] = 1
        if idx_1[i] == 2:
            b_EE_S2[idx_0[i],i] = 1

    return b_E, b_D, b_EE_S1, b_EE_S2, b_DD_S1, b_DD_S2

#Compute the new u and m vectors 
def get_u_and_m(b_E, b_D, b_EE_S1, b_EE_S2, b_DD_S1, b_DD_S2, mo_e, count_new_trial_vector_pairs):

    Number_of_all_trial_vect_pairs = b_E.shape[1]
    O = Number_of_all_trial_vect_pairs - count_new_trial_vector_pairs #Number of Old trial vector pairs

    u_E = np.zeros((Length_of_A,count_new_trial_vector_pairs))           ;   u_D = np.zeros((Length_of_A,count_new_trial_vector_pairs))
    u_EE_S1 = np.zeros((Length_of_D_S1, count_new_trial_vector_pairs))   ;   u_EE_S2 = np.zeros((Length_of_D_S2, count_new_trial_vector_pairs))
    u_DD_S1 = np.zeros((Length_of_D_S1, count_new_trial_vector_pairs))   ;   u_DD_S2 = np.zeros((Length_of_D_S2, count_new_trial_vector_pairs))

    m_E = np.zeros((Length_of_A,count_new_trial_vector_pairs))      ;   m_D = np.zeros((Length_of_A,count_new_trial_vector_pairs))
    m_EE_S1 = b_EE_S1[:,-count_new_trial_vector_pairs:]             ;   m_EE_S2 = b_EE_S2[:,-count_new_trial_vector_pairs:]
    m_DD_S1 = -b_DD_S1[:,-count_new_trial_vector_pairs:]            ;   m_DD_S2 = -b_DD_S2[:,-count_new_trial_vector_pairs:]

    for o in range(count_new_trial_vector_pairs):

        index_ai = 0

        for i in range(occ): 
            for a in range(vir):

                index_bj = 0
                index_bjck_S1 = 0
                index_bjck_S2 = 0
                
                for j in range(occ):
                    for b in range(vir):

                        A_aibj = A(i,a,j,b, A_order)
                        #A_aibj = A_matrix[index_ai,index_bj]
                        B_aibj = B(i, a, j, b, order=2)


                        u_E[index_ai,o] += A_aibj * b_E[index_bj,o+O] + B_aibj * b_D[index_bj,o+O]
                        u_D[index_ai,o] += B_aibj * b_E[index_bj,o+O] + A_aibj * b_D[index_bj,o+O]

                        ph_S_aibj = ph_S(i,a,j,b)

                        m_E[index_ai,o] += ph_S_aibj *  b_E[index_bj,o+O]
                        m_D[index_ai,o] -= ph_S_aibj *  b_D[index_bj,o+O]
                        
                        for k in range(j, occ):
                            for c in range(b, vir):

                                C_S1_bjckai = C_S1_func( j,b,k,c,i,a)

                                u_E[index_ai,o] += C_S1_bjckai * b_EE_S1[index_bjck_S1,o+O]
                                u_D[index_ai,o] += C_S1_bjckai * b_DD_S1[index_bjck_S1,o+O]
                                
                                u_EE_S1[index_bjck_S1,o] += C_S1_bjckai * b_E[index_ai,o+O]
                                u_DD_S1[index_bjck_S1,o] += C_S1_bjckai * b_D[index_ai,o+O]

                                if index_ai == 0:
                                    D_S1_bjck = D_func_S1S2(j,b,k,c)

                                    u_EE_S1[index_bjck_S1,o] += D_S1_bjck * b_EE_S1[index_bjck_S1,o+O]
                                    u_DD_S1[index_bjck_S1,o] += D_S1_bjck * b_DD_S1[index_bjck_S1,o+O]

                                    if (b!=c)*(j!=k) == 1:
                                        D_S2_bjck = D_func_S1S2(j,b,k,c)

                                        u_EE_S2[index_bjck_S2,o] += D_S2_bjck * b_EE_S2[index_bjck_S2,o+O]
                                        u_DD_S2[index_bjck_S2,o] += D_S2_bjck * b_DD_S2[index_bjck_S2,o+O]


                                if (b!=c)*(j!=k) == 1:
                                    C_S2_bjckai = C_S2_func( j,b,k,c,i,a)

                                    u_E[index_ai,o] += C_S2_bjckai * b_EE_S2[index_bjck_S2,o+O]
                                    u_D[index_ai,o] += C_S2_bjckai * b_DD_S2[index_bjck_S2,o+O]

                                    u_EE_S2[index_bjck_S2,o] += C_S2_bjckai * b_E[index_ai,o+O]
                                    u_DD_S2[index_bjck_S2,o] += C_S2_bjckai * b_D[index_ai,o+O]

                                    index_bjck_S2 += 1
                                index_bjck_S1 += 1
                        index_bj += 1
                index_ai += 1


    return u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2, m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2


#Computes the reduced matrices. And checks symmetry
def get_reduced_matrices(b_E, b_D, b_EE_S1, b_EE_S2, b_DD_S1, b_DD_S2, u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2, m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2):
    
    alpha_B = (np.einsum('kl,km->lm',b_E[:,:],u_E[:,:]) + np.einsum('kl,km->lm',b_D[:,:],u_D[:,:]) + np.einsum('kl,km->lm',b_EE_S1[:,:],u_EE_S1[:,:]) 
                        +  np.einsum('kl,km->lm',b_EE_S2[:,:],u_EE_S2[:,:]) + np.einsum('kl,km->lm',b_DD_S1[:,:],u_DD_S1[:,:]) +np.einsum('kl,km->lm',b_DD_S2[:,:],u_DD_S2[:,:]))

    beta_B = (np.einsum('kl,km->lm',b_E[:,:],u_D[:,:]) + np.einsum('kl,km->lm',b_D[:,:],u_E[:,:]) + np.einsum('kl,km->lm',b_EE_S1[:,:],u_DD_S1[:,:]) 
                        +  np.einsum('kl,km->lm',b_EE_S2[:,:],u_DD_S2[:,:]) + np.einsum('kl,km->lm',b_DD_S1[:,:],u_EE_S1[:,:]) +np.einsum('kl,km->lm',b_DD_S2[:,:],u_EE_S2[:,:]))

    gamma_B = (np.einsum('kl,km->lm',b_E[:,:],m_E[:,:]) + np.einsum('kl,km->lm',b_D[:,:],m_D[:,:]) + np.einsum('kl,km->lm',b_EE_S1[:,:],m_EE_S1[:,:]) 
                        +  np.einsum('kl,km->lm',b_EE_S2[:,:],m_EE_S2[:,:])  + np.einsum('kl,km->lm',b_DD_S1[:,:],m_DD_S1[:,:]) +np.einsum('kl,km->lm',b_DD_S2[:,:],m_DD_S2[:,:]))
    
    delta_B = -(np.einsum('kl,km->lm',b_E[:,:],m_D[:,:]) + np.einsum('kl,km->lm',b_D[:,:],m_E[:,:]) + np.einsum('kl,km->lm',b_EE_S1[:,:],m_DD_S1[:,:]) 
                        +  np.einsum('kl,km->lm',b_EE_S2[:,:],m_DD_S2[:,:])  + np.einsum('kl,km->lm',b_DD_S1[:,:],m_EE_S1[:,:]) +np.einsum('kl,km->lm',b_DD_S2[:,:],m_EE_S2[:,:]))

    E_B = np.block([[alpha_B , beta_B], [beta_B, alpha_B]])
    S_B = np.block([[gamma_B, delta_B], [-delta_B, -gamma_B]])

    #Test symmetry
    if np.allclose(E_B,E_B.T, rtol=1e-05, atol=1e-08) == False:
        print('Fatal error: E_B is NOT symmetric')
        sys.exit()

    if np.allclose(S_B,S_B.T, rtol=1e-05, atol=1e-08) == False:
        print('Fatal error: S_B is NOT symmetric')
        sys.exit()

    #I should also test rank??


    return E_B, S_B

def normalize_over_metric(C, metric_B):

    for i in range(k):
        C[:,i] =  C[:,i] /np.sqrt( np.abs((C[:,i].T) @ metric_B @ C[:,i]))
                     
    return C

#Compute the residuals 
def Residual(u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2, m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2, C_orig, C_pair, omegas_B):

    Number_of_paired_trialvectors = C_orig.shape[0]

    Length_of_a_trial_vector = 2*(len(u_E[:,0]) + len(u_EE_S1[:,0]) + len(u_EE_S2[:,0]))

    R_j = np.zeros((Length_of_a_trial_vector,k))
    R_j_norm = np.zeros((k))


    for j in range(k):
        
        E_R_j, D_R_j, S1_EE_R_j, S2_EE_R_j, S1_DD_R_j, S2_DD_R_j  = 0,0,0,0,0,0

        eig_val_j = omegas_B[j]

        for i in range(Number_of_paired_trialvectors): 
        
            E_R_j += (u_E[:,i] - eig_val_j * m_E[:,i]) * C_orig[i,j] + (u_D[:,i] + eig_val_j  * m_D[:,i]) * C_pair[i,j]

            D_R_j += (u_D[:,i] - eig_val_j * m_D[:,i]) * C_orig[i,j] + (u_E[:,i] + eig_val_j  * m_E[:,i]) * C_pair[i,j] 

            S1_EE_R_j += (u_EE_S1[:,i] - eig_val_j * m_EE_S1[:,i]) * C_orig[i,j] + (u_DD_S1[:,i] + eig_val_j * m_DD_S1[:,i]) * C_pair[i,j]

            S2_EE_R_j += (u_EE_S2[:,i] - eig_val_j * m_EE_S2[:,i]) * C_orig[i,j] + (u_DD_S2[:,i] + eig_val_j * m_DD_S2[:,i]) * C_pair[i,j]

            S1_DD_R_j += (u_DD_S1[:,i] - eig_val_j * m_DD_S1[:,i]) * C_orig[i,j] + (u_EE_S1[:,i] + eig_val_j * m_EE_S1[:,i]) * C_pair[i,j]

            S2_DD_R_j += (u_DD_S2[:,i] - eig_val_j * m_DD_S2[:,i]) * C_orig[i,j] + (u_EE_S2[:,i] + eig_val_j * m_EE_S2[:,i]) * C_pair[i,j]


        R_j[:,j] = np.concatenate((E_R_j, D_R_j, S1_EE_R_j, S2_EE_R_j, S1_DD_R_j, S2_DD_R_j))
    

        R_j_norm[j] = np.linalg.norm(R_j[:,j])



    return R_j, R_j_norm


def new_trial_vectors(omegas_B,R_j):

    b_E_new = np.zeros((Length_of_A))             ;   b_D_new = np.zeros((Length_of_A))
    b_EE_S1_new = np.zeros((Length_of_D_S1))      ;   b_EE_S2_new = np.zeros((Length_of_D_S2))
    b_DD_S1_new = np.zeros((Length_of_D_S1))      ;   b_DD_S2_new = np.zeros((Length_of_D_S2))

    index_ai = 0
    index_aibj_S1 = 0
    index_aibj_S2 = 0
    for i in range(occ): 
        for a in range(vir):

            A_aiai = A(i,a,i,a, A_order)
            ph_S_aiai = ph_S(i,a,i,a)

            b_E_new[index_ai]  = R_j[index_ai] / (A_aiai - omegas_B * ph_S_aiai)
            b_D_new[index_ai] =  R_j[index_ai+Length_of_A] / (A_aiai + omegas_B * ph_S_aiai )
            
            for j in range(i, occ):
                for b in range(a, vir):
                    D_S1_aibj = D_func_S1S2(i,a,j,b)
                    
                    b_EE_S1_new[index_aibj_S1]  = R_j[index_aibj_S1+Length_of_A*2] / (D_S1_aibj - omegas_B)
                    b_DD_S1_new[index_aibj_S1]  = R_j[index_aibj_S1+Length_of_A*2+Length_of_D_S1+Length_of_D_S2] / (D_S1_aibj + omegas_B)

                    index_aibj_S1 += 1

                    if (a!=b)*(i!=j) == 1:
                        D_S2_aibj = D_func_S1S2(i,a,j,b)

                        b_EE_S2_new[index_aibj_S2]  = R_j[index_aibj_S2+Length_of_A*2+Length_of_D_S1] / (D_S2_aibj - omegas_B)
                        b_DD_S2_new[index_aibj_S2]  = R_j[index_aibj_S2+Length_of_A*2+Length_of_D_S1*2+Length_of_D_S2] / (D_S2_aibj + omegas_B)

                        index_aibj_S2 += 1

            index_ai += 1
       
            
    b_new = np.concatenate((b_E_new, b_D_new,b_EE_S1_new,b_EE_S2_new,b_DD_S1_new,b_DD_S2_new), axis=0)

    return b_new



def Gram_Schmidt(b_E, b_D,b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2, b_new, GS_tol):
 
    Number_of_pairs_of_trial_vectors = b_E.shape[1]

    b_k = np.concatenate([b_E, b_D,b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2], axis = 0)
    b_k_reverse = np.concatenate([b_D, b_E,b_DD_S1,b_DD_S2,b_EE_S1,b_EE_S2], axis = 0)

    b_new_gs = np.zeros((1))

    #First normalize vector
    b_new = b_new / np.linalg.norm(b_new)

    I = 0
    R = 0

    for i in range(Number_of_pairs_of_trial_vectors):
        I += np.dot(b_k[:,i], b_new) * b_k[:,i]
        R +=np.dot(b_k_reverse[:,i], b_new) * b_k_reverse[:,i]


    b_orthogonalized = b_new - I - R

    if np.linalg.norm(b_orthogonalized) > GS_tol:
        b_new_gs  = b_orthogonalized / np.linalg.norm(b_orthogonalized)

    return b_new_gs


def Lowdin_ortho(b_new_gs):

    e_start     = 0                                                     ; e_end     =     Length_of_A
    d_start     =     Length_of_A                                       ; d_end     = 2 * Length_of_A
    S1_ee_start = 2 * Length_of_A                                       ; S1_ee_end = 2 * Length_of_A +   Length_of_D_S1 
    S2_ee_start = 2 * Length_of_A +   Length_of_D_S1                    ; S2_ee_end = 2 * Length_of_A +   Length_of_D_S1 +   Length_of_D_S2
    S1_dd_start = 2 * Length_of_A +   Length_of_D_S1 + Length_of_D_S2   ; S1_dd_end = 2 * Length_of_A + 2*Length_of_D_S1 +   Length_of_D_S2
    S2_dd_start = 2 * Length_of_A + 2*Length_of_D_S1 + Length_of_D_S2   ; S2_dd_end = 2 * Length_of_A + 2*Length_of_D_S1 + 2*Length_of_D_S2


    # Build reversed paired vector
    b_new_gs_reverse = np.zeros(b_new_gs.shape)

    b_new_gs_reverse[e_start:e_end],           b_new_gs_reverse[d_start:d_end]           = b_new_gs[d_start:d_end],           b_new_gs[e_start:e_end]
    b_new_gs_reverse[S1_ee_start : S1_ee_end], b_new_gs_reverse[S1_dd_start : S1_dd_end] = b_new_gs[S1_dd_start : S1_dd_end], b_new_gs[S1_ee_start : S1_ee_end]
    b_new_gs_reverse[S2_ee_start : S2_ee_end], b_new_gs_reverse[S2_dd_start : S2_dd_end] = b_new_gs[S2_dd_start : S2_dd_end], b_new_gs[S2_ee_start : S2_ee_end]

    a = np.dot(b_new_gs_reverse,b_new_gs)

    K = 1/ np.sqrt(1 + a)
    L = 1/ np.sqrt(1 - a)
    c_plus =  0.5 * (K + L)
    c_minus = 0.5 * (K - L)

    # Final symmetrized vector
    b_i = c_minus * b_new_gs_reverse + c_plus * b_new_gs
    #b_o = c_plus * b_new_gs_reverse + c_minus * b_new_gs


    # Final symmetrized vector into components
    b_E_new = b_i[e_start:e_end]
    b_D_new = b_i[d_start:d_end]
    b_EE_S1_new = b_i[S1_ee_start : S1_ee_end]
    b_EE_S2_new = b_i[S2_ee_start : S2_ee_end]
    b_DD_S1_new = b_i[S1_dd_start : S1_dd_end]
    b_DD_S2_new = b_i[S2_dd_start : S2_dd_end]

    return b_E_new, b_D_new, b_EE_S1_new, b_EE_S2_new, b_DD_S1_new, b_DD_S2_new


def get_eigenvectors(b_E, b_D, b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2, C_orig, C_pair,k):

    eigvect = np.zeros((2*(Length_of_A+Length_of_D_S1+Length_of_D_S2),k))

    for i in range(k):
        eigvector_e = 0 
        eigvector_d = 0
        eigvector_ee_S1 = 0
        eigvector_ee_S2 = 0
        eigvector_dd_S1 = 0
        eigvector_dd_S2 = 0

        for n in range(len(C_pair)):

            eigvector_e += b_E[:,n] * C_orig[n,i] + b_D[:,n] * C_pair[n,i]
            eigvector_d += b_D[:,n] * C_orig[n,i] + b_E[:,n] * C_pair[n,i]
            eigvector_ee_S1 += b_EE_S1[:,n] * C_orig[n,i] + b_DD_S1[:,n] * C_pair[n,i]
            eigvector_ee_S2 += b_EE_S2[:,n] * C_orig[n,i] + b_DD_S2[:,n] * C_pair[n,i]
            eigvector_dd_S1 += b_DD_S1[:,n] * C_orig[n,i] + b_EE_S1[:,n] * C_pair[n,i]
            eigvector_dd_S2 += b_DD_S2[:,n] * C_orig[n,i] + b_EE_S2[:,n] * C_pair[n,i]
    
        eigvect[:,i] = np.concatenate((eigvector_e,eigvector_d,eigvector_ee_S1,eigvector_ee_S2,eigvector_dd_S1,eigvector_dd_S2),axis=0)
    return eigvect


def update_um(x_E, x_D, x_EE_S1, x_EE_S2, x_DD_S1, x_DD_S2, x_E_new, x_D_new, x_EE_S1_new, x_EE_S2_new, x_DD_S1_new, x_DD_S2_new):

    x_E = np.concatenate((x_E,x_E_new),axis=1)
    x_D = np.concatenate((x_D,x_D_new),axis=1)
    x_EE_S1 = np.concatenate((x_EE_S1,x_EE_S1_new),axis=1)
    x_EE_S2 = np.concatenate((x_EE_S2,x_EE_S2_new),axis=1)
    x_DD_S1 = np.concatenate((x_DD_S1,x_DD_S1_new),axis=1)
    x_DD_S2 = np.concatenate((x_DD_S2,x_DD_S2_new),axis=1)   

    return x_E, x_D, x_EE_S1, x_EE_S2, x_DD_S1, x_DD_S2

def update_b(b_E, b_D, b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2,b_E_new, b_D_new, b_EE_S1_new, b_EE_S2_new, b_DD_S1_new, b_DD_S2_new):
    b_E = np.concatenate((b_E,b_E_new[:,np.newaxis]),axis=1)
    b_D = np.concatenate((b_D,b_D_new[:,np.newaxis]),axis=1)
    b_EE_S1 = np.concatenate((b_EE_S1,b_EE_S1_new[:,np.newaxis]),axis=1)
    b_EE_S2 = np.concatenate((b_EE_S2,b_EE_S2_new[:,np.newaxis]),axis=1)
    b_DD_S1 = np.concatenate((b_DD_S1,b_DD_S1_new[:,np.newaxis]),axis=1)
    b_DD_S2 = np.concatenate((b_DD_S2,b_DD_S2_new[:,np.newaxis]),axis=1)   

    return b_E, b_D, b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2

#Function to get the k lowest positive eigenvalues and their corresponding eigenvectors
def get_k_lowest_positive_eigenvalues_and_vectors(omegas_B, C, k):

    idx = np.arange(omegas_B.size)[omegas_B > 0]
    idx_val_sorted = idx[np.argsort(omegas_B[idx])][:k]
    omegas_B = omegas_B[idx_val_sorted]   
    C = C[:,idx_val_sorted]

    return omegas_B, C