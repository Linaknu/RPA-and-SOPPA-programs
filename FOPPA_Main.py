from pyscf import scf, ao2mo, cc
import sys
from collections import namedtuple
import numpy as np
import scipy as sc

import Molecules as Molecule                        #Molecules are defined in this module
from Testmodule import *                            #Functions to ensure numerical accuracy, stability, and correctness of the eigensolver


#------------------------------------------------------------
#Key Variables and Parameters
#------------------------------------------------------------

mol = Molecule.COTest
k = 10                           # Number of excitation energies

R_tol = 10**-4                      # Residual tolerance for Eigensolver
GS_tol = 10**-3                     # Gram-Schmidt tolerance for vectors orthogonal to basis
frozen = 1                          # Freezing lowest lying orbitals
max_cycles = 30                     # Sets the maximum number of iterations/cycles that should be performed


AU_to_eV_factor_DALTON = (((1.05457168*10**-34) **2)/((0.5291772108*10**-10)*(0.5291772108*10**-10)*(9.1093826*10**-31)))/(1.60217653*10**-19 )
#au to eV factor. Written as in Dalton
AU_to_eV_factor = 27.211386245981   # NIST

compute_CCSD = True                 #If True; compute CCSD ground state energy and k singlet excitation energies   


#------------------------------------------------------------
# Hartree Fock Calculation
#------------------------------------------------------------
# This section sets up and runs a Hartree-Fock calculation on the molecule. 
# It computes the molecular orbital (MO) energies and two-electron integrals (ri)
#as well as the number of occupied and virtual orbitals (occ, vir)

Conv_tol = 1e-10
Hartree_fock = scf.HF(mol)
Hartree_fock.conv_tol = Conv_tol
hf = Hartree_fock.run()
# MO energies:
mo_e = hf.mo_energy
# MO coefficients:
mo_c = hf.mo_coeff
# Two-electron integrals:
ri = ao2mo.kernel(mol.intor('int2e'),mo_c)

orbitals = int(len(hf.mo_occ))
occ = int(np.sum(hf.mo_occ)/2)
vir = orbitals - occ

print('Molecule symmetry group:', mol.topgroup)
print('Occupied orbitals:',occ,'. Virtual orbitals:',vir)


#------------------------------------------------------------
# CCSD Calculation
#------------------------------------------------------------
#CCSD ground state energy and excitatiion energies
if compute_CCSD == True:
    mycc = cc.CCSD(hf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc = mycc.run()
    #print(mycc.e_tot)
    #mycc.kernel()
    e_s, c_s = mycc.eomee_ccsd_singlet(nroots=k)
    print(e_s*AU_to_eV_factor )


#------------------------------------------------------------
#Freezing orbitals, if requested by variable frozen
#------------------------------------------------------------
#If specified, the lowest occupied orbitals are removed from the list of orbitals considered in the calculation. 
print('Freezing', frozen, 'orbitals')
mo_e = mo_e[frozen:]
ri = ri[frozen:,frozen:,frozen:,frozen:]
occ = occ - frozen


#------------------------------------------------------------
#Functions for Matrix Elements
#------------------------------------------------------------

#A(0) Eq. (100) in AOSOPPA
def A_0( i, a ,j, b):
    return (mo_e[a+occ]-mo_e[i]) * (i==j) * (a==b)

#A(1) Eq. (101) AOSOPPA
def A_1( i,a,j,b):
    return 2 * ri[a+occ,i,j,b+occ] - ri[a+occ,b+occ,j,i] 

#B(1) Eq. (109) AOSOPPA
def B_1(i,a,j,b):
    return (ri[a+occ,j,b+occ,i] - 2*ri[a+occ,i,b+occ,j])



#------------------------------------------------------------
#Functions for eigensolver
#------------------------------------------------------------

Length_of_A = occ*vir

#Functions for eigensolver

def initial_k_trialvec_comp(k):    

    A_zeroth_order_diag = np.zeros((Length_of_A))
    index_ai = 0
    for i in range(occ):
        for a in range(vir):
            A_zeroth_order_diag[index_ai] = A_0(i, a ,i,a)
            index_ai += 1


    idx = np.arange(Length_of_A)
    idx_val_sorted = idx[np.argsort(A_zeroth_order_diag[idx])][:k]

    b_E = np.zeros((Length_of_A,k))
    b_D = np.zeros((Length_of_A,k))

    b_E[idx_val_sorted,range(k)] = 1

    return b_E, b_D

def get_u_and_m(b_E,b_D):
    
    b_E_shape = b_E.shape
    Number_of_all_trial_vect_pairs = b_E.shape[1]
    
    u_E = np.zeros((b_E_shape))         ;           u_D = np.zeros((b_E_shape))
    m_E = b_E                           ;           m_D = -b_D

    for o in range(Number_of_all_trial_vect_pairs):
        I=0
        for i in range(occ):
            for a in range(vir):
                K = 0
                for j in range(occ):
                    for b in range(vir):                        
                        A_element = A_0( i, a ,j, b) + A_1( i,a,j,b)
                        B_element = B_1( i,a,j,b)

                        u_E[I,o] += A_element * b_E[K,o] + B_element * b_D[K,o]
                        u_D[I,o] += B_element * b_E[K,o] + A_element * b_D[K,o]
                        K+= 1
                I += 1

    return u_E, u_D, m_E, m_D
        
def get_reduced_matrices(b_E, b_D, u_E, u_D, m_E, m_D):
    
    n = b_E.shape[1]

    A_B = np.zeros((n, n))          ;           B_B = np.zeros((n, n))      
    Sigma_B = np.zeros((n, n))      ;           Delta_B = np.zeros((n, n))           

    for i in range(n):
        for j in range(n):
            A_B[i,j] = b_E[:,i].T @ u_E[:,j] + b_D[:,i].T @ u_D[:,j]
            B_B[i,j] = b_E[:,i].T @ u_D[:,j] + b_D[:,i].T @ u_E[:,j]
            Sigma_B[i, j] = b_E[:,i].T @ m_E[:,j] + b_D[:,i].T @ m_D[:,j]
            Delta_B[i, j] = -b_E[:,i].T @ m_D[:,j] - b_D[:,i].T @ m_E[:,j]
            

    E_B = np.block([[A_B, B_B], [B_B, A_B]])
    S_B = np.block([[Sigma_B, Delta_B], [-Delta_B, -Sigma_B]])

    return E_B, S_B

def normalize_over_metric(C, metric_B):

    for i in range(k):
        C[:,i] =  C[:,i] /np.sqrt( np.abs((C[:,i].T) @ metric_B @ C[:,i]))
                     
    return C

def Residual(u_E,u_D,m_E,m_D, C_orig, C_pair, omegas_B):
    
    Number_of_paired_trialvectors = C_orig.shape[0]

    R_j = np.zeros((Length_of_A*2,k))
    R_j_norm = np.zeros((k))


    for j in range(k):
        
        E_R_j, D_R_j = 0,0


        for i in range(Number_of_paired_trialvectors):  
        
            E_R_j += (u_E[:,i] - omegas_B[j] * m_E[:,i]) * C_orig[i,j] + (u_D[:,i] + omegas_B[j]  * m_D[:,i]) * C_pair[i,j]

            D_R_j += (u_D[:,i] - omegas_B[j] * m_D[:,i]) * C_orig[i,j] + (u_E[:,i] + omegas_B[j]  * m_E[:,i]) * C_pair[i,j] 
        
    
        R_j[:Length_of_A,j] = E_R_j     ;       R_j[Length_of_A:,j] = D_R_j


        R_j_norm[j] = np.linalg.norm(R_j[:,j]) 


    return R_j, R_j_norm

def new_trial_vectors(omegas_B,R_j):

    b_E_new = np.zeros((Length_of_A))
    b_D_new = np.zeros((Length_of_A))

    index_ai = 0
    for i in range(occ): 
        for a in range(vir):

            A_element = A_0( i, a ,i,a) + A_1( i,a,i,a)
            b_E_new[index_ai]  = R_j[index_ai] / (A_element - omegas_B)
            b_D_new[index_ai] = R_j[index_ai+Length_of_A] / (A_element + omegas_B)
            index_ai += 1

    return b_E_new, b_D_new

def Gram_Schmidt(b_E,b_D,b_E_new,b_D_new, GS_tol):

    Number_of_pairs_of_trial_vectors  = b_E.shape[1]
 
    b_k = np.concatenate([b_E,b_D], axis = 0)
    b_k_reverse = np.concatenate([b_D, b_E], axis = 0)

    b_new = np.concatenate((b_E_new,b_D_new), axis=0)
    b_new_gs = np.zeros((1))

    #First normalized vector
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

    n = len(b_new_gs)
    m = int(n/2)

    b_new_gs_reverse = np.zeros(b_new_gs.shape)
    b_new_gs_reverse[:m], b_new_gs_reverse[m:] = b_new_gs[m:], b_new_gs[:m]
    beta = np.dot(b_new_gs_reverse,b_new_gs)

    K = 1/ np.sqrt(1+beta)
    L = 1/ np.sqrt(1-beta)
    c_plus =  0.5 * (K + L)
    c_minus = 0.5 * (K - L)

    b_i = c_minus * b_new_gs_reverse + c_plus * b_new_gs
    #b_o = c_plus * b_new_gs_reverse + c_minus * b_new_gs


    B_E_new = b_i[:m]
    B_D_new = b_i[m:]

    return B_E_new ,B_D_new

def get_eigenvector(b_E,b_D, C_orig, C_pair):

    eigvect = np.zeros((2*Length_of_A,k))

    for i in range(k):

        eigvector_e = 0 
        eigvector_d = 0

        for n in range(len(C_pair)):

            eigvector_e += b_E[:,n] * C_orig[n,i] + b_D[:,n] * C_pair[n,i]
            eigvector_d += b_D[:,n] * C_orig[n,i] + b_E[:,n] * C_pair[n,i]

        eigvect[:,i] = np.concatenate((eigvector_e, eigvector_d),axis=0)

    return eigvect



def eigensolver(k, R_tol, GS_tol):
        
    b_E, b_D = initial_k_trialvec_comp(k)
    count_new_trial_vector_pairs = k

    for iterations in range(max_cycles):

        u_E, u_D, m_E, m_D = get_u_and_m(b_E,b_D)
        E_B, S_B = get_reduced_matrices(b_E, b_D, u_E, u_D, m_E, m_D)   
        omegas_B, C =  sc.linalg.eig(E_B, S_B)

        #Check that eigenvalues are paired. Check that neither the eigenvalues or -vectors contain a 
        # large complex part relative to the real part

        omegas_B, C = Check_expected_behaviour_of_values(omegas_B, C)


        idx = np.arange(omegas_B.size)[omegas_B > 0]
        idx_val_sorted = idx[np.argsort(omegas_B[idx])][:k]
        omegas_B = omegas_B[idx_val_sorted]   
        C = C[:,idx_val_sorted]


        C_normalized = normalize_over_metric(C, S_B)

        C_orig, C_pair = np.split(C_normalized, 2, axis=0)

        R_j, R_j_norm = Residual(u_E,u_D,m_E,m_D, C_orig, C_pair, omegas_B)  

        count_new_trial_vector_pairs = 0
        converged_eigval= 0

        
        for i in range(k):
            if R_j_norm[i] < R_tol:
                converged_eigval += 1

            else:
                b_E_new, b_D_new = new_trial_vectors(omegas_B[i],R_j[:,i])

                b_new_gs = Gram_Schmidt(b_E,b_D,b_E_new,b_D_new, GS_tol)
                if np.linalg.norm(b_new_gs) < 0.9:
                    print('Linear dependence GS')
                    break

                b_E_new ,b_D_new = Lowdin_ortho(b_new_gs)
                if np.dot(b_E_new,b_D_new) > 10**-3:
                    print('Linear dependence Löwdin')
                    break
            
                b_E = np.concatenate((b_E,b_E_new[:,np.newaxis]),axis=1)
                b_D = np.concatenate((b_D,b_D_new[:,np.newaxis]),axis=1)

                count_new_trial_vector_pairs += 1


        if converged_eigval == k:
            eig_values = omegas_B
            eig_vectors = get_eigenvector(b_E,b_D, C_orig, C_pair)
            print('Converged!')
            print('Norms',R_j_norm)
            break

        if count_new_trial_vector_pairs == 0:
            print('Warning. No trial vectors were added. Try to lower the GS tolerance')
            sys.exit()

        print('New basis vector pairs',count_new_trial_vector_pairs)


    return eig_values, eig_vectors,  b_E.shape[1]


print('max k:',Length_of_A)
eig_values, eig_vectors, trial_vect_pairs = eigensolver(k, R_tol, GS_tol)

np.set_printoptions(formatter={'float': '{: 0.10f}'.format})
print('Excitaiton energies (au)',  eig_values )

print(' Excitaiton energies (eV)', eig_values * AU_to_eV_factor)

print(' Number of pairs of trial vectors:', trial_vect_pairs)

