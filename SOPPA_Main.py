from pyscf import scf, ao2mo
import sys
from collections import namedtuple
import numpy as np
import scipy as sc

import Molecules as Molecule                                         #Molecules are defined in this module
from eigensolver_data import set_globals, set_AmpsAndDens_global     #This module manages global variables used throughout the code
from Amplitudes import *                                             #All functions for amplitudes and density matrices are in this module
from Testmodule import *                                             #Functions to ensure numerical accuracy, stability, and correctness of the eigensolver


#------------------------------------------------------------
#Key Variables and Parameters
#------------------------------------------------------------


mol = Molecule.COTest
k = 21                             # Number of excitation energies
A_order = 2                        # 2 = SOPPA ; 3 = SOPPA augmented with A(3)

R_tol = 10**-10                    # Residual tolerance for Eigensolver
GS_tol = 10**-3                    # Gram-Schmidt tolerance for vectors orthogonal to basis
LÖW_tol =  10**-3                  # Löwdin tolerance

frozen = 0                         # Freezing lowest (occupied) lying orbitals

max_cycles = 40                    # Sets the maximum number of iterations/cycles that should be performed for eigensolver

paired_tol = 10**-8                # Absolute tolerance for comparing eigenvalue pairs

#au to eV factors 
AU_to_eV_factor_Dalton = (((1.05457168*10**-34) **2)/((0.5291772108*10**-10)*(0.5291772108*10**-10)*(9.1093826*10**-31)))/(1.60217653*10**-19 )
#Written as in Dalton
AU_to_eV_factor = 27.211386245981  # Ref. NIST


#------------------------------------------------------------
# Hartree Fock Calculation
#------------------------------------------------------------
# This section sets up and runs a Hartree-Fock calculation on the molecule. 
# It computes the molecular orbital (MO) energies and two-electron integrals (ri)
#as well as the number of occupied and virtual orbitals (occ, vir)

hf_Conv_tol = 1e-10 
Hartree_fock = scf.HF(mol)
Hartree_fock.conv_tol = hf_Conv_tol
hf = Hartree_fock.run()
# MO energies:
mo_e = hf.mo_energy
# Two-electron integrals:
ri = ao2mo.kernel(mol.intor('int2e'),hf.mo_coeff)
#The number of the occupied and virtual orbitals are:
All_orbitals = int(len(hf.mo_occ))
occ = int(np.sum(hf.mo_occ)/2)
vir = All_orbitals - occ


print('Occupied orbitals:',occ,'. Virtual orbitals:',vir)

#------------------------------------------------------------
#Freezing orbitals, if requested by variable frozen
#------------------------------------------------------------
#If specified, the lowest occupied orbitals are removed from the list of orbitals considered in the calculation. 
print('Freezing', frozen, 'orbitals')
mo_e = mo_e[frozen:]
ri = ri[frozen:,frozen:,frozen:,frozen:]
occ = occ - frozen



array_eig = np.zeros((20,21))
array_Res = np.zeros((20,21))

#------------------------------------------------------------
#Global Variable Setup
#------------------------------------------------------------

#The length of the A, D_S1 and D_S2 matrices are needed in the functions for the eigensolver
Length_of_A = occ * vir
Length_of_D_S1 =int(vir*(vir+1)/2*((occ)*((occ)+1)/2))
Length_of_D_S2 = int(vir*(vir-1)/2*((occ)*((occ)-1)/2))

#The variables, needed in the MatrixFunctions.py, EigenSolverFunctions.py modules, 
# are packed into 2 named tuple and passed to either set_globals or set_AmpsAndDens_global. These functions are defined in 
# the eigensolver_data.py module and #ensures that these variables are 
# accessible globally in the modules that import the eigensolver_data.py module.

#The first tuple contain data that is needed in one or both of the modules: MatrixFunctions.py and EigenSolverEquations.py.
vars_tuple = namedtuple('vars_tuple', ['occ', 'vir', 'mo_e', 'ri', 'Length_of_A', 'Length_of_D_S1','Length_of_D_S2','k', 'A_order'])
vars_tuple = vars_tuple(occ, vir, mo_e, ri, Length_of_A, Length_of_D_S1, Length_of_D_S2, k, A_order)
set_globals(vars_tuple)

#The other tuple contains the amplitude and density matrices.
AmpsAndDens = compute_all_amplitudes_and_Density_matrices(occ,vir,ri,mo_e)
set_AmpsAndDens_global(AmpsAndDens)

#------------------------------------------------------------
#Importing the modules, that import eigensolver_data.py
#------------------------------------------------------------

from MatrixFunctions import *           #All functions for matrix elements are in this module
from EigenSolverFunctions import *      #All functions for the Davidson-type eigensolver is in here


#------------------------------------------------------------
# MP2 and MP3 corrections
#------------------------------------------------------------
E_MP2_correction = 0.5 * np.einsum('iajb,ijab->',ri[:occ,occ:,:occ,occ:],AmpsAndDens.m_amp)
print('MP2 E_corr =',E_MP2_correction)
print('MP2 Energy =',E_MP2_correction + hf.e_tot)

E_MP3_correction = 0.5 * np.einsum('iajb,ijab->',ri[:occ,occ:,:occ,occ:],AmpsAndDens.m_amp_2nd)
print('MP3 E_corr =',E_MP3_correction)
print('MP3 Energy =',E_MP3_correction + E_MP2_correction + hf.e_tot)



#---------------------------------------------------------
#Eigensolver
#---------------------------------------------------------

def eigensolver(k, R_tol, GS_tol, LÖW_tol):
        
    b_E, b_D, b_EE_S1, b_EE_S2, b_DD_S1, b_DD_S2 = initial_k_trialvec(k)

    u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2, m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2 = get_u_and_m(b_E, b_D, b_EE_S1, b_EE_S2, b_DD_S1, b_DD_S2, mo_e, count_new_trial_vector_pairs = k)

    # Iterate until convergence or maximum iterations reached
    for iterations in range(max_cycles):
        E_B, S_B = get_reduced_matrices(b_E, b_D, b_EE_S1, b_EE_S2, b_DD_S1, b_DD_S2, u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2, m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2)
        omegas_B, C =  sc.linalg.eig(E_B, S_B)
        

        #Check that eigenvalues are paired. Check that neither the eigenvalues or -vectors contain a 
        # large complex part relative to the real part

        omegas_B, C = Check_expected_behaviour_of_values(omegas_B, C )

        
        omegas_B, C = get_k_lowest_positive_eigenvalues_and_vectors(omegas_B, C, k)

        array_eig[iterations+1,:] = omegas_B

        C_normalized = normalize_over_metric(C, S_B)

        C_orig, C_pair = np.split(C_normalized, 2, axis=0)

        R_j, R_j_norm = Residual(u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2, m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2, C_orig, C_pair, omegas_B)

        array_Res[iterations+1,:] = R_j_norm


        count_new_trial_vector_pairs = 0
        converged_eigval= 0


        for i in range(k):
            if R_j_norm[i] < R_tol:             # Check for convergence
                converged_eigval += 1

            else:   #Compute new trial vectors
                b_new = new_trial_vectors(omegas_B[i],R_j[:,i])

                b_new_gs = Gram_Schmidt(b_E, b_D,b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2, b_new, GS_tol)
                if np.linalg.norm(b_new_gs) < 0.9:
                    print('Linear dependence GS. New trial vector discarded')
                    break

                b_E_new, b_D_new, b_EE_S1_new, b_EE_S2_new, b_DD_S1_new, b_DD_S2_new = Lowdin_ortho(b_new_gs)
                if 2*(np.dot(b_E_new,b_D_new)+np.dot(b_EE_S1_new,b_DD_S1_new)+np.dot(b_EE_S2_new,b_DD_S2_new)) > LÖW_tol:
                    print('Linear dependence Löwdin')
                    break
                
                count_new_trial_vector_pairs += 1

                b_E, b_D, b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2 = update_b(b_E, b_D, b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2,b_E_new, b_D_new, b_EE_S1_new, b_EE_S2_new, b_DD_S1_new, b_DD_S2_new)


        if converged_eigval == k:
            eig_values = omegas_B
            eig_vectors = get_eigenvectors(b_E, b_D, b_EE_S1,b_EE_S2,b_DD_S1,b_DD_S2, C_orig, C_pair,k)
            print('Converged!')
            print('2-Norm of residual',R_j_norm)
            break

        if count_new_trial_vector_pairs == 0:
            print('Fatal error: No trial vectors were added. Try to lower the GS tolerance')
            sys.exit()

        print('New basis vector pairs added to the basis:',count_new_trial_vector_pairs)

        #Update U and M
        u_E_new, u_D_new, u_EE_S1_new, u_EE_S2_new, u_DD_S1_new, u_DD_S2_new, m_E_new, m_D_new, m_EE_S1_new, m_EE_S2_new, m_DD_S1_new, m_DD_S2_new  = get_u_and_m(b_E, b_D, b_EE_S1, b_EE_S2, b_DD_S1, b_DD_S2, mo_e, count_new_trial_vector_pairs)

        u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2 = update_um(u_E, u_D, u_EE_S1, u_EE_S2, u_DD_S1, u_DD_S2, u_E_new, u_D_new, u_EE_S1_new, u_EE_S2_new, u_DD_S1_new, u_DD_S2_new)
        m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2 = update_um(m_E, m_D, m_EE_S1, m_EE_S2, m_DD_S1, m_DD_S2, m_E_new, m_D_new, m_EE_S1_new, m_EE_S2_new, m_DD_S1_new, m_DD_S2_new )

    return eig_values, eig_vectors, b_E.shape[1]


print('max k:',Length_of_A+Length_of_D_S1+Length_of_D_S2)

eigvalues, eig_vectors, trial_vect_pairs = eigensolver(k, R_tol, GS_tol, LÖW_tol)

np.set_printoptions(formatter={'float': '{: 0.12f}'.format})
print('Excitaiton energies (au)',  eigvalues )
print(' Excitaiton energies (eV)', eigvalues * AU_to_eV_factor)
print(' Number of pairs of trial vectors:', trial_vect_pairs)
#print(array)
#np.savetxt('array_eig_Be_TOPPA_11_05.txt', array_eig)
#np.savetxt('array_Res_Be_TOPPA_11_05.txt', array_Res)

#print(array-exact)