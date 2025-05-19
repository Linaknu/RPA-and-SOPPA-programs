import numpy as np
import sys

#Test module. This module contains functions to ensure numerical accuracy, stability, and correctness of the eigensolver

paired_tol = 10**-8             # Absolute tolerance for comparing eigenvalue pairs
complex_tol = 10**-6            # Relative tolerance for comparing complex and real components

def Check_expected_behaviour_of_values(omegas, eigenvectors):

    omegas = check_and_remove_complex_part(omegas,complex_tol)
    eigenvectors = check_and_remove_complex_parts_vector(eigenvectors, complex_tol)

    Test_paired_structure_of_eigenvalues(omegas, paired_tol)

    return omegas, eigenvectors



def Test_paired_structure_of_eigenvalues(omegas, paired_tol):
    positive = np.sort(omegas[omegas > 0])
    negative = np.sort(-omegas[omegas < 0])  # Flip sign for easier comparison

    # Compare sorted lists
    if len(positive) != len(negative):
        print('Unequal number of positive and negative eigenvalues')

    else:
        for omegas_pos, omegas_neg in zip(positive, negative):
            if abs(omegas_pos - omegas_neg) > paired_tol:
                print('Warning: eigenvalues are not paired')
                print( omegas_pos,'and', - omegas_neg)

    return

def check_and_remove_complex_part(omegas,complex_tol):

    # Check if any imaginary part exists larger than threshold
    if np.any(np.abs(np.imag(omegas)) > complex_tol * np.abs(np.real(omegas))):
        print('Warning: eigenvalues contain complex parts')

        #print the real and complex parts of the eigenvalue(s) containing a complex part over threshold:
        for i, value in enumerate(omegas):
            if np.abs(np.imag(value)) > complex_tol * np.abs(np.real(value)):
                print(i,': Real part = ',  np.real(value) ,' Imaginary part =', np.imag(value))

        # Remove complex parts by taking only the real part
    return np.real(omegas)



def check_and_remove_complex_parts_vector(eigenvectors, complex_tol):

    # Check if any imaginary components are larger than threshold exists
    for i, vec in enumerate(eigenvectors):
        if np.any(np.abs(np.imag(vec)) > complex_tol * np.abs(np.real(vec))):
            print('Warning: Eigenvector',i,'has complex components')

            #print the real and complex parts of the elements of the eigenvectors containing a complex part over threshold:
            for j, (r, im) in enumerate(zip(np.real(vec), np.imag(vec))):
                if np.abs(im) > complex_tol * np.abs(r):
                    print('Element', j,': real =', r, 'imag =', im)
    # Remove imaginary part
    return np.real(eigenvectors)



