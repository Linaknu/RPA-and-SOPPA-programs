import numpy as np
from collections import namedtuple



#Functions for amplitudes and densitiy matrices

AmpsAndDens = namedtuple('AmpsAndDens', ['un_amp1', 'un_amp2', 'amp', 'm_amp', 'amp_sing', 'dens_secn_oo', 
                                                               'dens_secn_ov', 'dens_secn_vv','amp_2nd', 'm_amp_2nd', 'dens_third_oo','dens_third_vv'])

def compute_all_amplitudes_and_Density_matrices(occ,vir,ri,mo_e):
    un_amp1 = make_amp1(occ,vir,ri,mo_e) #(51) AOSOPPA un_kappa_(1) first order, indexed ijab.
    un_amp2 = make_amp2(occ,vir,ri,mo_e) # (52) AOSOPPA un_kappa_(2) first order, indexed ijab. un_amp2 = amp2 
    amp = 0.5 * (un_amp1 + (1/np.sqrt(3))*un_amp2) #(53) AOSOPPA
    m_amp = make_mod_amp(amp)
    amp_sing = make_amp_sing(occ,vir,mo_e,ri,m_amp) #(70) AOSOPPA
    dens_secn_oo = make_dens_2nd_oo(amp,m_amp) #(75) AOSOPPA 
    dens_secn_ov = make_dens_2nd_ov(amp_sing) #(76) AOSOPPA
    dens_secn_vv = make_dens_2nd_vv(amp,m_amp) #(77) AOSOPPA

    amp_2nd = make_amp_2nd(occ,vir,ri,un_amp1,un_amp2,mo_e) #ref. # and Lilly's thesis
    m_amp_2nd = make_mod_amp(amp_2nd)                       #(54) AOSOPPA
    dens_third_oo = make_dens_3rd_oo(amp,amp_2nd,m_amp_2nd,m_amp) #Juliane
    dens_third_vv =  make_dens_3rd_vv(amp,amp_2nd,m_amp_2nd,m_amp) #Juliane

    return AmpsAndDens(un_amp1, un_amp2, amp, m_amp, amp_sing, dens_secn_oo, dens_secn_ov, dens_secn_vv, amp_2nd, m_amp_2nd, dens_third_oo, dens_third_vv)


# (51) AOSOPPA Unnormalized doubles first order amplitude (1) :
def make_amp1(occ,vir,ri,mo_e):
    temp = np.ndarray.flatten(np.add(np.column_stack(occ*[mo_e[:occ]]),mo_e[:occ]))
    temp2 = np.ndarray.flatten(np.subtract(np.column_stack(vir*[temp]),mo_e[occ:]))
    coeff_mat = np.subtract(np.column_stack(vir*[temp2]),mo_e[occ:])
    coeff_mat.shape = (occ,occ,vir,vir)

    return (np.einsum('aibj,ijab->ijab',ri[occ:,:occ,occ:,:occ],np.power(coeff_mat,-1))
            +np.einsum('ajbi,ijab->ijab',ri[occ:,:occ,occ:,:occ],np.power(coeff_mat,-1)))


def make_amp2(occ,vir,ri,mo_e):
    temp = np.ndarray.flatten(np.add(np.column_stack(occ*[mo_e[:occ]]),mo_e[:occ]))
    temp2 = np.ndarray.flatten(np.subtract(np.column_stack(vir*[temp]),mo_e[occ:]))
    coeff_mat = np.subtract(np.column_stack(vir*[temp2]),mo_e[occ:])
    coeff_mat.shape = (occ,occ,vir,vir)

    return (np.sqrt(3)*(np.einsum('aibj,ijab->ijab',ri[occ:,:occ,occ:,:occ],np.power(coeff_mat,-1))
                          -np.einsum('ajbi,ijab->ijab',ri[occ:,:occ,occ:,:occ],np.power(coeff_mat,-1))))

def make_mod_amp(amp): #(54) AOSOPPA
    return 4*amp -2*np.transpose(amp,(1,0,2,3))

def make_dens_2nd_oo(amp,m_amp): # (75) AOSOPPA
    return -np.einsum('ikab,jkab->ij',amp,m_amp)

def make_dens_2nd_vv(amp,m_amp):# (76) AOSOPPA
    return np.einsum('ijac,ijbc->ab',amp,m_amp)

def make_dens_2nd_ov(amp_sing): #(77) AOSOPPA
    return np.sqrt(2)*amp_sing

def make_amp_sing(occ,vir,mo_e,ri,m_amp): #(70) AOSOPPA
    return (1/np.sqrt(2))*np.einsum("ia, ia -> ia",np.power(np.subtract(np.column_stack(vir*[mo_e[:occ]]),mo_e[occ:]),-1),
                                    (np.einsum('abjc,ijbc->ia',ri[occ:,occ:,:occ,occ:],m_amp))-np.einsum('kijb,jkba->ia',ri[:occ,:occ,:occ,occ:],m_amp))


def make_amp_2nd(occ,vir,ri,amp1,amp2,mo_e):

    mat1 = ((0.5*np.einsum('kcbj,ikac->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2)
                           +0.5*np.einsum('kcbi,jkac->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2)
                           +0.5*np.einsum('kcaj,ikbc->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2)
                           +0.5*np.einsum('kcai,jkbc->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2))

                    -np.einsum('kjbc,ikac->ijab',ri[:occ,:occ,occ:,occ:],amp1)-np.einsum('kiac,jkbc->ijab',ri[:occ,:occ,occ:,occ:],amp1)
                    -(np.einsum('kjac,ikbc->ijab',ri[:occ,:occ,occ:,occ:],amp1)+np.einsum('kibc,jkac->ijab',ri[:occ,:occ:,occ:,occ:],amp1)))
    
    mat2 = ((3)**(0.5)*(0.5*np.einsum('kcbj,ikac->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2)
                           -0.5*np.einsum('kcbi,jkac->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2)
                           -0.5*np.einsum('kcaj,ikbc->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2)
                           +0.5*np.einsum('kcai,jkbc->ijab',ri[:occ,occ:,occ:,:occ],amp1+np.sqrt(3)*amp2))

                    -np.einsum('kjbc,ikac->ijab',ri[:occ,:occ,occ:,occ:],amp2)-np.einsum('kiac,jkbc->ijab',ri[:occ,:occ,occ:,occ:],amp2)
                    +(np.einsum('kjac,ikbc->ijab',ri[:occ,:occ,occ:,occ:],amp2)+np.einsum('kibc,jkac->ijab',ri[:occ,:occ:,occ:,occ:],amp2)))

    mat3 = np.einsum('acbd,ijcd->ijab',ri[occ:,occ:,occ:,occ:],amp1) ; mat4 = np.einsum('acbd,ijcd->ijab',ri[occ:,occ:,occ:,occ:],amp2)
    mat5 = np.einsum('kilj,klab->ijab',ri[:occ,:occ,:occ,:occ],amp1) ; mat6 = np.einsum('kilj,klab->ijab',ri[:occ,:occ,:occ,:occ],amp2)

    temp = np.ndarray.flatten(np.add(np.column_stack(occ*[mo_e[:occ]]),mo_e[:occ]))
    temp2 = np.ndarray.flatten(np.subtract(np.column_stack(vir*[temp]),mo_e[occ:]))
    coeff_mat = np.subtract(np.column_stack(vir*[temp2]),mo_e[occ:])
    coeff_mat.shape = (occ,occ,vir,vir)
    
    secn1 = (1/coeff_mat) * (mat1+mat3+mat5)
    secn2 = (1/coeff_mat) * (mat2+mat4+mat6)

    return 0.5*(secn1+(1/np.sqrt(3))*secn2)



# Functions for making a matrix for the 3rd order density contribution:
#Juliane (3.20)
#Indexed ij
def make_dens_3rd_oo(amp,amp_2nd,m_amp_2nd,m_amp):
    return -(np.einsum('ikcd,jkcd->ij',amp,m_amp_2nd) + np.einsum('ikcd,jkcd->ij',amp_2nd,m_amp))

#Juliane (3.31). Indexed ab
def make_dens_3rd_vv(amp,amp_2nd,m_amp_2nd,m_amp):
    return np.einsum('klac,klbc->ab',amp,m_amp_2nd) + np.einsum('klac,klbc->ab',amp_2nd,m_amp)


