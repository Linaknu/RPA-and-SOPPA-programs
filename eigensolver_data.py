#All the data, that is needed in both MatrixFunctions.py and EigenSolverEquations.py, lives here

# Define the global variables to be shared across modules

def set_globals(data):
    global occ, vir, mo_e, ri, Length_of_A, Length_of_D_S1, Length_of_D_S2, k, A_order
    occ = data.occ
    vir = data.vir
    mo_e = data.mo_e
    ri = data.ri
    Length_of_A = data.Length_of_A
    Length_of_D_S1 = data.Length_of_D_S1
    Length_of_D_S2 = data.Length_of_D_S2
    k = data.k
    A_order = data.A_order


#The amplitude and density matrices are unpacked into global variables
def set_AmpsAndDens_global(AmpsAndDens):
    global amp1, amp2, amp, m_amp, amp_sing, dens_secn_oo, dens_secn_ov, dens_secn_vv, amp_2nd, m_amp_2nd, dens_third_oo,dens_third_vv
    amp1 = AmpsAndDens.un_amp1
    amp2 = AmpsAndDens.un_amp2
    amp = AmpsAndDens.amp
    m_amp = AmpsAndDens.m_amp
    amp_sing = AmpsAndDens.amp_sing
    dens_secn_oo = AmpsAndDens.dens_secn_oo
    dens_secn_ov = AmpsAndDens.dens_secn_ov
    dens_secn_vv = AmpsAndDens.dens_secn_vv

    amp_2nd = AmpsAndDens.amp_2nd
    m_amp_2nd = AmpsAndDens.m_amp_2nd
    dens_third_oo = AmpsAndDens.dens_third_oo
    dens_third_vv =  AmpsAndDens.dens_third_vv


