import numpy as np
from eigensolver_data import *

#All matrix elements are defined in this module


# The A matrix

#A(0) Eq. (100) in AOSOPPA
def A_0( i, a ,j, b):
    return (mo_e[a+occ]-mo_e[i]) * (i==j) * (a==b)

#A(1) Eq. (101) AOSOPPA
def A_1( i,a,j,b):
    return 2 * ri[a+occ,i,j,b+occ] - ri[a+occ,b+occ,j,i] 


#A(2) Eq. (102) in AOSOPPA
def A_2(i,a,j,b):

    a=occ+a
    b=occ+b

    sums1 = np.einsum('ckd,kcd->', ri[j,occ:,:occ,occ:],m_amp[i,:,:,:])
    sums2 = np.einsum('klc,klc->', ri[:occ,b,:occ,occ:],m_amp[:,:,a-occ,:])
    
    secn_ord = (0.5*(mo_e[b]-mo_e[j])*((a==b)*dens_secn_oo[i,j]-(i==j)*dens_secn_vv[b-occ,a-occ])-0.5*(a==b)*sums1-0.5*(i==j)*sums2)

    return secn_ord

#A'(3)(1) (2.158) Kasper
def A_3_1_1_Kasper( i, a, j ,b):
    s1, s2, s3, s4 = 0,0,0,0

    if a==b:
        s1_ = - ri[i,j,occ:,occ:] + 0.5 * ri[j,occ:,occ:,i]
        s1 =  np.einsum('cd,cd->',s1_ ,dens_secn_vv) 

        s2_ = - ri[i,j,:occ,:occ] + 0.5 * ri[j,:occ,:occ,i]
        s2 =  np.einsum('kl,kl->',s2_ ,dens_secn_oo) 

    if i==j:
        s3_ =  ri[a+occ,b+occ,occ:,occ:] - 0.5 * ri[a+occ,occ:,occ:,b+occ]
        s3 =  np.einsum('cd,cd->',s3_ ,dens_secn_vv) 

        s4_ =  ri[a+occ,b+occ,:occ,:occ] - 0.5 * ri[a+occ,:occ,:occ,b+occ]
        s4 =  np.einsum('kl,kl->',s4_ ,dens_secn_oo) 

    s5_ = - ri[a+occ,i,j,occ:] + 0.5 * ri[j,i,occ:,a+occ]
    s5 =  np.einsum('c,c->',s5_ ,dens_secn_vv[b,:]) 

    s6_ = - ri[b+occ,j,i,occ:] + 0.5 * ri[j,i,occ:,b+occ]
    s6 =  np.einsum('c,c->',s6_ ,dens_secn_vv[a,:])

    s7_ =  ri[a+occ,i,b+occ,:occ] - 0.5 * ri[a+occ,b+occ,i,:occ]
    s7 =  np.einsum('l,l->',s7_ ,dens_secn_oo[j,:]) 

    s8_ =  ri[b+occ,j,a+occ,:occ] - 0.5 * ri[a+occ,b+occ,j,:occ]
    s8 =  np.einsum('l,l->',s8_ ,dens_secn_oo[i,:]) 

    return s1 +s2 + s3 +s4 + s5 + s6 + s7 + s8

#A'(3)(2) (2.159) Kasper

def A_3_1_2_Kasper(i,a,j,b):

    a = a + occ
    b = b + occ

    i_su1 = np.einsum('mnd,mnc->cd',amp[:,:,:,b-occ],m_amp[:,:,a-occ,:]) + np.einsum('mnd,mnc->cd',amp[:,:,b-occ,:],m_amp[:,:,:,a-occ])
    i_su2 = np.einsum('mnc,mnd->cd',amp[:,:,:,b-occ],m_amp[:,:,:,a-occ]) + np.einsum('mnc,mnd->cd',amp[:,:,b-occ,:],m_amp[:,:,a-occ,:])
    i_su3 = np.einsum('med,mce->cd',m_amp[:,j,:,:],m_amp[i,:,:,:])
    i_su4 = np.einsum('mec,med->cd',amp[:,j,:,:],m_amp[:,i,:,:]) + np.einsum('mce,mde->cd',amp[:,j,:,:],m_amp[:,i,:,:]) 

    su1 = (np.einsum('cd,cd->',i_su1 ,ri[j,occ:,occ:,i]) 
    + np.einsum('cd,cd->', i_su2, ri[j,i,occ:,occ:])
    + np.einsum('cd,cd->',i_su3,ri[a,occ:,occ:,b])
    -2*np.einsum('cd,cd->',i_su4,ri[a,b,occ:,occ:]))

    i_su5 = np.einsum('kme,mle->kl',m_amp[:,:,b-occ,:],m_amp[:,:,:,a-occ])
    i_su6 = np.einsum('mle,mke->kl',amp[:,:,:,b-occ],m_amp[:,:,:,a-occ]) + np.einsum('mle,mke->kl',amp[:,:,b-occ,:],m_amp[:,:,a-occ,:])
    i_su7 = np.einsum('kef,lef->kl',amp[j,:,:,:],m_amp[:,i,:,:]) +  np.einsum('kef,lef->kl',amp[:,j,:,:],m_amp[i,:,:,:])
    i_su8 = np.einsum('lef,kef->kl',amp[:,j,:,:],m_amp[:,i,:,:]) + np.einsum('lef,kef->kl',amp[j,:,:,:],m_amp[i,:,:,:])

    su2 = (np.einsum('kl,kl->',i_su5,ri[j,:occ,:occ,i]) 
    -2* np.einsum('kl,kl->',i_su6,ri[j,i,:occ,:occ]) 
    + np.einsum('kl,kl->',i_su7,ri[a,:occ,:occ,b]) 
    + np.einsum('kl,kl->',i_su8,ri[a,b,:occ,:occ]))

    i_su9 = np.einsum('mke,mce->kc',m_amp[:,:,:,b-occ],m_amp[i,:,:,:])
    i_su10 = np.einsum('mec,mke->kc',amp[:,j,:,:],m_amp[:,:,:,a-occ]) + np.einsum('mce,mke->kc',amp[:,j,:,:],m_amp[:,:,a-occ,:])
    i_su11 = np.einsum('mke,mec->kc',amp[:,:,:,b-occ],m_amp[:,i,:,:]) + np.einsum('mke,mce->kc',amp[:,:,b-occ,:],m_amp[:,i,:,:])
    i_su12 = np.einsum('mec,kme->kc',m_amp[:,j,:,:],m_amp[:,:,a-occ,:])

    su3 = - np.einsum('kc,ck->',i_su9,ri[a,occ:,j,:occ]) + 2* np.einsum('kc,ck->',i_su10, ri[occ:,i,:occ,b])
    su4 = -2* np.einsum('kc,kc->',i_su11,ri[a,:occ,j,occ:]) - np.einsum('kc,kc->',i_su12,ri[:occ,i,occ:,b])


    sums1,sums2,sums3,sums4,sums5,sums6,sums7,sums8 = (0,0,0,0,0,0,0,0)

    if (i==j):
        sums1_ = 0.5 * (np.einsum('noec,nod->cde',amp,m_amp[:,:,a-occ,:]) + np.einsum('noce,nod->cde',amp,m_amp[:,:,:,a-occ]) 
        + np.einsum('node,noc->cde',amp,m_amp[:,:,:,a-occ]) + np.einsum('noed,noc->cde',amp,m_amp[:,:,a-occ,:]))
        sums1 = np.einsum('cde,cde->',sums1_,ri[occ:,occ:,occ:,b])

        sums3_ = (0.5 * np.einsum('kncf,nlf->ckl',m_amp,m_amp[:,:,:,a-occ]) - np.einsum('nlfc,nkf->ckl',amp,m_amp[:,:,:,a-occ])
        -np.einsum('nlcf,nkf->ckl',amp,m_amp[:,:,a-occ,:]))
        sums3 = np.einsum('ckl,ckl->',sums3_,ri[occ:,:occ,:occ,b])

        sums6_ = -(np.einsum('nlfc,nkf->klc',amp,m_amp[:,:,:,a-occ]) + np.einsum('nlcf,nkf->klc',amp,m_amp[:,:,a-occ,:]))
        sums6 = np.einsum('klc,klc->',sums6_,ri[:occ,:occ,occ:,b])

        sums8_ = np.einsum('nlfc,knf->kcl',m_amp,m_amp[:,:,a-occ,:])
        sums8 = np.einsum('kcl,kcl->',sums8_,ri[:occ,b,occ:,:occ])

    if (a==b):
        sums4_ = (0.5 * np.einsum('nkfd,ncf->cdk',m_amp,m_amp[i,:,:,:])- np.einsum('nkfc,nfd->cdk',amp,m_amp[:,i,:,:]) 
        - np.einsum('nkcf,ndf->cdk',amp,m_amp[:,i,:,:]))
        sums4 = np.einsum('cdk,cdk->',sums4_,ri[occ:,occ:,j,:occ])

        sums5_ = (np.einsum('nkfc,nfd->cdk',amp,m_amp[:,i,:,:]) + np.einsum('nkcf,ndf->cdk',amp,m_amp[:,i,:,:]))
        sums5 = np.einsum('cdk,kcd->',sums5_,ri[j,:occ,occ:,occ:])

        sums7_ = np.einsum('kncf,nfd->cd',m_amp,m_amp[:,i,:,:]) 
        sums7 = np.einsum('cd,cdd->',sums7_,ri[occ:,occ:,j,occ:])

        sums2_ =0.5*( np.einsum('mlfg,kfg->klm',amp,m_amp[i,:,:,:]) + np.einsum('kmfg,lfg->klm',amp,m_amp[:,i,:,:])
                + np.einsum('lmfg,kfg->klm',amp,m_amp[:,i,:,:]) +np.einsum('mkfg,lfg->klm',amp,m_amp[i,:,:,:]))
        sums2 = np.einsum('klm,klm->',sums2_,ri[:occ,:occ,j,:occ])

    output = 0.25*(su1+su2+su3+su4-sums1-sums2-sums3-sums4-sums5-sums6)-0.125*(sums7+sums8)
    return output



def A_3_1_Geertsen(i,a,j,b): #un_amp

    sum_ab,sum_ij = 0,0
    
    if a == b:
        s1_ri = ri[:occ,i,j,:occ] - 2 * ri[j,i,:occ,:occ] 
        s1_k = np.einsum('klcd,mlcd->km',amp1,amp1) +  np.einsum('klcd,mlcd->km',amp2,amp2)
        s1 = np.einsum('km,km->',s1_ri,s1_k)

        s2_ri = 2 * ri[j,i,occ:,occ:] - ri[occ:,i,j,occ:] 
        s2_k = np.einsum('klcd,kled->ce',amp1,amp1) +  np.einsum('klcd,kled->ce',amp2,amp2)
        s2 = np.einsum('ec,ce->',s2_ri,s2_k)

        s3_k = np.einsum('mcd,klcd -> lmk',amp1[:,i,:,:],amp1) +  np.einsum('mcd,klcd -> lmk',amp2[:,i,:,:],amp2)
        s3 = np.einsum('lmk,lmk->',ri[j,:occ,:occ,:occ],s3_k)

        s4_k = np.einsum('ked,klcd -> lce',amp1[:,i,:,:],amp1) +  np.einsum('ked,klcd -> lce',amp2[:,i,:,:],amp2)
        s4 = np.einsum('lce,lce->',ri[j,:occ,occ:,occ:],s4_k)

        s5_k = np.einsum('led,klcd->eck',amp1[i,:,:,:]+np.sqrt(3)*amp2[i,:,:,:], amp1+np.sqrt(3)*amp2)
        s5 = np.einsum('eck,eck->',ri[j,occ:,occ:,:occ],s5_k)

        sum_ab = 0.25 * (s1+s2+s3+s5) - 0.5 * s4 


    if i == j:
        s1_ri = ri[occ:,b+occ,a+occ,occ:] - 2 * ri[a+occ,b+occ,occ:,occ:] 
        s1_k = np.einsum('klcd,kled->ce',amp1,amp1) +  np.einsum('klcd,kled->ce',amp2,amp2)
        s1 = np.einsum('ec,ce->',s1_ri,s1_k)

        s2_ri = 2 * ri[a+occ,b+occ,:occ,:occ] - ri[:occ,b+occ,a+occ,:occ] 
        s2_k = np.einsum('klcd,mlcd->km',amp1,amp1) +  np.einsum('klcd,mlcd->km',amp2,amp2)
        s2 = np.einsum('km,km->',s2_ri,s2_k)

        s3_k = np.einsum('kle,klcd->dce',amp1[:,:,:,a],amp1) +  np.einsum('kle,klcd->dce',amp2[:,:,:,a],amp2)
        s3 = np.einsum('dce,dce->',ri[occ:,b+occ,occ:,occ:],s3_k)

        s4_k = np.einsum('mlc,klcd->dmk',amp1[:,:,:,a],amp1) +  np.einsum('mlc,klcd->dmk',amp2[:,:,:,a],amp2)
        s4 = np.einsum('dmk,dmk->',ri[occ:,b+occ,:occ,:occ],s4_k)

        s5_k = np.einsum('mld,klcd->mck',amp1[:,:,a,:]+np.sqrt(3)*amp2[:,:,a,:], amp1+np.sqrt(3)*amp2)
        s5 = np.einsum('mck,mck->',ri[:occ,b+occ,occ:,:occ],s5_k)

        sum_ij = 0.25 * (s1+s2+s3+s5) - 0.5 * s4        

    
    #Page 2

    s1_ri = ri[j,i,occ:,occ:] + ri[occ:,i,j,occ:] 
    s1_k = np.einsum('klc,kld->dc',amp1[:,:,a,:],amp1[:,:,b,:]) 
    s1 = np.einsum('dc,dc->',s1_ri,s1_k)

    s2_ri = ri[j,i,occ:,occ:] - ri[occ:,i,j,occ:] 
    s2_k = np.einsum('klc,kld->dc',amp2[:,:,a,:],amp2[:,:,b,:]) 
    s2 = np.einsum('dc,dc->',s2_ri,s2_k)

    s3_k = np.einsum('klc,mlc->km',amp1[:,:,a,:],amp1[:,:,b,:]) +  np.einsum('kld,mld->km',amp2[:,:,a,:],amp2[:,:,b,:])
    s3 = np.einsum('km,km->',ri[j,i,:occ,:occ],s3_k)

    s4_k = np.einsum('klc,mlc->mk',amp1[:,:,a,:]+np.sqrt(3)*amp2[:,:,a,:], amp1[:,:,b,:]+np.sqrt(3)*amp2[:,:,b,:]) # c changed to b!!!!!!
    s4 = np.einsum('mk,mk->',ri[j,:occ,:occ,i],s4_k)

    s5_ri = 0.5 * ri[j,occ:,a+occ,i] - 0.25 * ri[j,i,a+occ,occ:] 
    s5_k = np.einsum('klcd,klc->d',amp1,amp1[:,:,:,b]) +  np.einsum('klcd,klc->d',amp2,amp2[:,:,:,b])
    s5 = np.einsum('d,d->',s5_ri,s5_k)

    s6_ri = 0.5 * ri[a+occ,:occ,j,b+occ] - 0.25 * ri[a+occ,b+occ,j,:occ] 
    s6_k = np.einsum('klcd,lcd->k',amp1,amp1[i,:,:,:]) +  np.einsum('klcd,lcd->k',amp2,amp2[i,:,:,:])
    s6 = np.einsum('k,k->',s6_ri,s6_k)

    sum_2 = 0.25 * (s1+s2+s4) - 0.5*s3 -s5 - s6

    #Page 3
    
    s1_k = np.einsum('kcd,lkc->ld',amp1[:,i,:,:]+np.sqrt(3)*amp2[:,i,:,:], amp1[:,:,b,:]+np.sqrt(3)*amp2[:,:,b,:])
    s1 = np.einsum('ld,ld->',ri[j,:occ,a+occ,occ:],s1_k)

    s2_k = np.einsum('kcd,lkc->dl',amp1[:,i,:,:],amp1[:,:,b,:]) +   np.einsum('kcd,lkc->dl',amp2[:,i,:,:],amp2[:,:,b,:])
    s2 = np.einsum('dl,dl->',ri[j,occ:,a+occ,:occ],s2_k)

    s3_ri = ri[a+occ,b+occ,:occ,:occ] + ri[:occ,b+occ,a+occ,:occ] 
    s3_k = np.einsum('lcd,kcd->lk',amp1[i,:,:,:],amp1[j,:,:,:]) 
    s3 = np.einsum('lk,lk->',s3_ri,s3_k)

    s4_ri = ri[a+occ,b+occ,:occ,:occ] - ri[:occ,b+occ,a+occ,:occ] 
    s4_k = np.einsum('lcd,kcd->lk',amp2[i,:,:,:],amp2[j,:,:,:]) 
    s4 = np.einsum('lk,lk->',s4_ri,s4_k)

    s5_k = np.einsum('kcd,ked->ec',amp1[i,:,:,:],amp1[j,:,:,:]) +   np.einsum('kcd,ked->ec',amp2[i,:,:,:],amp2[j,:,:,:])
    s5 = np.einsum('ec,ec->',ri[a+occ,b+occ,occ:,occ:],s5_k)

    s6_k = np.einsum('kcd,ked->ec',amp1[i,:,:,:]+np.sqrt(3)*amp2[i,:,:,:], amp1[j,:,:,:]+np.sqrt(3)*amp2[j,:,:,:]) # CHANGED!!!
    s6 = np.einsum('ec,ec->',ri[occ:,b+occ,a+occ,occ:],s6_k)

    s7_ri = 0.5*ri[:occ,b+occ,a+occ,i] - 0.25 * ri[a+occ,b+occ,:occ,i] 
    s7_k = np.einsum('klcd,kcd->l',amp1,amp1[:,j,:,:]) + np.einsum('klcd,kcd->l',amp2,amp2[:,j,:,:]) #minus
    s7 = np.einsum('l,l->',s7_ri,s7_k)

    s8_ri = 0.5*ri[occ:,i,j,b+occ] - 0.25 * ri[j,i,occ:,b+occ] 
    s8_k = np.einsum('klcd,kld->c',amp1,amp1[:,:,a,:]) + np.einsum('klcd,kld->c',amp2,amp2[:,:,a,:])
    s8 = np.einsum('c,c->',s8_ri,s8_k)

    sum_3 = 0.25 * (-s1+s3+s4+s6) + 0.5 * (s2 - s5) - s7 - s8

    # Page 4
    s1_k = np.einsum('klc,kec->el',amp1[:,:,:,a]+np.sqrt(3)*amp2[:,:,:,a], amp1[j,:,:,:]+np.sqrt(3)*amp2[j,:,:,:])
    s1 = np.einsum('el,el->',ri[occ:,b+occ,:occ,i],s1_k)

    s2_k = np.einsum('klc,kdc->ld',amp1[:,:,:,a],amp1[j,:,:,:]) + np.einsum('klc,kdc->ld',amp2[:,:,:,a],amp2[j,:,:,:])
    s2 = np.einsum('ld,ld->',ri[:occ,b+occ,occ:,i],s2_k)

    sum_4 = -0.25 * s1 + 0.5 * s2
    

    return (-sum_ab - sum_ij + sum_2 + sum_3 + sum_4)

#A''(3) Eq. (2.160) Kasper
def A_3_2_Kasper(i,a,j,b):
    a = a + occ
    b = b + occ

    s1,s2,s3,s4,s5,s6,s7,s8,s9,s10 = 0,0,0,0,0,0,0,0,0,0
    
    if a==b:
        s1_ = (2 * np.einsum('ckk->c',ri[j,occ:,:occ,:occ]) -4 * np.einsum('kkc->c',ri[j,:occ,:occ,occ:])
               + np.einsum('ddc->c',ri[j,occ:,occ:,occ:]) -np.einsum('cdd->c',ri[j,occ:,occ:,occ:]))
        s1 = np.einsum('c,c->',s1_, dens_secn_ov[i,:])

        s2 =(-2 * np.einsum('ck,k->',ri[occ:,:occ,j,i],dens_secn_ov[:,a-occ]))
        
        s3_ = (2 * np.einsum('ck->kc',ri[occ:,i,j,:occ]) - np.einsum('kc->kc',ri[:occ,occ:,j,i])
        - np.einsum('kc->kc',ri[:occ,i,j,occ:]))
        s3 = np.einsum('kc,kc->',s3_,dens_secn_ov[:,:])

    if i==j:
        s4_ = (2 * np.einsum('kll->k',ri[:occ,b,:occ,:occ])- 2* np.einsum('kll->k',ri[:occ,:occ,:occ,b])
               +np.einsum('kcc->k',ri[:occ,occ:,occ:,b]) - np.einsum('kcc->k',ri[:occ,b,occ:,occ:]))
        s4 = np.einsum('k,k->',s4_,dens_secn_ov[:,a-occ])

        s5_ = (- 2* np.einsum('ck->kc',ri[occ:,b,a,:occ]) + 3 * np.einsum('kc->kc',ri[:occ,occ:,a,b])
                +np.einsum('ck->kc',ri[a,occ:,:occ,b]) )
        s5 =  np.einsum('kc,kc->',s5_,dens_secn_ov[:,:])
        
    s6_ = - ri[:occ,i,j,b]- ri[j,i,:occ,b]
    s6 = np.einsum('k,k->',s6_,dens_secn_ov[:,a-occ])

    s7_ =( -2 * ri[a,i,j,:occ]  + 3*ri[j,i,a,:occ])
    s7 = np.einsum('k,k->',s7_,dens_secn_ov[:,b-occ])

    s8_ =  (2 * ri[a,i,occ:,b] - 2 * ri[a,b,occ:,i])
    s8 = np.einsum('c,c->',s8_,dens_secn_ov[j,:])

    s9_ =  ri[a,occ:,j,b]  + 3* ri[a,b,j,occ:]  + 2 * ri[a,j,occ:,b]
    s9 = np.einsum('c,c->',s9_,dens_secn_ov[i,:]) 

    s10 = -2* np.einsum('kk,->',ri[a,:occ,j,:occ],dens_secn_ov[i,b-occ])

    return 0.25*(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10)


def A_3_2_Geertsen(i,a,j,b):
    a = a + occ
    b = b + occ

    s1,s2=0,0
    
    if a==b:
        s1_ = (4 * ri[j,i,:occ,occ:] - ri[j,:occ,occ:,i] - ri[:occ,i,j,occ:])
        s1 = -0.5 * np.einsum('kc,kc->',s1_, dens_secn_ov)

    if i==j:
        s2_ = (-4 * ri[a,b,:occ,occ:] + ri[a,:occ,occ:,b] + ri[:occ,b,a,occ:])
        s2 = -0.5 * np.einsum('kc,kc->',s2_,dens_secn_ov)

    s3 = - np.einsum('k,k->',ri[j,:occ,a,i],dens_secn_ov[:,b-occ]) - np.einsum('k,k->',ri[:occ,i,j,b],dens_secn_ov[:,a-occ])

    s4 =  (np.einsum('c,c->',ri[a,i,occ:,b],dens_secn_ov[j,:]) + np.einsum('c,c->',ri[a,occ:,j,b],dens_secn_ov[i,:]))

    s5 = - 0.5 * (np.einsum('c,c->',ri[a,b,occ:,i],dens_secn_ov[j,:]) + np.einsum('c,c->',ri[a,b,j,occ:],dens_secn_ov[i,:]))

    s6 = 0.5 * (np.einsum('k,k->',ri[j,i,a,:occ],dens_secn_ov[:,b-occ]) + np.einsum('k,k->',ri[j,i,:occ,b],dens_secn_ov[:,a-occ]))


    return (s1+s2+s3+s4+s5+s6)



#A'''(3) Eq. (2.161) Kasper. Mistake in Kasper's thesis!!
def A3_3(i,a,j,b):
    a = a + occ
    b = b + occ
    s1, s2 = 0,0

    if a==b:
        s1 = np.einsum('ckd,kcd->', ri[j,occ:,:occ,occ:],m_amp_2nd[i,:,:,:])

    if i==j:
        s2 = np.einsum('klc,klc->', ri[:occ,b,:occ,occ:],m_amp_2nd[:,:,a-occ,:])

    return 0.5 * -(s1+s2)

#A''''(3) Eq. (2.162) Kasper.
#Derived by Juliane
def A_3_4(i,a,j,b):
    a = a+occ
    b = b+occ
    output = 0.5*(mo_e[b]-mo_e[j])*((a==b)*dens_third_oo[i,j]-(i==j)*dens_third_vv[b-occ,a-occ])
    return output


def A(i,a,j,b, A_order): 

    zeroth_order = A_0(i, a ,j, b)

    if A_order == 0:
        return zeroth_order

    if A_order==1:
        first_order  = A_1(i, a, j, b)
        return zeroth_order + first_order



    if A_order==2:
        first_order  = A_1(i, a, j, b)
        Second_order = 0.5 * (A_2(i,a,j,b)
                                            +A_2(j,b,i,a))
        return (zeroth_order + first_order + Second_order)
    
    if A_order==3:
        first_order  = A_1(i, a, j, b)
        Second_order = 0.5*(A_2(i,a,j,b) + A_2(j,b,i,a))
                                           
        Third_order  = (0.5 * (A_3_1_Geertsen( i,a,j,b) + A_3_1_Geertsen( j,b,i,a))
                      +0.5 * (A_3_2_Geertsen( i,a,j,b)+A_3_2_Geertsen( j,b,i,a))
                       + 0.5 * (A3_3(i,a,j,b) + A3_3(j,b,i,a))   +  0.5*(A_3_4(i,a,j,b)+A_3_4(j,b,i,a))
                       )
        
                
                    # ((A_3_1_1_Kasper(  i, a, j ,b)))
                    #+  0.5 * (A_3_1_2_Kasper(i,a,j,b) + A_3_1_2_Kasper(j,b,i,a)) )
                    #+  0.5 * (A_3_2_Kasper(i,a,j,b)+A_3_2_Kasper(j,b,i,a))
                    #+   0.5 * (A3_3(i,a,j,b)+A3_3(j,b,i,a))) 
                    #+  0.5*(A_3_4(i,a,j,b)+A_3_4(j,b,i,a)))

                    #0.5 * (A_3_1_Geertsen(i,a,j,b) + A_3_1_Geertsen(j,b,i,a))
                    # 0.5 * (A_3_2_Geertsen(i,a,j,b)+A_3_2_Geertsen(j,b,i,a))


        return (zeroth_order + first_order + Second_order + Third_order)


#B(0) = 0
#B(1) Eq. (109) AOSOPPA
def B_1(i,a,j,b):
    return (ri[a+occ,j,b+occ,i] - 2*ri[a+occ,i,b+occ,j])

#B(2) Eq. (110) AOSOPPA
def B_2( i, a, j, b):
    element = 0.5*(np.einsum('kc,kc->',ri[a+occ,j,:occ ,occ:],m_amp[i,:,b,:])
            + np.einsum('kc,kc->',ri[b+occ,i,:occ ,occ:],m_amp[j,:,a,:])
            + np.einsum('ck,kc->',ri[a+occ,occ:,:occ ,j],m_amp[:,i,b,:])
            + np.einsum('ck,kc->',ri[b+occ,occ:,:occ ,i],m_amp[:,j,a,:])

            -np.einsum('kl,kl->',ri[:occ,i,:occ,j],m_amp[:,:,a,b])
            -np.einsum('cd,cd->',ri[a+occ,occ:,b+occ,occ:],m_amp[i,j,:,:]))

    return element

def B(i, a, j, b, order):

    if order == 1:
        return  B_1(i,a,j,b)
    
    if order == 2:
        return  B_1( i,a,j,b) + B_2(i, a, j, b)


#C matrices:

def C_S1_func(i,a,j,b,k,c):
    a = a + occ
    b = b + occ
    c = c + occ 

    factor = (1/(np.sqrt(2*(1+(a==b))*(1+(i==j)))))
    element =((i==k)*(ri[a,j,b,c]+ri[a,c,b,j])
        +(j==k)*(ri[a,i,b,c]+ri[a,c,b,i])
        -(a==c)*(ri[k,j,b,i]+ri[k,i,b,j])
        -(b==c)*(ri[a,i,k,j]+ri[a,j,k,i]))

    return factor * element


def C_S2_func( i,a,j,b,k,c):
    a = a + occ
    b = b + occ
    c = c + occ 

    factor = np.sqrt(3/2)
    element = ((i==k)*(ri[b,j,a,c]-ri[a,j,b,c]) + (j==k)*(ri[a,i,b,c]-ri[a,c,b,i]) 
    + (a==c)*(ri[b,i,k,j]-ri[b,j,k,i]) + (b==c)*(ri[a,j,k,i]-ri[a,i,k,j]))

    return factor * element


#D(0) matrix
def D_func_S1S2(i,a,j,b):
    return (mo_e[a+occ]+mo_e[b+occ]-mo_e[i]-mo_e[j])



#ph_S matrix
def ph_S(i,a,j,b):

    zeroth_order = (a==b)*(i==j)

    second_order=0.5*(a==b)*dens_secn_oo[i,j]-0.5*(i==j)*dens_secn_vv[a,b]

    return zeroth_order + second_order



