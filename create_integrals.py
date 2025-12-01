# V2RDM-PBC
# Copyright Anna O. Schouten and David A. Mazziotti 2025

import numpy as np
import scipy.linalg
import sparse

def active_F_matrix(kmf, cell, kpts, active, act_ind1, act_ind2, core, nkpts, nao):
    C = np.asarray(kmf.mo_coeff)

    # Create P
    P = np.zeros((nkpts, nao, nao), dtype=complex)
    for k in range(nkpts):
        for sigma in range(nao):
            for rho in range(nao):
                for i in range(core):
                    P[k, sigma, rho] = P[k, sigma, rho] + C[k, sigma, i]*C[k, rho, i].conj()

    # T
    T_ao = kmf.get_hcore()
    T_mo1 = np.einsum('kmi,kmn,knj->kij', C.conj(), T_ao, C)
    T_mo = scipy.linalg.block_diag(*T_mo1).reshape(nkpts, nao, nkpts, nao).transpose(0,2,1,3)

    # Active Space 1-electron integrals
    F_act_ao = np.zeros((nkpts, nao, nao), dtype=complex)
    J, K = kmf.get_jk(cell, P, hermi=1, kpt = kpts) 
    for k in range(nkpts):
        for u in range(nao):
            for v in range(nao):
                F_act_ao[k,u,v] = T_ao[k,u,v]/(nkpts) + 2*J[k,u,v]/(nkpts) - K[k,u,v]/(nkpts)
                
    F_act_mo1 = np.einsum('kmi,kmn,knj->kij', C.conj(), F_act_ao, C)
    F_act_mo = scipy.linalg.block_diag(*F_act_mo1).reshape(nkpts, nao, nkpts, nao)

    F_act_mo_as = np.zeros((nkpts, nkpts, active, active), dtype=complex)
    for i in range(nkpts):
        x1 = act_ind1
        x2 = act_ind2
        F_act_mo_as[i,i] = F_act_mo.transpose(0,2,1,3)[i,i,x1:x2,x1:x2]

    F_act_mo_as = F_act_mo_as.transpose(0,2,1,3).reshape((nkpts)*active, (nkpts)*active)

    return F_act_mo_as

def active_two_electron(kmf, cell, kpts, active, act_ind1, act_ind2, nkpts, nao):
    C = np.asarray(kmf.mo_coeff)
    # 2-electron integrals
    from pyscf.pbc.lib import kpts_helper
    khelper = kpts_helper.KptsHelper(cell, kpts)
    kconserv = khelper.kconserv
    V = np.zeros((nkpts, nkpts, nkpts, nkpts, active, active, active, active), dtype=complex)
    fao2mo = kmf.with_df.ao2mo
    sym_list = np.zeros(((nkpts)**3, 4))
    count = 0
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        # Select individual p/q/r/s coefficients and kpts
        mo_p, mo_q, mo_r, mo_s = C[[kp,kq,kr,ks]]
        p1 = act_ind1
        p2 = act_ind2
        mo_p = mo_p[:, p1:p2]
        q1 = act_ind1
        q2 = act_ind2
        mo_q = mo_q[:, q1:q2]
        r1 = act_ind1
        r2 = act_ind2
        mo_r = mo_r[:, r1:r2]
        s1 = act_ind1
        s2 = act_ind2
        mo_s = mo_s[:, s1:s2]
        k_p, k_q, k_r, k_s = kpts[[kp, kq, kr, ks]]

        V[kp, kq, kr, ks] = fao2mo((mo_p, mo_q, mo_r, mo_s), (k_p, k_q, k_r, k_s), compact=False).reshape((active, active, active, active))
        # Create the symmetry list in Physicist notation
        sym_list[count] = (kp+1), (kr+1), (kq+1), (ks+1)
        count += 1

    V = V.transpose(0,4,1,5,2,6,3,7).reshape(((nkpts)*active,)*4)
    V = V.transpose(0,2,1,3)
    V = V/(nkpts**2)
    V = sparse.COO(V)

    return V

def core_energy(kmf, cell, kpts, core_ind, core, nkpts, nao):
    C = np.asarray(kmf.mo_coeff)
    # T
    T_ao = kmf.get_hcore()
    T_mo1 = np.einsum('kmi,kmn,knj->kij', C.conj(), T_ao, C)
    T_mo = scipy.linalg.block_diag(*T_mo1).reshape(nkpts, nao, nkpts, nao).transpose(0,2,1,3)
    T_mo2 = T_mo.transpose(0,2,1,3).reshape((nkpts)*nao, (nkpts)*nao)
	
    # Core Energy
    core_energy = 0
    for i in core_ind:
    	core_energy += 2*T_mo2[i,i]/(nkpts)
    from pyscf.pbc.lib import kpts_helper
    khelper = kpts_helper.KptsHelper(cell, kpts)
    fao2mo = kmf.with_df.ao2mo
    for a in range(nkpts):
    	for c in range(nkpts):
            mo1C, mo2C, mo3C, mo4C = C[[a,a,c,c]]
            moC_1 = mo1C[:,0:core]
            moC_2 = mo2C[:,0:core]
            moC_3 = mo3C[:,0:core]
            moC_4 = mo4C[:,0:core]
            k1C, k2C, k3C, k4C = kpts[[a, a, c, c]]
            V1 = fao2mo((moC_1,moC_2,moC_3,moC_4),(k1C,k2C,k3C,k4C), compact=False).reshape((core,core,core,core))
            mo1E, mo2E, mo3E, mo4E = C[[a,c,c,a]]
            moE_1 = mo1E[:,0:core]
            moE_2 = mo2E[:,0:core]
            moE_3 = mo3E[:,0:core]
            moE_4 = mo4E[:,0:core]
            k1E, k2E, k3E, k4E = kpts[[a, c, c, a]]
            V2 = fao2mo((moE_1,moE_2,moE_3,moE_4),(k1E,k2E,k3E,k4E), compact=False).reshape((core,core,core,core))
            for b in range(core):
            	for d in range(core):
                    core_energy += 2*V1[b,b,d,d]/(nkpts)**2 - V2[b,d,d,b]/(nkpts)**2

    return core_energy


def get_kconserve_for2RDM(cell, kpts, nkpts):
    # 2-electron integrals
    from pyscf.pbc.lib import kpts_helper
    khelper = kpts_helper.KptsHelper(cell, kpts)
    kconserv = khelper.kconserv
    sym_list = np.zeros(((nkpts)**3, 4))
    count = 0
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        # Create the symmetry list in Physicist notation
        sym_list[count] = (kp+1), (kr+1), (kq+1), (ks+1)
        count += 1 

    return sym_list
