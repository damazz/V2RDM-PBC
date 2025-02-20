import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.lib import logger
import scipy.linalg
import pyscf.scf as mscf
import sparse
from create_integrals import active_F_matrix
from create_integrals import active_two_electron
from create_integrals import core_energy
from create_integrals import get_kconserve_for2RDM


cell = gto.Cell()
cell.atom = "NiO_primitive_cell.xyz"
dim = 3
cell.dimension = dim
cell.a = [[2.9816,0.000,0.000],[1.49080,2.58214,0.000],[1.49080,0.86071,2.43447]]
cell.basis = {'O':'gth-dzvp','Ni':'gth-dzvp-molopt-sr'}
#pseudo file must be obtained from Pyscf
pseudo_file = 'gth-hf-rev.dat'
cell.pseudo = {'Ni':gto.pseudo.load(pseudo_file, 'Ni'),'O': gto.pseudo.load(pseudo_file, 'O')}
cell.verbose = 4
cell.unit = 'A'  # Angstroms
cell.exp_to_discard=0.1
cell.spin = 0
cell.precision = 1e-12
cell.low_dim_ft_type = 'inf_vacuum'#'analytic_2d_1'
cell.build()

nkpts = 27
kpts = cell.make_kpts([3, 3, 3])

# ---------- HF calculation --------- #
nao = cell.nao_nr()
kmf = scf.KROHF(cell, kpts).density_fit()
kmf.max_cycle = 150
kmf = mscf.addons.remove_linear_dep_(kmf).run()

# ------ HF results
print("HF results:")
print("Nuclear energy", kmf.energy_nuc())
print("total energy:", kmf.e_tot)
print()

C = np.asarray(kmf.mo_coeff)
np.save("NiO_primitive_333_mo_coeff", C)


#active space size 6
# Starting and ending indices of active space
active = 4
act_ind1 = 10
act_ind2 = 14
core = 10
core_ind = list(list(range(0 + nao*k, act_ind1 + nao*k)) for k in range(nkpts))
core_ind = sum(core_ind, [])


F_act = active_F_matrix(kmf, cell, kpts, active, act_ind1, act_ind2, core, nkpts, nao)

np.save("NiO_F_act_333_primitive_active4", F_act)

V = active_two_electron(kmf, cell, kpts, active, act_ind1, act_ind2, nkpts, nao)

sparse.save_npz("NiO_eri_333_primitive_active4", V)

core_eng  = core_energy(kmf, cell, kpts, core_ind, core, nkpts, nao)

print("Core Energy active 4:", core_eng)
print("Warning: Remember that Maple core energy includes the nuclear energy but this does not!!!")
print("Nuclear energy:", kmf.energy_nuc())

# 2-electron energy
from pyscf.pbc.tools import madelung
madelung_const = madelung(kmf.cell, kmf.kpts)
nocc = cell.nelectron//2
print(" - Madelung correction:", -madelung_const*nocc)
print(nocc)
# ----------------------------------- #

sym_list = get_kconserve_for2RDM(cell, kpts, nkpts)


np.save('NiO_333_primitive_kconserve', np.asarray(sym_list, dtype = int))
