import mdtraj as md
import bpmfwfft.grids as grids
import netCDF4 as nc
import numpy as np
from os import walk
import socket
from bpmfwfft.grids import RecGrid
from bpmfwfft.grids import LigGrid
import copy
import pickle as p


if socket.gethostname() == 'jim-Mint':
    ppi_path = '/media/jim/Research_TWO/FFT_PPI'
    home = '/home/jim'
else:
    ppi_path = '/home/jtufts/Desktop/FFT_PPI'
    home = '/home/jtufts'

def rmsd(P: np.ndarray, Q: np.ndarray, **kwargs) -> float:
    diff = P - Q
    return np.sqrt((diff * diff).sum() / P.shape[0])

grid_path = f'{ppi_path}/2.redock/4.receptor_grid/2OOB_A:B'
grid_nc = '%s/grid_0_5.nc'%grid_path


#parsing receptor grid variables
grid_variables = nc.Dataset(grid_nc, 'r').variables
counts = nc.Dataset(grid_nc, 'r').variables["counts"][:]
x = nc.Dataset(grid_nc, 'r').variables["x"][:]
y = nc.Dataset(grid_nc, 'r').variables["y"][:]
z = nc.Dataset(grid_nc, 'r').variables["z"][:]
electrostatic = nc.Dataset(grid_nc, 'r').variables["electrostatic"][:]
lja = nc.Dataset(grid_nc, 'r').variables["LJa"][:]
ljr = nc.Dataset(grid_nc, 'r').variables["LJr"][:]
sasai = nc.Dataset(grid_nc, 'r').variables["SASAi"][:]
sasar = nc.Dataset(grid_nc, 'r').variables["SASAr"][:]
trans_crd = nc.Dataset(grid_nc, 'r').variables["trans_crd"][:]
rec_disp = nc.Dataset(grid_nc, 'r').variables["displacement"][:]


rec_prmtop = f"{ppi_path}/2.redock/1.amber/2OOB_A:B/receptor.prmtop"
lj_sigma_scal_fact = 1.0
rec_inpcrd = f"{ppi_path}/2.redock/2.minimize/2OOB_A:B/receptor.inpcrd"

bsite_file = None
grid_nc_file = f"{ppi_path}/2.redock/4.receptor_grid/2OOB_A:B/grid_0_5.nc"

lig_prmtop = f"{ppi_path}/2.redock/1.amber/2OOB_A:B/ligand.prmtop"
# lig_inpcrd = f"{ppi_path}/2.redock/2.minimze/2OOB_A:B/ligand.inpcrd"

rot_nc = f"{ppi_path}/2.redock/3.ligand_rand_rot/2OOB_A:B/rotation.nc"
lig_rot = nc.Dataset(rot_nc, 'r').variables['positions']
lig_inpcrd = f"{ppi_path}/2.redock/2.minimize/2OOB_A:B/ligand.inpcrd"

rho = 9.0
rc_scale = 0.76
rs_scale = 0.53
rm_scale = 0.55
lc_scale = 0.81
ls_scale = 0.50
lm_scale = 0.54


def _create_rec_grid(rec_prmtop, lj_sigma_scal_fact, rc_scale, rs_scale, rm_scale, rho, rec_inpcrd, bsite_file, grid_nc_file):
    rec_grid = RecGrid(rec_prmtop, lj_sigma_scal_fact, rc_scale, rs_scale, rm_scale, rho, rec_inpcrd, bsite_file, 
                        grid_nc_file, new_calculation=False)
    return rec_grid

def _create_lig_grid(lig_prmtop, lj_sigma_scal_fact, lc_scale, ls_scale, lm_scale, lig_inpcrd, rec_grid):
    lig_grid = LigGrid(lig_prmtop, lj_sigma_scal_fact, lc_scale, ls_scale, lm_scale, lig_inpcrd, rec_grid)
    return lig_grid

def get_top_scores(score, k):

	flat_score = score.flatten()
	flat_argp = np.argpartition(flat_score, -k)[-k:]

	top_scores = flat_score[flat_argp]
	top_scores.sort()
	translations = []
	trans_scores = []

	for score_val in top_scores:
	    t = np.where(score == score_val)
	    a = np.array(t)
	    if a.shape == (3,2):
	        for x,y,z in a.transpose():
	            x = np.array([x])
	            y = np.array([y])
	            z = np.array([z])
	            translations.append((x,y,z))
	            trans_scores.append(score[x,y,z])
	            # print((x,y,z),  score[x,y,z], cfft[x,y,z])
	    else:
	        translations.append(t)
	        trans_scores.append(score[t])

	return translations, trans_scores


def cat_grids(grid1, grid2):
    prmtop = cat_dictionaries(grid1._prmtop,grid2._prmtop)
    crd = cat_dictionaries(grid1._crd,grid2._crd)
    cat_grid = copy.deepcopy(grid1)
    cat_grid._prmtop = prmtop
    cat_grid._crd = crd
    return cat_grid

def cat_dictionaries(dict1, dict2):
    dict1_copy = copy.deepcopy(dict1)
    dict2_copy = copy.deepcopy(dict2)
    
    if isinstance(dict1_copy, dict):
        keys = list(dict1_copy.keys())
        for key in keys:
            # print(keys)
            # print(key)
            dict1_copy[key] = cat_values(dict1[key], dict2[key])
        return dict1_copy
    
    elif isinstance(dict1_copy, np.ndarray):
        return np.concatenate((dict1_copy,dict2_copy))

def cat_values(a1, a2):
    array1 = copy.deepcopy(a1)
    array2 = copy.deepcopy(a2)
    if isinstance(array1, np.ndarray):
        return np.concatenate((array1,array2))
    elif isinstance(array1, dict):
        return cat_dictionaries(array1, array2)
    elif isinstance(array1, (int, np.int64)):
        # print(f"{array1+array2}")
        return array1 + array2
    elif isinstance(array1, list):
        # print(type(array1))
        # print(f'array1:{len(array1)}, array2:{len(array2)}')
        array1.extend(array2)
        # print(f'complex:{len(array1)}, sum:{len(a1)+len(a2)}')
        return array1

def regplot(trans_scores, delta_sasas, colors, name):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    from sklearn.linear_model import LinearRegression

    x = np.array(trans_scores)
    y = np.array(delta_sasas)
    C = np.array(colors, dtype=str)

    change = np.where(C == 'yellow')
    C[change] = "black"

    # indicies = np.where(y < 0.8*y.max())
    indicies = np.where(y == 0)

    indicies = indicies[::-1]

    for i in indicies:
        x = np.delete(x, i)
        y = np.delete(y, i)
        C = np.delete(C, i)

    indicies = np.where(x == 0)
    indicies = indicies[::-1] 

    for i in indicies:
        x = np.delete(x, i)
        y = np.delete(y, i)
        C = np.delete(C, i)

    x = x.reshape((-1,1))

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)


    # scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, 10, c=C, alpha=0.5, marker='+',
            label="SC SCORE vs Delta SASA")
    # regression plot
    rx = np.linspace(x.min(), x.max(), 165)
    ry = model.coef_*rx + model.intercept_

    ax.plot(rx, ry, '-r', label=f'DELTA_SASA={float(model.coef_):.4f}*SC_SCORE + {float(model.intercept_):.2f}')
    plt.xlabel("SC_score")
    plt.ylabel("Delta_SASA (A^2)")
    ax.set_title(f'R2: {r_sq}')
    ax.legend(loc='upper left')
    plt.savefig(f'{name}')

#MAIN SCRIPT, this is awful and lazy, I know. #FIXME

rec_grid = _create_rec_grid(rec_prmtop, lj_sigma_scal_fact, rc_scale, rs_scale, rm_scale, rho, rec_inpcrd, bsite_file, grid_nc_file)

lig_grid = _create_lig_grid(lig_prmtop, lj_sigma_scal_fact, lc_scale, ls_scale, lm_scale, lig_inpcrd, rec_grid)
lig_grid._move_ligand_to_lower_corner()
lig_grid.translate_ligand(np.array([34,43.5,22.5]))
#crystal pose for rmsd
ref = copy.deepcopy(lig_grid._crd)

all_translations = [] 
all_trans_scores = []
all_delta_sasas = []
all_colors = []

rotations = {}
for i,rot_crd in enumerate(lig_rot):
	lig_grid._crd = np.array(rot_crd, dtype=np.float64)
	lig_grid._move_ligand_to_lower_corner()

	names = ["SASA"]
	lgrid = lig_grid.get_ligand_grids(names, [0,0,0])
	rgrid = np.add(sasar, sasai*1.j)

	# do FFT for grids
	lfft = np.fft.fftn(lgrid["SASA"].conjugate())
	rfft = np.fft.fftn(rgrid)
	cfft = np.fft.ifftn(rfft*lfft.conjugate())
	score = (np.real(cfft) - np.imag(cfft)*1000000)#/counts[0]**3

	k = 100  # number of scores to save
	translations, trans_scores = get_top_scores(score, k)
	all_translations.extend(translations)
	all_trans_scores.extend(trans_scores)

	delta_sasas = []
	colors = []
	for v,vector in enumerate(translations):
		vector = (np.array(vector).transpose()*lig_grid._spacing)[0]
		lig_grid._move_ligand_to_lower_corner()
		print(vector)
		lig_grid.translate_ligand(vector)
		com_grid = cat_grids(rec_grid, lig_grid)
		c_sasa = com_grid._get_molecule_sasa(0.14, 960).sum()
		l_sasa = lig_grid._get_molecule_sasa(0.14, 960).sum()
		r_sasa = rec_grid._get_molecule_sasa(0.14, 960).sum()
		d_sasa = (l_sasa + r_sasa) - c_sasa
		delta_sasas.append(d_sasa)
		rmsd_to_native = rmsd(ref, lig_grid._crd)
		if rmsd_to_native < 5.:
			# colors.append([255, 0, 0])
			colors.append("red")
		elif rmsd_to_native < 10 and rmsd_to_native >= 5:
			colors.append("magenta")
		elif rmsd_to_native < 15 and rmsd_to_native >= 10:
			colors.append("purple")
		else:
			colors.append("black")

		all_delta_sasas.extend(delta_sasas)
		all_colors.extend(colors)
	name = f'/home/jim/src/p39/fftanalysis/plots/dSASA_0_5/{i}.jpg'
	
	regplot(trans_scores, delta_sasas, colors, name)
	values = {}
	values['translations'] = translations
	values['trans_scores'] = trans_scores
	values['delta_sasas'] = delta_sasas
	values['colors'] = colors
	p.dump(values, open( f"{name[:-4]}values.p", "wb" ) )
	rotations[i] = values


final = f'/home/jim/src/p39/fftanalysis/plots/dSASA_0_5/2OOB_A.jpg'
p.dump(rotations, open( "/home/jim/src/p39/fftanalysis/plots/dSASA_0_5/rotations.p", "wb" ) )
regplot(all_trans_scores, all_delta_sasas, all_colors, final)

	