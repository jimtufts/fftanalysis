import itertools
import MDAnalysis as mda
import MDAnalysis.analysis.base
from MDAnalysis.analysis import distances
import numpy as np


class DistanceAnalysis(MDAnalysis.analysis.base.AnalysisBase):
    def __init__(self, ag1, ag2, **kwargs):
        super(DistanceAnalysis, self).__init__(
            ag1.universe.trajectory, **kwargs)
        self._ag1 = ag1
        self._ag2 = ag2

    def _prepare(self):
        self.result = []

    def _single_frame(self):
        self.result.append(distances.dist(self._ag1, self._ag2)[-1])

    def _conclude(self):
        self.result = np.array(self.result)


class COM_analysis(MDAnalysis.analysis.base.AnalysisBase):
    def __init__(self, ag, **kwargs):
        super(COM_analysis, self).__init__(ag.universe.trajectory, **kwargs)
        self._ag = ag

    def _prepare(self):
        self.result = []

    def _single_frame(self):
        self.result.append(self._ag.center_of_mass())

    def _conclude(self):
        self.result = np.array(self.result)


def dis(vector):
    dis = []
    for i, vec in enumerate(vector):
        dis.append(np.linalg.norm(vec))
    dis = np.array(dis)
    return dis


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def distance_Phe_His(pdb, traj):  # info = [pdb, traj, ]
    sim = mda.Universe(pdb, traj)

    # Select atoms
    His_ring = sim.select_atoms(
        'resid 163 and not backbone and not name CB and not name H*')
    Phe_ring = sim.select_atoms(
        'resid 140 and not backbone and not name CB and name C*')
    # Calculation
    COM_Phe = []
    for n in range(2):
        com2 = COM_analysis(Phe_ring[int(n*6):int(n*6+6)])
        com2.run()
        COM_Phe.append(com2.result[:])
    COM_Phe = np.array(COM_Phe)
    COM_His = []
    for n in range(2):
        com3 = COM_analysis(His_ring[int(n*5):int(n*5+5)])
        com3.run()
        COM_His.append(com3.result[:])
    COM_His = np.array(COM_His)
    vec_Phe_His_A = COM_Phe[0] - COM_His[0]
    vec_Phe_His_B = COM_Phe[1] - COM_His[1]
    dist = np.array([dis(vec_Phe_His_A)]+[dis(vec_Phe_His_B)])
    dist = dist.T
    return dist


def distance_Arg_Glu(pdb, traj):
    sim = mda.Universe(pdb, traj)
    # Select atoms
    N_Arg = sim.select_atoms('resid 4 and name NH*')
    Glu = sim.select_atoms('resid 290 and name OE*')
    # Calculation
    N_Arg = N_Arg[3] + N_Arg[2] + N_Arg[3] + N_Arg[2] + \
        N_Arg[1] + N_Arg[0] + N_Arg[1] + N_Arg[0]
    Glu = Glu[0] + Glu[0] + Glu[1] + Glu[1] + Glu[2] + Glu[2] + Glu[3] + Glu[3]
    dis = DistanceAnalysis(Glu, N_Arg)
    dis.run()
    dis = np.array([np.min(dis.result[:, 0:3], 1)] +
                   [np.min(dis.result[:, 4:7], 1)])
    dis = dis.T
    return dis


def distance_Gln_Arg(pdb, traj):
    sim = mda.Universe(pdb, traj)
    Gln = sim.select_atoms('resid 299 and (name OE1 or name HE22)')
    Arg = sim.select_atoms('resid 4 and (name H or name O)')
    dis = DistanceAnalysis(Gln, Arg)
    dis.run()
    dis = np.array([np.min(dis.result[:, 0:1], 1)] +
                   [np.min(dis.result[:, 2:3], 1)])
    dis = dis.T
    return dis


def distance_Leu_Phe(pdb, traj):
    sim = mda.Universe(pdb, traj)
    Leu141 = sim.select_atoms(
        'resid 141 and (name HD12 or name HD21 or name HB2)')
    Phe3 = sim.select_atoms('resid 3 and (name CD1 or name CZ or name CD2)')
    Leu = Leu141[1] + Leu141[2] + Leu141[0] + Leu141[4] + Leu141[5] + Leu141[3]
    Phe = Phe3[3] + Phe3[4] + Phe3[5] + Phe3[0] + Phe3[1] + Phe3[2]
    dis = DistanceAnalysis(Leu, Phe)
    dis.run()
    dis = np.array([np.min(dis.result[:, 0:3], 1)] +
                   [np.min(dis.result[:, 4:7], 1)])
    dis = dis.T
    return dis


def oxyanion_dis(pdb, traj, res1, res2):
    sim = mda.Universe(pdb, traj)
    # Select atoms
    NH1 = sim.select_atoms(f'resid {res1} and name H')
    NH2 = sim.select_atoms(f'resid {res2} and name H')

    # Calculation
    dis1 = DistanceAnalysis(NH1, NH2)
    dis1.run()
    dis = dis1.result[:]
    return dis
