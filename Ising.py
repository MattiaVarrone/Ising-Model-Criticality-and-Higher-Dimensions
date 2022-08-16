# dependencies:
import numpy as np


# The lattice side cannot be odd, so we create an exception to avoid it
class OddSideException(Exception):
    def __init__(self):
        print("The side length (N) must be even")


class Lattice:

    # N: side
    # d: dimensions
    # K: number of lattices in the ensemble
    # free_bound: determines if boundary is free or periodic
    def __init__(self, N, d=2, K=1, free_bound=False):
        if N % 2 == 1:
            raise OddSideException
        self.N = N
        self.d = d
        self.free_bound = free_bound
        self.tot_sites = N ** d
        self.shape = [K] + [N for _ in range(d)]
        self.lat = np.ones(self.shape)
        if free_bound:
            pad_width = [[0, 0]] + [[1, 0] for _ in range(d)]
            self.lat = np.pad(self.lat, pad_width)

    # randomises spin configuration
    def random(self):
        self.lat = np.random.binomial(1, 0.5, self.shape) * 2 - 1
        if self.free_bound:
            pad_width = [[0, 0]] + [[1, 0] for _ in range(self.d)]
            self.lat = np.pad(self.lat, pad_width)

    # aligns spins in one direction
    def align(self):
        self.lat = np.ones(self.shape)
        if self.free_bound:
            pad_width = [[0, 0]] + [[1, 0] for _ in range(self.d)]
            self.lat = np.pad(self.lat, pad_width)

    # Hamiltonian term due to neighbours interaction and magnetic field
    def energy(self, H=0):
        E_J = np.zeros_like(self.lat)
        for i in range(1, self.d + 1):
            E_J += np.roll(self.lat, 1, i) + np.roll(self.lat, -1, i)
        E_J *= -self.lat
        E_H = -H * self.lat
        return E_J + E_H

    # intermediate update step of the spin lattice using the chessboard algorithm
    # we return the energy because it is a useful stat
    def _update_chess(self, mask, T, H=0):
        E = self.energy(H)
        d_E = - 2 * E
        p = np.exp(-d_E / T) * mask
        self.lat[p > np.random.random(self.lat.shape)] *= -1
        energy = np.sum(E, axis=tuple([i for i in range(1, self.d + 1)])) / self.tot_sites
        return energy

    #full update step of the lattice
    def update(self, T, H=0, energy=False):
        mask = np.sum(np.indices(self.lat.shape), 0) % 2  # create multi-dimensional chessboard
        E = self._update_chess(mask, T, H)
        self._update_chess(1 - mask, T, H)
        if energy:
            return E / 2

    # performs a given number of lattice sweeps
    def evolve(self, n, T, H=0):
        for _ in range(n):
            self.update(T, H)

    def magn(self):
        return np.sum(self.lat, axis=tuple([i for i in range(1, self.d + 1)])) / self.tot_sites

    # produces a time series of Energy and Magnetisation values
    def stats_equ(self, n, T, H=0):
        Es, Ms = [], []
        for _ in range(n):
            M = np.average((self.magn()))
            E = np.average(self.update(T, H, energy=True))
            Es.append(E), Ms.append(M)
        return Es, Ms

    # the decorrelation time for the system is computed
    # choose n >> tau
    # first the system runs evolve() t0 times to reach equilibrium
    def time_decorr(self, n, T, H=0, tau=0, t0=200):
        magns, acorrs, decorr_times = [], [], []
        self.evolve(t0, T, H)
        for _ in range(n):
            self.update(T, H)
            magns.append(self.magn())
        m_mean = np.average(magns, axis=0)
        d_m = magns - m_mean
        m_var = np.var(magns, axis=0)
        tau = n // 10 if tau == 0 else tau  # default tau
        for t in range(tau):
            acorr = np.average(d_m[:n - t] * d_m[t:], axis=0) / m_var
            if np.any(acorr < 0.1):  # error becomes greater than the expected correlation
                if t < 4:
                    return 1, 0
                else:
                    break
            acorrs.append(acorr)
        acorrs = np.array(acorrs).T
        for series in acorrs:
            exp_fit = np.polyfit(range(len(series)), np.log(series), 1)
            decorr_times.append(-1 / exp_fit[0])
        decorr_time = np.average(decorr_times)
        error = np.std(decorr_times)
        return decorr_time, error

    # calculates Magnetisation, Magnetic Susceptibility, Energy, and Specific Heat with corresponding errors
    def stats(self, n, T, H=0, t0=200, abs=True):
        magns, E = [], []
        self.evolve(t0, T, H)
        for _ in range(n):
            magns.append(self.magn())
            E_spec = self.update(T, H, energy=True)
            E.append(E_spec)
        magns = np.abs(magns) if abs else magns
        Ms = np.average(magns, axis=0)
        Chis = np.var(magns, axis=0) * self.tot_sites / T
        Es = np.average(E, axis=0)
        Cs = np.var(E, axis=0) * self.tot_sites / (T ** 2)
        stats = np.average([Ms, Chis, Es, Cs], axis=1)
        errs = np.std([Ms, Chis, Es, Cs], axis=1)
        return stats, errs

    # computes averages of M, Chi, C for temperatures varying in a range T_range
    # it is a useful function to calculate critical exponents if T_crit is contained in T_range
    def crit_data(self, T_range, H=0, n=1000, t0=400):
        Ms, M_errs = [], []
        Chis, Chi_errs = [], []
        Cs, C_errs = [], []
        for T in T_range:
            stat, err = self.stats(n, T, H=H, t0=t0)
            Ms.append(stat[0]), M_errs.append(err[0])
            Chis.append(stat[1]), Chi_errs.append(err[1])
            Cs.append(stat[3]), C_errs.append(err[3])
        return [Ms, Chis, Cs], [M_errs, Chi_errs, C_errs]

    # the system is run through a range of external magnetic fields and the magnetisation is recorded to measure hysteresis.
    def hysteresys(self, n, T, H_max, H_steps, cycles):
        magns = []
        Hs = np.linspace(-H_max, H_max, H_steps)
        Hs = np.concatenate([Hs[:-1], np.flip(Hs[1:])])
        Hs = np.concatenate([Hs for _ in range(cycles)])
        Hs = Hs[H_steps // 2:]
        for H in Hs:
            self.evolve(n, T, H)
            magns.append(self.magn())
        return Hs, magns
