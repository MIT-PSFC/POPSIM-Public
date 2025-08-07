import chex

from popsim.interfaces.cfspopcon_scenario import load_cfspopcon_scenario_for_td_popcon


def load_cfspopcon_prd():
    return load_cfspopcon_scenario_for_td_popcon("SPARC_PRD")


@chex.dataclass
class Sparc2020TestData:
    """Data from Creely et al., 2020 and Rodriguez-Fernandez et al., 2020. Meant to serve as a reference for testing."""

    R0: float = 1.85  # major radius [m]
    a_minor: float = 0.57  # minor radius [m]
    epsilon: float = 0.31  # inverse aspect ratio [-]
    B0: float = 12.2  # magnetic field [T]
    Ip: float = 8.7  # plasma current [MA]
    kappa_sep: float = 1.97  # elongation at separatrix [-]
    delta_sep: float = 0.54  # triangularity at separatrix [-]
    Q: float = 11.0  # Energy gain factor [-]
    qstar_uckan: float = 3.05  # qstar calculated with the Uckan formula [-]
    rhostar: float = 0.0027  # normalized ion Larmor radius [-]
    nu_eff: float = 0.16  # effective collisionality [-]
    nu_star: float = 0.029  # dimensionless collisionality [-]
    Hfactor: float = 1.0  # H-factor [-]
    tauE: float = 0.77  # energy confinement time [s]
    Paux: float = 11.1  # auxiliary power [MW]
    Pohm: float = 1.7  # ohmic power [MW]
    Zeff: float = 1.5  # effective charge [-]
    dilution: float = 0.85  # dilution [-]
    Te_vol: float = 7.3  # volume-averaged electron temperature [keV]
    Ti_vol: float = 7.3  # volume-averaged ion temperature [keV]
    ne_vol: float = 31.0  # volume-averaged electron density [1e19 m^-3]
    ni_vol: float = 27.0  # volume-averaged ion density [1e19 m^-3]
    nu_Te: float = 2.5  # electron temperature peaking factor [-]
    nu_ni: float = 2.5  # ion density peaking factor [-]
    greenwald_frac: float = 0.37  # Greenwald fraction [-]
    beta: float = 0.012  # plasma beta [-]
    betaN: float = 1.0  # normalized beta [-]
    Pfusion: float = 140  # fusion power [MW]
    Prad: float = 10.4  # radiated power [MW]
    PsepB0R0: float = 191.0  # (Psep * B0)/R0 [MW T m^-1]
