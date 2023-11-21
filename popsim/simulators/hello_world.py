import equinox as eqx
from cfspopcon.named_options import ReactionType
from popsim.algorithms.zeff_and_dilution_from_impurities import CalcZeffAndDilutionFromImpurities


class State(eqx.Module):
    stored_energy: float
    average_ion_density: float


class Params(eqx.Module):
    major_radius: float
    magnetic_field_on_axis: float
    inverse_aspect_ratio: float
    areal_elongation: float
    elongation_ratio_sep_to_areal: float
    triangularity_psi95: float
    triangularity_ratio_sep_to_psi95: float
    plasma_current: float
    fraction_of_external_power_coupled: float
    heavier_fuel_species_fraction: float
    normalized_inverse_temp_scale_length: float
    electron_density_peaking_offset: float
    ion_density_peaking_offset: float
    temperature_peaking: float
    ion_to_electron_temp_ratio: float




class HelloWorldModel(eqx.Module):
    fusion_reaction: ReactionType
    zeff_dilution_calc: CalcZeffAndDilutionFromImpurities

    def __init__(self, fusion_reaction: ReactionType, zeff_dilution_calc: CalcZeffAndDilutionFromImpurities):
        self.fusion_reaction = fusion_reaction
        self.zeff_dilution_calc = zeff_dilution_calc

    def __call__(self, state: State, params: Params) -> State:
        self.zeff_dilution_calc(
            state.average_ion_density,
        )
