import chex

from cfspopcon.jax_compatible import current_drive, geometry

"""
TODO: implement a "time-varying-geom" module that accepts a time (or path) parameter.
TODO: implement an interface into Grad-Shafranov solutions.
"""


@chex.dataclass
class GeometryCFSPopcon:
    major_radius: float
    inverse_aspect_ratio: float
    areal_elongation: float
    elongation_ratio_sep_to_areal: float
    triangularity_psi95: float
    triangularity_ratio_sep_to_psi95: float

    @property
    def plasma_volume(self):
        plasma_volume = geometry.calc_plasma_volume(
            major_radius=self.major_radius,
            inverse_aspect_ratio=self.inverse_aspect_ratio,
            areal_elongation=self.areal_elongation,
        )
        return plasma_volume

    @property
    def separatrix_triangularity(self):
        separatrix_triangularity = self.triangularity_psi95 * self.triangularity_ratio_sep_to_psi95
        return separatrix_triangularity

    @property
    def minor_radius(self):
        minor_radius = self.major_radius * self.inverse_aspect_ratio
        return minor_radius

    @property
    def separatrix_elongation(self):
        separatrix_elongation = self.areal_elongation * self.elongation_ratio_sep_to_areal
        return separatrix_elongation

    @property
    def f_shaping(self):
        f_shaping = current_drive.calc_f_shaping(self.inverse_aspect_ratio, self.areal_elongation, self.triangularity_psi95)
        return f_shaping

    @property
    def surface_area(self):
        sa = geometry.calc_plasma_surface_area(
            major_radius=self.major_radius,
            inverse_aspect_ratio=self.inverse_aspect_ratio,
            areal_elongation=self.areal_elongation,
        )
        return sa
