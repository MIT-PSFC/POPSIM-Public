from enum import IntEnum
from typing import Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from jaxtyping import Array
from scipy.constants import eV, mu_0

from popsim.basis import Basis1DProtocol, BSplineBasis, InterpedLinearBasis
from popsim.cfspopcon_jax.current_drive import calc_f_shaping, calc_q_star
from popsim.cfspopcon_jax.geometry import calc_plasma_volume
from popsim.ml import DataLoader
from popsim.ml.rtd_mlp import Activation, RtdMLP


class ProfileShape(eqx.Module):
    """
    A module defining a profile shape on the rho grid [0, 1].

    The profile shape can either be specified either directly with points on the rho grid or with a set of coefficients for a B-spline basis.
    """

    basis: Basis1DProtocol  # B-spline basis used to define the profile shape
    coeffs: Array  # Coefficients for the B-spline basis
    normalize: bool = eqx.field(static=True)  # Whether to normalize the profile shape

    def __init__(self, basis: Basis1DProtocol, coeffs: Array, normalize: bool = True):
        self.basis = basis
        self.coeffs = coeffs
        self.normalize = normalize

    def __call__(self, rho: Array, n_normalize: int = 100) -> Array:
        """Evaluate the profile shape at the given rho values.

        Args:
            rho (Array): Grid values at which to evaluate the profile shape.
            n_normalize (int, optional): Number of points to use for normalization. Defaults to 100.

        Returns:
            Array: The profile shape evaluated at each rho value.
        """

        # Check that the rho values are within the range [0, 1]
        rho = eqx.error_if(rho, jnp.any(jnp.logical_or(rho < 0.0, rho > 1.0)), "rho values must be in the range [0, 1]")

        vals = self.basis(self.coeffs, rho)
        if self.normalize:
            vals = vals / self.integral(n_normalize=n_normalize)
        return vals

    def integral(
        self,
        n_normalize: int = 100,
    ) -> float:
        """Calculate the integral of the profile shape."""
        rho_norm = jnp.linspace(0, 1, n_normalize)
        vals_rho_norm = self.basis(self.coeffs, rho_norm)
        integral = jnp.trapezoid(vals_rho_norm, rho_norm)
        return integral

    def visualize(self, rho: Array = None, ax=None):
        """Visualize the profile shape and the components"""
        if rho is None:
            rho = jnp.linspace(0, 1, 100)
        # Calculate the profile shape and its components
        profile_shape = self(rho)

        # Create a new figure and axis if none are provided
        if ax is None:
            fig, ax = plt.subplots()

        # Plot the overall profile shape
        ax.plot(rho, profile_shape, label="Profile Shape", color="black", linewidth=2)

        # Add labels and legend
        ax.set_xlabel("rho")
        ax.set_ylabel("Profile value")
        ax.legend()

        # Display the plot if a new figure was created
        if ax is None:
            plt.show()

    @classmethod
    def make_bspline(cls, coeffs: Array, normalize: bool = True) -> "ProfileShape":
        """Create a ProfileShape with a B-spline basis.

        Args:
            coeffs (Array): Coefficients for the B-spline basis.
            normalize (bool, optional): Whether to normalize the profile shape. Defaults to True.

        Returns:
            ProfileShape: A ProfileShape with the specified B-spline basis and coefficients.
        """
        n_splines = coeffs.size
        basis = BSplineBasis(n_comps=n_splines)

        return cls(basis=basis, coeffs=coeffs, normalize=normalize)

    @classmethod
    def make_points(cls, points: Array, grid: Array, normalize: bool = True) -> "ProfileShape":
        """Create a ProfileShape with points on the grid. To evaluate the profile shape on an arbitrary grid, we use interpolation.

        Args:
            points (Array): points defining the profile shape.
            grid (Array): grid points at which the profile shape is defined.
            normalize (bool, optional): Whether to normalize the profile shape. Defaults to True.

        Returns:
            ProfileShape: A ProfileShape with the specified points and grid.
        """
        assert points.size == grid.size
        # In this case, the "coeffs" are just the points on the grid.
        basis = InterpedLinearBasis(
            grid=grid,
        )
        return cls(basis=basis, coeffs=points, normalize=normalize)


@chex.dataclass
class Inputs:
    R0: float  # Major radius [m]
    B0: float  # On-axis toroidal field [T]
    Ip: float  # Plasma current [MA]
    a_minor: float  # Minor radius [m]
    kappa: float  # Elongation [-]
    delta: float  # Triangularity [-]
    Paux: float  # Auxiliary heating power [MW]
    ne20_line_avg: float  # Electron density [10^20 m^-3]
    Wtot_MJ: float  # Thermal energy [MJ]
    rho: Array  # Toroidal flux coordinate to evaluate the profiles at
    ne_edge: Optional[float] = None  # Edge electron density [10^20 m^-3]

    @property
    def epsilon(self):
        return self.a_minor / self.R0

    @property
    def fGW(self):
        greenwald_limit = self.Ip / (jnp.pi * self.a_minor**2)
        return self.ne20_line_avg / greenwald_limit

    @property
    def volume_approx(self):
        return calc_plasma_volume(
            major_radius=self.R0,
            inverse_aspect_ratio=self.epsilon,
            areal_elongation=self.kappa,
        )

    @property
    def beta_t(self):
        """Calculate toroidal beta in percentages."""
        average_pressure = (2.0 / 3.0) * self.Wtot_MJ * 1e6 / self.volume_approx
        beta_t = 100.0 * (average_pressure / (self.B0**2 / (2.0 * mu_0)))
        return beta_t

    @property
    def te_approx(self):
        pressure_MPa = (2.0 / 3.0) * self.Wtot_MJ / self.volume_approx
        pressure_Pa = pressure_MPa * 1e6
        pressure_eV = pressure_Pa / eV
        pressure_keV20 = pressure_eV / 1e3 / 1e20
        temp_keV = pressure_keV20 / self.ne20_line_avg
        return temp_keV

    @property
    def q_star(self):
        f_shaping = calc_f_shaping(self.epsilon, self.kappa, self.delta)
        return calc_q_star(
            magnetic_field_on_axis=self.B0,
            major_radius=self.R0,
            inverse_aspect_ratio=self.epsilon,
            plasma_current=self.Ip,
            f_shaping=f_shaping,
        )

    @property
    def nn_inputs(self):
        """An incomplete attempt at having maximally device-independent normalized inputs."""
        ne_edge = self.ne_edge if self.ne_edge is not None else 0.0
        inp_array = jnp.array([self.B0, self.q_star, self.epsilon, self.kappa, self.delta, self.Paux, self.fGW, self.beta_t, ne_edge])
        return inp_array


@chex.dataclass
class Outputs:
    ne: xr.DataArray  # Electron density profile [10^20 m^-3]
    te: xr.DataArray  # Electron temperature profile [keV]
    debug_info: Optional[dict] = None


class ShapeType(IntEnum):
    PCA_LIKE = 0
    CONVEX_COMBINATION = 1


def kmeans_initial_guess(n_shapes: int, te_data: xr.DataArray, ne_data: xr.DataArray, sample_dim: str, seed: int = 0):
    from sklearn.cluster import KMeans

    te_data = te_data.transpose(sample_dim, ...)
    ne_data = ne_data.transpose(sample_dim, ...)

    te_kmeans = KMeans(n_clusters=n_shapes, random_state=seed).fit(te_data.values)

    ne_kmeans = KMeans(n_clusters=n_shapes, random_state=seed).fit(ne_data.values)

    te_shapes = [ProfileShape.make_points(points=te_kmeans.cluster_centers_[i], grid=te_data.rho.values) for i in range(n_shapes)]
    ne_shapes = [ProfileShape.make_points(points=ne_kmeans.cluster_centers_[i], grid=ne_data.rho.values) for i in range(n_shapes)]
    return te_shapes, ne_shapes


def pca_initial_guess(n_shapes: int, te_data: xr.DataArray, ne_data: xr.DataArray, sample_dim: str):
    from xeofs.single import EOF

    te_eof = EOF(n_modes=n_shapes)
    te_eof.fit(te_data, dim=sample_dim)
    te_components = te_eof.components()
    te_shapes = [
        ProfileShape.make_points(points=te_components.sel(mode=i).values, grid=te_data.rho.values, normalize=False)
        for i in te_components.mode.values
    ]
    ne_eof = EOF(n_modes=n_shapes)
    ne_eof.fit(ne_data, dim=sample_dim)
    ne_components = ne_eof.components()
    ne_shapes = [
        ProfileShape.make_points(points=ne_components.sel(mode=i).values, grid=ne_data.rho.values, normalize=False)
        for i in ne_components.mode.values
    ]
    return te_shapes, ne_shapes


class ProfilePredictor(eqx.Module):
    te_shapes: list[ProfileShape]
    ne_shapes: list[ProfileShape]

    nn: RtdMLP
    shape_type: ShapeType = eqx.field(static=True)
    softmax_temp: float = eqx.field(static=True, default=1.0)
    use_ne_edge: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        te_shapes: list[ProfileShape],
        ne_shapes: list[ProfileShape],
        nn_width: int,
        nn_depth: int,
        softmax_temp: float,
        shape_type: ShapeType,
        use_ne_edge: bool,
        key: jax.random.PRNGKey,
    ):
        self.te_shapes = te_shapes
        self.ne_shapes = ne_shapes

        key, subkey = jax.random.split(key)
        self.nn = RtdMLP(
            in_size=9,
            out_size=len(te_shapes) + len(ne_shapes) + 1,
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            final_activation=Activation.IDENTITY,
            key=subkey,
        )
        self.softmax_temp = softmax_temp
        self.shape_type = shape_type
        self.use_ne_edge = use_ne_edge

    def __call__(self, inputs: Inputs, debug: bool = False) -> Outputs:
        nn_inputs = inputs.nn_inputs

        # Predict the coefficients for the shapes and the correction factor.
        coeffs = self.nn(nn_inputs)
        te_coeffs = coeffs[: len(self.te_shapes)]
        ne_coeffs = coeffs[len(self.te_shapes) : -1]
        te_correction = jnp.abs(coeffs[-1])

        if self.shape_type == ShapeType.CONVEX_COMBINATION:
            te_coeffs = jax.nn.softmax(te_coeffs / self.softmax_temp)
            ne_coeffs = jax.nn.softmax(ne_coeffs / self.softmax_temp)
        elif self.shape_type == ShapeType.PCA_LIKE:
            # NN outputs are already ready to be used as coefficients.
            pass
        else:
            raise ValueError(f"Invalid shape type: {self.shape_type}")

        # Compute the shapes.
        ne_shapes = jnp.stack([w * shape(inputs.rho) for shape, w in zip(self.ne_shapes, ne_coeffs)], axis=0)
        te_shapes = jnp.stack([w * shape(inputs.rho) for shape, w in zip(self.te_shapes, te_coeffs)], axis=0)

        # Compute the ne profile. If we are using the edge density as an input, we subtract out the predicted edge density and add the input edge density.
        ne = jnp.sum(ne_shapes, axis=0) * inputs.ne20_line_avg
        if self.use_ne_edge:
            ne = ne - ne[-1] + inputs.ne_edge

        # Compute the te profile using the learned correction.
        te = jnp.sum(te_shapes, axis=0) * inputs.te_approx * te_correction

        if debug:
            debug_info = {
                "te_coeffs": te_coeffs,
                "ne_coeffs": ne_coeffs,
                "te_correction": te_correction,
            }
        else:
            debug_info = None

        out = Outputs(
            ne=xr.DataArray(data=ne, dims=("rho",), coords={"rho": ("rho", inputs.rho)}),
            te=xr.DataArray(data=te, dims=("rho",), coords={"rho": ("rho", inputs.rho)}),
            debug_info=debug_info,
        )

        return out

    @classmethod
    def init(
        cls,
        dl: DataLoader,
        te_shape_var: str,
        ne_shape_var: str,
        n_shapes: int,
        nn_width: int,
        nn_depth: int,
        shape_type: ShapeType,
        softmax_temp: float,
        use_ne_edge: bool,
        key: jax.random.PRNGKey,
    ) -> "ProfilePredictor":
        ds = dl.ds
        sample_dim = dl.dataset.training_metadata.sample_dim

        if shape_type == ShapeType.PCA_LIKE:
            te_shapes, ne_shapes = pca_initial_guess(n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim)
        elif shape_type == ShapeType.CONVEX_COMBINATION:
            te_shapes, ne_shapes = kmeans_initial_guess(n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim)
        else:
            raise ValueError(f"Invalid shape type: {shape_type}")

        return cls(
            te_shapes=te_shapes,
            ne_shapes=ne_shapes,
            nn_width=nn_width,
            nn_depth=nn_depth,
            softmax_temp=softmax_temp,
            shape_type=shape_type,
            use_ne_edge=use_ne_edge,
            key=key,
        )

    @classmethod
    def load_latest_sparc(cls):
        from popsim.modules.profile_predictor.train import get_training_objs
        from popsim.modules.profile_predictor.train_configs import SPARC_CONFIG

        trainer, train_dl, val_dl, test_dl = get_training_objs(SPARC_CONFIG)
        trainer.restore_best_checkpoint()
        return trainer, train_dl, val_dl, test_dl


class EvalEnv(eqx.Module):
    profile_predictor: ProfilePredictor
    rhogrid: np.ndarray = eqx.field(static=True)

    def __init__(self, profile_predictor, rhogrid: Array):
        self.profile_predictor = profile_predictor
        self.rhogrid = np.asarray(rhogrid)

    def make_input(self, inputs):
        return Inputs(
            R0=inputs["R0"],
            B0=inputs["B0"],
            Ip=inputs["Ip_MA"],
            a_minor=inputs["a_minor"],
            kappa=inputs["kappa"],
            delta=inputs["delta"],
            Paux=inputs["Paux_MW"],
            ne20_line_avg=inputs["ne20_line_avg"],
            Wtot_MJ=inputs["Wtot_MJ"],
            rho=self.rhogrid,
            ne_edge=inputs["ne20_edge"],
        )

    def __call__(self, inputs):
        inputs = self.make_input(inputs)

        outputs = self.profile_predictor(inputs, debug=True)
        return outputs

    def get_trainable(self, freeze_shapes: bool = False):
        if freeze_shapes:
            # Get all leaves that are not a part of te_shapes and ne_shapes.
            # All of these leaves are trainable.
            ids_of_shape_leaves = [id(x) for x in jax.tree.leaves((self.profile_predictor.te_shapes, self.profile_predictor.ne_shapes))]
            return [x for x in jax.tree.leaves(self.profile_predictor) if id(x) not in ids_of_shape_leaves]
        else:
            return self.profile_predictor
