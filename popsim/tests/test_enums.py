import cfspopcon.named_options as cfsno
import popsim.enums as popsim_enums


def test_enums():
    impurity_cfspopcon = cfsno.Impurity.Argon
    impurity_popsim = popsim_enums.Impurity.Argon

    cfs_to_popsim = popsim_enums.Impurity(impurity_cfspopcon.value)

    profile_form_cfspopcon = cfsno.ProfileForm.analytic
    profile_form_popsim = popsim_enums.ProfileForm.analytic

    assert impurity_cfspopcon.value == impurity_popsim.value
    assert impurity_popsim == cfs_to_popsim
    assert profile_form_popsim.value == profile_form_cfspopcon.value


def test_jax_jit_enums():
    """Test that IntEnum instances can be compiled with jax.jit."""
    from enum import IntEnum

    import jax
    import jax.numpy as jnp

    class MyEnum(IntEnum):
        A = 1
        B = 2

    # Function that uses this enum.
    def f(enum, x):
        return enum + x

    enum = MyEnum(1)
    out = jax.jit(f)(enum, 2.0)
    assert enum == MyEnum.A
    assert out == 3.0

    """
    Now test that we can vmap across an array of IntEnum.
    """
    intenums = jnp.array([MyEnum(1), MyEnum(2)])
    out = jax.jit(jax.vmap(f, in_axes=(0, None)))(intenums, 2.0)
    assert jnp.all(out == jnp.array([3.0, 4.0]))
