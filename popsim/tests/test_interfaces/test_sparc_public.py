from popsim.interfaces import sparc_public


def test_load_transp():
    ds = sparc_public.load_prd_transp_profiles()

    assert "Te_keV" in ds.data_vars
    assert "ne19" in ds.data_vars
    assert "ne20" in ds.data_vars
    assert "Ti_keV" in ds.data_vars
    assert "q" in ds.data_vars
    assert "polflux" in ds.data_vars
    assert "rho" in ds.coords
