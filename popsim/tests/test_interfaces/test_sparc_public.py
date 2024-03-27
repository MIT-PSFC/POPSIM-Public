import popsim.interfaces.sparc_public as sparc_public

def test_load_transp():
    ds = sparc_public.load_prd_transp()

    assert "te" in ds.data_vars
    assert "ne" in ds.data_vars
    assert "ti" in ds.data_vars
    assert "q" in ds.data_vars
    assert "polflux" in ds.data_vars
    assert "rho" in ds.coords