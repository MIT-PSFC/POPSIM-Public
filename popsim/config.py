"""Package-wide configuration settings for POPSIM."""

config = {
    "DIFFRAX_MAX_STEPS": 100_000_000,  # We want a large, but not infinite number of steps as an infinite number of steps can cause the simulation to hang.
}
