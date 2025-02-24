from popsim.tests.fixtures import profile_predictor_latest_sparc
from popsim.ml._export import export
import tempfile
from pathlib import Path

def test_export(profile_predictor_latest_sparc):
    trainer, _, _, test_dl = profile_predictor_latest_sparc

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        export(trainer.train_state.model, tmpdir, test_dl)
        assert (tmpdir / "model.json").exists()
        assert (tmpdir / "validation_data.nc").exists()