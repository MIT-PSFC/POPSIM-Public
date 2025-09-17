from popsim.tests.fixtures import profile_predictor_latest_sparc
from popsim.ml._export import export
import tempfile
from pathlib import Path
import numpy as np
import json

def test_export(profile_predictor_latest_sparc):
    model, _, _, test_dl = profile_predictor_latest_sparc

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        export(model, tmpdir, test_dl)
        assert (tmpdir / "model.json").exists()
        assert (tmpdir / "eval_data.nc").exists()
    
        # Load the exported model in JSON format and spot-check some matrices.
        with open(tmpdir / "model.json", "r") as f:
            model_json = json.load(f)
            
            n_layers = len(model_json['model_parameters']['nn']['layers'])
            for i in range(n_layers):
                wj = np.asarray(model_json['model_parameters']['nn']['layers'][i]['weight'])
                wm = model.nn.layers[i].weight
                assert np.allclose(wj, wm), "First layer weights do not match!"
                
                bj = np.asarray(model_json['model_parameters']['nn']['layers'][i]['bias'])
                bm = model.nn.layers[i].bias
                assert np.allclose(bj, bm), "First layer biases do not match!"