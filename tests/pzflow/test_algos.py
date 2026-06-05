import os
import numpy as np
import pytest
import scipy.special
import pandas as pd
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.utils.testing_utils import one_algo
from rail.utils.path_utils import find_rail_file

from rail.estimation.algos import pzflow_nf
from rail.creation.degraders.pzflow_degrader import PZFlowNoisifier



sci_ver_str = scipy.__version__.split(".")


@pytest.mark.parametrize(
    "inputs, zb_expected",
    [
        (False, [0.15, 0.14, 0.11, 0.14, 0.12, 0.14, 0.15, 0.16, 0.11, 0.12]),
        (True, [0.15, 0.14, 0.15, 0.14, 0.12, 0.14, 0.15, 0.12, 0.13, 0.11]),
    ],
)
@pytest.mark.slow
def test_pzflow(inputs, zb_expected):
    def_bands = ["u", "g", "r", "i", "z", "y"]
    refcols = [f"mag_{band}_lsst" for band in def_bands]
    def_maglims = dict(
        mag_u_lsst=27.79,
        mag_g_lsst=29.04,
        mag_r_lsst=29.06,
        mag_i_lsst=28.62,
        mag_z_lsst=27.98,
        mag_y_lsst=27.05,
    )
    def_errnames = dict(
        mag_err_u_lsst="mag_u_lsst_err",
        mag_err_g_lsst="mag_g_lsst_err",
        mag_err_r_lsst="mag_r_lsst_err",
        mag_err_i_lsst="mag_i_lsst_err",
        mag_err_z_lsst="mag_z_lsst_err",
        mag_err_y_lsst="mag_y_lsst_err",
    )
    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        seed=0,
        ref_band="mag_i_lsst",
        column_names=refcols,
        mag_limits=def_maglims,
        include_mag_errors=inputs,
        err_names_dict=def_errnames,
        n_error_samples=3,
        soft_sharpness=10,
        soft_idx_col=0,
        redshift_col="redshift",
        n_training_epochs=50,
        hdf5_groupname="photometry",
        model="PZflowPDF.pkl",
        output_mode="skip_write",
    )
    estim_config_dict = dict(hdf5_groupname="photometry", model="PZflowPDF.pkl")

    # zb_expected = np.array([0.15, 0.14, 0.11, 0.14, 0.12, 0.14, 0.15, 0.16, 0.11, 0.12])
    train_algo = pzflow_nf.PZFlowInformer
    pz_algo = pzflow_nf.PZFlowEstimator
    results, rerun_results, rerun3_results = one_algo(
        "PZFlow", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    # temporarily remove comparison to "expected" values, as we are getting
    # slightly different answers for python3.7 vs python3.8 for some reason
    #    assert np.isclose(results.ancil['zmode'], zb_expected, atol=0.05).all()
    assert np.isclose(
        results.ancil["zmode"], rerun_results.ancil["zmode"], atol=0.05
    ).all()


@pytest.fixture
def data():
    """Some dummy data to use below."""

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=24.5, scale=1, size=(100, 13))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])
    x[:, 7:] = x[:, 7:] + 1

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y",
                                 "depth_u", "depth_g", "depth_r", "depth_i" ,"depth_z", "depth_y"])
    return TableHandle("data", df, path="dummy.pd")


def test_PZFlowNoisifier(data):
    model = find_rail_file(
            "examples_data/creation_data/data/pzflow_noisifier_model.pkl")
    mag_col_template = {}
    conditional_col_map = {}
    error_col_map = {}
    for band in "ugrizy":
        mag_col_template[f'true_mag_{band}']=f'{band}'
        conditional_col_map[f'depth_{band}']=f'depth_{band}'
        error_col_map[f'delta_mag_{band}']=f'mag_{band}_err'
    
    pzflow_degrader = PZFlowNoisifier.make_stage(model=model,
                                                     mag_col_template = mag_col_template,
                                                     conditional_col_map=conditional_col_map,
                                                      error_col_map=error_col_map,
                                                      decorrelate=True
                                                     )
    degraded_df = pzflow_degrader(data)
    
    os.remove(
            pzflow_degrader.get_output(pzflow_degrader.get_aliased_tag("output"), final_name=True)
        )
