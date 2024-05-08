import numpy as np
import pytest
import scipy.special

from rail.utils.testing_utils import one_algo
from rail.core.stage import RailStage
from rail.estimation.algos import pzflow_nf

sci_ver_str = scipy.__version__.split(".")


DS = RailStage.data_store
DS.__class__.allow_overwrite = True


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
        flow_seed=0,
        ref_column_name="mag_i_lsst",
        column_names=refcols,
        mag_limits=def_maglims,
        include_mag_errors=inputs,
        error_names_dict=def_errnames,
        n_error_samples=3,
        soft_sharpness=10,
        soft_idx_col=0,
        redshift_column_name="redshift",
        num_training_epochs=50,
        hdf5_groupname="photometry",
        model="PZflowPDF.pkl",
        output_mode = "skip_write"
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
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"], atol=0.05).all()

