"""PZFlow-based noisifier that samples photometric errors from a trained conditional flow."""

import pandas as pd
from ceci.config import StageParameter as Param
from pzflow import Flow
from rail.core.data import PqHandle
from rail.creation.noisifier import Noisifier
from rail.tools.flow_handle import FlowHandle
from collections import OrderedDict

_def_bands = ["u", "g", "r", "i", "z", "y"]
_def_mag_col_template = {band: f"mag_{band}_lsst" for band in _def_bands}
_def_error_col_map = {band: f"mag_{band}_lsst_err" for band in _def_bands}

class PZFlowNoisifier(Noisifier):
    """Noisifier that samples photometric errors from a trained conditional PZFlow model.

    The flow is expected to model p(mag_errors | magnitudes, ...), i.e. a conditional
    normalizing flow where the conditions are the ugrizy magnitudes and the outputs
    are the corresponding magnitude errors. The user can also add other conditions if any.

    The sampled errors are appended to the input catalog as new columns.
    """

    name = "PZFlowNoisifier"
    entrypoint_function = "__call__"
    inputs = [("model", FlowHandle), ("input", PqHandle)]

    config_options = Noisifier.config_options.copy()
    config_options.update(
        mag_col_template = Param(
            OrderedDict,
            _def_mag_col_template,
            msg=(
                "Mapping from magnitude-specific columns to data magnitude column name. "
                "Default: {'u': 'mag_u_lsst', ...}."
            )
        ),
        conditional_col_map=Param(
            dict,
            {},
            msg=(
                "Additional mapping from conditional columns in the model to data columns. "
                "Default: {'u_depth': 'u_depth', ...}."
            )
        ),
        error_col_map=Param(
            OrderedDict,
            _def_error_col_map,
            msg=(
                "Mapping from band to output error column name. "
                "Keys must be single-character band letters (e.g. 'u', 'g'). "
                "Default: {'u': 'mag_u_lsst_err', 'g': 'mag_g_lsst_err', ...}."
            ),
        ),
        decorrelate=Param(
            bool,
            True,
            msg=(
                "Wether decorrelate error with the degraded magnitudes."
                "If true, error column will be re-computed on the new magnitudes."
            )
        )
    )

    def _initNoiseModel(self) -> None:
        """Load the trained PZFlow conditional flow from the model handle."""
        self.noiseModel = self.open_model("model", **self.config)

    def _addNoise(self) -> None:
        """Sample photometric errors from the flow and attach them to the input catalog."""
        data = self.get_data("input")
        data_df = pd.DataFrame(data)

        flow_cols = self.noiseModel.conditional_columns
        conditional_cols = []
        for col in flow_cols:
            if col in (self.config.mag_col_template).keys():
                conditional_cols.append(self.config.mag_col_template[col])
            elif col in (self.config.conditional_col_map).keys():
                conditional_cols.append(self.config.conditional_col_map[col])
            else:
                raise ValueError(f"Column: {col} required by the model is not provided.")
        conditions = data_df[conditional_cols]

        samples = self.noiseModel.sample(
            nsamples=1,
            conditions=conditions,
            save_conditions=False,
            seed=self.config.seed,
        )

        # now add errors on the magnitude:
        for mag_col, mag_err_col in zip(self.config.mag_col_template.keys(), self.noiseModel.data_columns):
            band = self.config.mag_col_template[mag_col]
            data_df[band] = data_df[band] + samples[mag_err_col]

        # decorreation:
        if self.config.decorrelate == True:
            # update conditions
            conditions = data_df[conditional_cols]
            # recompute error
            samples = self.noiseModel.sample(
            nsamples=1,
            conditions=conditions,
            save_conditions=False,
            seed=self.config.seed,
        )
        
        # The flow's data_columns are assumed to correspond to bands in the same order
        # as self.config.bands. Rename them to the desired output column names.
        cols_to_keep = list(self.config.error_col_map.keys())
        samples = samples[cols_to_keep]
        samples = samples.rename(columns=self.config.error_col_map)
        samples.index = data_df.index
        output_df = pd.concat([data_df, samples], axis=1)
        self.add_data("output", output_df)
