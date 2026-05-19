"""PZFlow-based noisifier that samples photometric errors from a trained conditional flow."""

import pandas as pd
from ceci.config import StageParameter as Param

from rail.core.data import PqHandle
from rail.creation.noisifier import Noisifier
from rail.tools.flow_handle import FlowHandle

_def_bands = ["u", "g", "r", "i", "z", "y"]
_def_mag_col_template = "mag_{band}_lsst"
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
        bands=Param(
            list,
            _def_bands,
            msg="Ordered list of photometric band letters (must match the order the flow was trained on).",
        ),
        mag_col_template=Param(
            str,
            _def_mag_col_template,
            msg="Template for magnitude column names in the input catalog. Use {band} as placeholder.",
        ),
        additional_cols=Param(
            list,
            [],
            msg="Additional columns used as conditions, e.g. 5-sigma depth, must be in the same order as the model.",
        ),
        error_col_map=Param(
            dict,s
            _def_error_col_map,
            msg=(
                "Mapping from band letter to output error column name. "
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

        bands = self.config.bands
        mag_cols = [self.config.mag_col_template.format(band=b) for b in bands]
        if len(self.config.additional_cols)>0:
            mag_cols += self.config.additional_cols
        conditions = data_df[mag_cols]

        samples = self.noiseModel.sample(
            nsamples=1,
            conditions=conditions,
            save_conditions=False,
            seed=self.config.seed,
        )

        # now add errors on the magnitude:
        flow_cols = self.noiseModel.data_columns
        for band, model_col in zip(bands, flow_cols):
            data_df[f'mag_{band}_lsst'] += samples[model_col]

        # decorreation:
        if self.config.decorrelate == True:
            # update conditions
            conditions = data_df[mag_cols]
            # recompute error
            samples = self.noiseModel.sample(
            nsamples=1,
            conditions=conditions,
            save_conditions=False,
            seed=self.config.seed,
        )
        
        # The flow's data_columns are assumed to correspond to bands in the same order
        # as self.config.bands. Rename them to the desired output column names.
        rename_map = {
            flow_col: self.config.error_col_map[band]
            for flow_col, band in zip(flow_cols, bands)
        }
        samples = samples.rename(columns=rename_map)
        samples.index = data_df.index

        error_cols = [self.config.error_col_map[b] for b in bands]
        output_df = pd.concat([data_df, samples[error_cols]], axis=1)
        self.add_data("output", output_df)
