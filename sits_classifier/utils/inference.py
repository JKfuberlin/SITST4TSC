from typing import Optional, Any, List, Union, Tuple
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, date
from re import compile, Pattern
import numpy as np
from enum import Enum
from time import time

class ModelType(Enum):
    LSTM        = 1
    TRANSFORMER = 2
    UNDEFINED   = 3


OUTPUT_NODATA: int = 255

def pad_doy_sequence(target: int, observations: List[datetime]) -> List[Union[datetime, float]]:
    diff: int = target - len(observations)
    if diff < 0:
        observations = observations[abs(diff):]  # deletes oldest entries first
    elif diff > 0:
        observations = observations + ([0.0] * diff)

    # TODO remove assertion for "production"
    assert(target == len(observations))

    return observations
    

def pad_datacube(target: int, datacube: np.ndarray) -> np.ndarray:
    diff: int = target - datacube.shape[0]
    if diff < 0:
        datacube = np.delete(datacube, list(range(abs(diff))), axis=0)  # deletes oldest entries first
    elif diff > 0:
        datacube = np.pad(datacube, ((0, diff), (0,0), (0,0), (0,0)))
    
    # TODO remove assertion for "production"
    assert(target == datacube.shape[0])

    return datacube


def fp_to_doy(file_path: str) -> datetime:
    date_in_fp: Pattern = compile(r"(?<=/)\d{8}(?=_)")
    sensing_date: str = date_in_fp.findall(file_path)[0]
    d: datetime = datetime.strptime(sensing_date,"%Y%m%d")
    doy: int = d.toordinal() - date(d.year, 1, 1).toordinal() + 1  # https://docs.python.org/3/library/datetime.html#datetime.datetime.timetuple
    return doy


@torch.inference_mode()
def predict(model, data: torch.tensor, it: ModelType) -> Any:
    """
    Apply previously trained model to new data
    :param model: previously trained model
    :param torch.tensor data: new input data
    :return Any: Array of predictions
    """
    outputs = model(data)
    _, predicted = torch.max(outputs.data if it == ModelType.LSTM else outputs, 1)
    return predicted


def predict_lstm(lstm: torch.nn.LSTM, dc: torch.tensor, mask: Optional[np.ndarray], c_step: int, r_step: int) -> torch.tensor:
    prediction: torch.Tensor = torch.full((r_step * c_step,), fill_value=OUTPUT_NODATA, dtype=torch.long)
    if mask:
        merged_row: torch.Tensor = torch.full(c_step, fill_value=OUTPUT_NODATA, dtype=torch.long)
        for chunk_rows in range(0, r_step):
            merged_row.zero_()
            squeezed_row: torch.Tensor = predict(
                lstm,
                dc[chunk_rows, mask[chunk_rows]],
                ModelType.LSTM)
            merged_row[mask[chunk_rows]] = squeezed_row
            prediction[chunk_rows, 0:c_step] = merged_row
    else:
        for chunk_rows in range(0, r_step):
            prediction[chunk_rows, 0:c_step] = predict(lstm, dc[chunk_rows], ModelType.LSTM)
    
    return prediction


@torch.inference_mode()
def predict_transformer(transformer: torch.nn.Transformer, dc: torch.tensor, mask: Optional[np.ndarray], c_step: int, r_step: int, batch_size: int) -> torch.tensor:
    rows, _ = dc.shape[:2]
    split_dc = [torch.squeeze(i, dim=0) for i in torch.vsplit(dc, rows)]
    # slightly slower approach but easier to capture output
    ds: TensorDataset = TensorDataset(torch.cat(split_dc, 0))  # TensorDataset splits along first dimension of input
    dl: DataLoader = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=4, persistent_workers=True)
    prediction: torch.Tensor = torch.full((r_step * c_step,), fill_value=OUTPUT_NODATA, dtype=torch.long)

    if mask is not None:
        mask_long: torch.Tensor = torch.from_numpy(np.reshape(mask, (-1,))).bool()
        for batch_index, batch in enumerate(dl):
            for _, samples in enumerate(batch):
                start: int = batch_index * batch_size
                end: int = start + len(samples)
                subset: torch.Tensor = mask_long[start:end]
                if not torch.any(subset):
                    next
                input_tensor: torch.Tensor = samples[subset].cuda(non_blocking=True)  # ordering of subsetting and moving makes little to no difference time-wise but big difference memory-wise
                prediction[start:end][subset] = predict(transformer, input_tensor, ModelType.TRANSFORMER).cpu()
    else:
        for batch_index, batch in enumerate(dl):
            for _, samples in enumerate(batch):
                start: int = batch_index * batch_size
                end: int = start + len(samples)
                prediction[start:end] = predict(transformer, samples.cuda(non_blocking=True), ModelType.TRANSFORMER).cpu()

    return torch.reshape(prediction, (r_step, c_step))

