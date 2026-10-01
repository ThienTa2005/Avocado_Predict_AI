"""Inference uses exactly the architecture and saved scalers from the notebook."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

class VanillaRNN(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, num_layers=1, nonlinearity="tanh", batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        sequence, _ = self.rnn(x)
        return self.head(self.dropout(sequence[:, -1, :]))

def load_bundle():
    metadata = json.loads((ROOT / "model/metadata.json").read_text(encoding="utf-8"))
    cfg = metadata["config"]
    torch.set_num_threads(1)
    model = VanillaRNN(cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(ROOT / "model/pytorch_best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return {"model": model, "metadata": metadata}

def prepare_data(frame, metadata, group):
    if group not in metadata["scalers"]:
        raise ValueError("Nhóm này chưa được model hỗ trợ.")
    target = metadata["target"]
    required = ["Date", target] + (["region", "type"] if target == "AveragePrice" else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("CSV thiếu cột: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    if target == "AveragePrice":
        region, kind = json.loads(group)
        frame = frame.loc[(frame.region == region) & (frame.type == kind)].copy()
    if len(frame) < metadata["lookback"]:
        raise ValueError(f"Cần ít nhất {metadata['lookback']} quan sát cho chuỗi đã chọn.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if frame.Date.isna().any():
        raise ValueError("CSV chứa ngày không hợp lệ.")
    if frame.Date.duplicated().any():
        raise ValueError("CSV có ngày trùng trong chuỗi đã chọn.")
    if frame[target].map(lambda x: isinstance(x, (bool, np.bool_))).any():
        raise ValueError("Giá phải là số, không phải True/False.")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    values = frame[target].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Giá phải là số không âm và hữu hạn.")
    return frame.sort_values("Date").reset_index(drop=True)

def predict(prices, bundle, group):
    metadata = bundle["metadata"]
    if group not in metadata["scalers"]:
        raise ValueError("Nhóm này chưa được model hỗ trợ.")
    if not isinstance(prices, (list, tuple, np.ndarray)) or any(isinstance(x, (bool, np.bool_)) for x in prices):
        raise ValueError("Cần một danh sách giá dạng số.")
    try:
        values = np.asarray(prices, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Giá phải là số.") from exc
    if values.shape != (metadata["lookback"],) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Cần đúng {metadata['lookback']} giá không âm và hữu hạn.")
    scaler = metadata["scalers"][group]
    z = ((values - scaler["mean"]) / scaler["scale"]).astype(np.float32).reshape(1, -1, 1)
    with torch.inference_mode():
        result = float(bundle["model"](torch.from_numpy(z)).item()) * scaler["scale"] + scaler["mean"]
    if not np.isfinite(result):
        raise ValueError("Model trả về giá trị không hữu hạn.")
    return result

