from pathlib import Path
import json
import pandas as pd
import streamlit as st
from inference import load_bundle, prepare_data, predict

ROOT = Path(__file__).resolve().parent
IS_AVOCADO = json.loads((ROOT / "model/metadata.json").read_text(encoding="utf-8"))["target"] == "AveragePrice"
TITLE = "Avocado" if IS_AVOCADO else "AMZN"
st.set_page_config(page_title=f"{TITLE} · Dự đoán giá", page_icon="🥑" if IS_AVOCADO else "📈", layout="wide")

@st.cache_resource
def cached_bundle():
    return load_bundle()

@st.cache_data
def read_sample():
    return pd.read_csv(ROOT / "data/prices.csv")

st.caption("RNN FORECAST · " + TITLE.upper())
st.title(f"Dự đoán giá {TITLE}")
st.write("Chọn dữ liệu, xem lịch sử và dự đoán giá ở bước tiếp theo bằng model RNN đã huấn luyện.")

try:
    bundle = cached_bundle()
except Exception:
    st.error("Không nạp được model. Kiểm tra model/pytorch_best.pt, metadata.json và thư viện đã cài.")
    st.stop()
metadata = bundle["metadata"]
target = metadata["target"]
lookback = metadata["lookback"]

with st.sidebar:
    st.header("Dữ liệu dự đoán")
    source = st.radio("Nguồn dữ liệu", ["Dữ liệu mẫu", "Tải CSV"])
    group = "AMZN"
    if IS_AVOCADO:
        groups = [json.loads(key) for key in metadata["scalers"]]
        regions = sorted({pair[0] for pair in groups})
        region = st.selectbox("Khu vực", regions, index=regions.index("TotalUS"))
        kinds = sorted({pair[1] for pair in groups if pair[0] == region})
        kind = st.selectbox("Loại bơ", kinds)
        group = json.dumps([region, kind], ensure_ascii=False)
    st.caption(f"Model sử dụng {lookback} quan sát gần nhất, theo thứ tự cũ → mới.")
    if source == "Tải CSV":
        upload = st.file_uploader("CSV của bạn (tối đa 10 MB)", type=["csv"])
        st.caption("Cột bắt buộc: " + ("Date, AveragePrice, region, type" if IS_AVOCADO else "Date, Close"))
    else:
        upload = None
    sample = read_sample()
    st.download_button("Tải CSV mẫu", sample.to_csv(index=False).encode("utf-8-sig"), f"{TITLE.lower()}-sample.csv", "text/csv")

if source == "Tải CSV" and upload is None:
    st.info("Tải một CSV ở thanh bên để bắt đầu.")
    st.stop()
try:
    if upload is not None and upload.size > 10 * 1024 * 1024:
        raise ValueError("Tệp vượt quá giới hạn 10 MB.")
    raw = pd.read_csv(upload) if upload is not None else sample
    frame = prepare_data(raw, metadata, group)
except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

window = frame.tail(lookback)
last = float(window[target].iloc[-1])
a, b, c = st.columns(3)
a.metric("Giá quan sát cuối", f"$ {last:,.4f}")
b.metric("Số quan sát", f"{len(frame):,}")
c.metric("Ngày quan sát cuối", frame.Date.iloc[-1].strftime("%d/%m/%Y"))
st.caption(f"Dữ liệu từ {frame.Date.iloc[0]:%d/%m/%Y} đến {frame.Date.iloc[-1]:%d/%m/%Y}. Dữ liệu mẫu là lịch sử, không cập nhật trực tiếp.")
st.subheader("Lịch sử giá")
st.line_chart(frame.set_index("Date")[[target]].rename(columns={target: "Giá (USD)"}), color="#3b9b76" if IS_AVOCADO else "#e99a26")
with st.expander(f"Xem {lookback} quan sát dùng để dự đoán"):
    st.dataframe(window[["Date", target]], hide_index=True, width="stretch")

if st.button("Dự đoán bước tiếp theo", type="primary"):
    try:
        result = predict(window[target].tolist(), bundle, group)
        st.metric("Giá dự đoán", f"$ {result:,.4f}", f"{result-last:+.4f} USD so với giá cuối")
        horizon = "Quan sát tiếp theo" if IS_AVOCADO else "Phiên giao dịch tiếp theo"
        if IS_AVOCADO and window.Date.diff().dropna().eq(pd.Timedelta(days=7)).all():
            horizon = "Tuần tiếp theo · " + (window.Date.iloc[-1] + pd.Timedelta(days=7)).strftime("%d/%m/%Y")
        st.caption(horizon + " · PyTorch RNN")
        output = pd.DataFrame([{"group": group, "last_observed_date": window.Date.iloc[-1].date(),
            "last_price_usd": last, "prediction_usd": result, "horizon": horizon}])
        st.download_button("Tải kết quả CSV", output.to_csv(index=False).encode("utf-8-sig"), f"{TITLE.lower()}-forecast.csv", "text/csv")
    except ValueError as exc:
        st.error(str(exc))

with st.expander("Đánh giá model trên dữ liệu kiểm thử"):
    st.caption("Kết quả từ lần huấn luyện đã lưu; CSV bạn tải lên không làm thay đổi bảng đánh giá.")
    st.dataframe(pd.read_csv(ROOT / "model/model_comparison.csv"), hide_index=True, width="stretch")
    evaluation = pd.read_csv(ROOT / "model/test_predictions.csv")
    evaluation = evaluation.loc[evaluation["group"] == group].copy()
    evaluation["Date"] = pd.to_datetime(evaluation["Date"])
    st.line_chart(evaluation.set_index("Date")[["actual", "PyTorch RNN"]].rename(columns={"actual": "Thực tế", "PyTorch RNN": "Dự đoán"}))

