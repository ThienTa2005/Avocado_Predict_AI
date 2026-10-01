# AVOCADO — Website dự đoán giá

Website Streamlit độc lập, sử dụng model RNN PyTorch và Keras đã huấn luyện. Có sẵn dữ liệu mẫu, tải CSV, biểu đồ, dự đoán một bước và xuất kết quả CSV.

## Đưa lên GitHub

Tạo một repo riêng, ví dụ `avocado-forecast`. Upload **toàn bộ nội dung bên trong thư mục này** vào gốc repo. Sau upload, `app.py`, `requirements.txt` và `render.yaml` phải nằm ngay ở gốc repo, cùng với `data/`, `model/` và `.streamlit/`.

Không upload thư mục cha chứa notebook. Giữ đủ model và metadata đã đi kèm.

## Deploy dễ nhất: Render

1. Đăng nhập [Render](https://dashboard.render.com/) và kết nối GitHub.
2. Chọn **New → Blueprint**, chọn repo vừa tạo.
3. Render đọc `render.yaml`. Kiểm tra dịch vụ `avocado-forecast` và chọn **Deploy Blueprint**.
4. Chờ build và trạng thái **Live**, rồi mở URL `https://...onrender.com`.
5. Bấm **Dự đoán bước tiếp theo** để kiểm tra model hoạt động.

Cấu hình đang dùng gói Free. Dịch vụ Free có thể ngủ khi không sử dụng, vì vậy lần mở lại có thể chậm. Giới hạn tài nguyên/gói dịch vụ do Render quyết định; nếu log báo hết bộ nhớ, đổi sang gói có nhiều RAM hơn.

Nếu tạo bằng **New → Web Service**:
- Root Directory: để trống khi nội dung thư mục này ở gốc repo.
- Runtime: Python.
- Build Command: `pip install -r requirements.txt`
- Start Command: `python start.py`
- Environment: `PYTHON_VERSION=3.11.11`
- Health Check Path: `/_stcore/health`

Mỗi repo tạo một dịch vụ Render riêng. Website chạy trọn vẹn trên Render, không cần thêm Vercel hay một backend khác.

## Chạy trên máy

Dùng Python 3.11 trở lên; cấu hình deploy dùng 3.11.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python start.py
```

Mở http://localhost:8501. Nếu chạy cả hai website cùng lúc, đặt `$env:PORT="8502"` trong terminal của website thứ hai trước khi chạy.

## CSV đầu vào

Cột bắt buộc: `Date,AveragePrice,region,type`. Cần ít nhất 12 quan sát cho vùng và loại bơ đã chọn. Các nhóm phải có trong model; có thể tải CSV mẫu ngay trong website.

Ngày dùng định dạng YYYY-MM-DD. Giá phải là số không âm, hữu hạn; không có ngày trùng trong cùng chuỗi. Tệp tối đa 10 MB. CSV tải lên chỉ được xử lý trong phiên sử dụng.

Dữ liệu mẫu là dữ liệu lịch sử; ngày quan sát cuối được hiển thị rõ trên trang. Dự đoán một bước tiếp theo từ cửa sổ đầu vào, không phải dữ liệu thị trường trực tiếp. Model dùng scaler đã lưu từ tập train.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Kiểm thử đối chiếu dự đoán với notebook, kiểm tra CSV không hợp lệ và thao tác dự đoán trên giao diện.

## Docker (tùy chọn)

```sh
docker build -t avocado-forecast .
docker run --rm -p 8501:8501 avocado-forecast
```

## Tài nguyên

- `app.py`: giao diện.
- `inference.py`: kiểm tra dữ liệu và chạy RNN.
- `model/`: checkpoint, scaler và kết quả đánh giá.
- `data/prices.csv`: dữ liệu mẫu.
- `start.py`, `render.yaml`, `Dockerfile`: chạy và deploy.

Tham khảo: [Render Blueprints](https://render.com/docs/infrastructure-as-code), [Render Free](https://render.com/docs/free).


## Chọn và so sánh model

Chọn PyTorch, Keras hoặc So sánh cả hai ở thanh bên. Keras chạy bằng backend torch, không cần TensorFlow. Tỉ lệ dự đoán đúng là phần trăm mẫu kiểm thử có sai số tương đối trong ngưỡng chọn (mặc định 5%). Đây không phải xác suất dự báo tương lai đúng. Bảng và biểu đồ so sánh dùng cùng các mẫu kiểm thử của chuỗi đang chọn; có thể xuất CSV.

## Chọn cách dự đoán

Thanh bên có **Giá trực tiếp** và **Log return**, mỗi cách hỗ trợ PyTorch, Keras hoặc so sánh cả hai.
CSV tải lên vẫn chứa giá gốc. Kết quả dự đoán và bảng đánh giá sử dụng model/scaler của cách đang chọn; kết quả trả về USD.

Model giá trực tiếp nằm trong `model/`. Model log return nằm trong `model/log_return/`.
Sau khi chạy notebook log return tương ứng, chép các file sau từ thư mục lần chạy trong `artifacts/` vào `model/log_return/`:
`metadata.json`, `pytorch_best.pt`, `keras_best.keras`, `test_predictions.csv`, `model_comparison.csv`, `next_step_forecast.csv`.
Khởi động lại website sau khi thay checkpoint. Nếu chưa đủ file, lựa chọn Log return hiển thị thông báo chưa có model.
