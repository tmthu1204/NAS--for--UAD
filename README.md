# NAS-ADE: Chapter 4 reproducibility package

Repo này là gói source code nộp kèm khóa luận, dùng để tái lập các thí nghiệm
NAS-ADE trong Chapter 4. Gói đã chứa đúng 28 cặp benchmark tiền xử lý của SMD,
MSL và SMAP; không cần tải raw dataset.

## Chạy nhanh

Yêu cầu Python 3.12. Sau khi tạo và activate môi trường ảo:

```powershell
python -m pip install -r requirements.txt
python run_chapter4.py --check-only
python run_chapter4.py --device auto
```

`--device auto` dùng CUDA khi PyTorch nhận GPU, nếu không sẽ dùng CPU. Thí nghiệm
đầy đủ có thể mất nhiều giờ. Để kiểm tra luồng chương trình bằng một cặp và cấu
hình một epoch:

```powershell
python run_chapter4.py --dataset smd --device cpu --quick
```

`--quick` không tạo kết quả dùng trong luận văn.

## Nội dung chính

- `run_chapter4.py`: lệnh chạy thống nhất cho SMD, MSL và SMAP.
- `data/chapter4/`: 12 + 6 + 10 cặp nguồn-đích đã tiền xử lý.
- `src/`: TS-TCC, AdaptNAS/NAS-ADE, DeepSVDD và các kiến trúc đối chứng.
- `scripts/run_manifest_benchmarks.py`: chạy hai mode và sinh báo cáo từng cặp.
- `reference_results/chapter4/`: bảng kết quả đã báo cáo trong khóa luận.
- `HuongDanCaiDat.txt`: hướng dẫn cài đặt chi tiết.
- `HuongDanSuDung.txt`: lệnh chạy, tiếp tục và đọc kết quả.

Kết quả mới được ghi vào `outputs/chapter4/` và không được Git theo dõi. Xem
`REPORT.md` và `summary.json` trong thư mục của từng bộ dữ liệu để đọc kết quả.
