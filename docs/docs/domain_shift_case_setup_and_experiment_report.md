# Thiết Lập Case Domain Shift và Báo Cáo Thí Nghiệm SMD

## 1. Mục tiêu

Tài liệu này mô tả đầy đủ cách:

1. tạo một **case domain shift** trên SMD,
2. chọn **cross-machine pair có độ shift lớn**,
3. dựng split thí nghiệm đúng flow của repo,
4. chạy và so sánh:
   - `adaptnas_combined`
   - `uad_source`
   - các `baselines` trong combined mode
5. tổng hợp kết quả để báo cáo.

Tài liệu này bám đúng những gì đã chạy trong workspace hiện tại.

## 2. Ý tưởng phương pháp

### 2.1. Bài toán thực tế

Ta muốn mô phỏng setting:

- train trên source machine,
- deploy sang target machine không nhãn,
- source và target thuộc cùng family,
- nhưng khác về máy, điều kiện vận hành, hiệu chuẩn, môi trường, thời điểm đo, hoặc aging.

Do đó, cần chọn một pair cross-machine có **domain shift đủ lớn** để kiểm tra khi nào `adaptnas_combined` thực sự hữu ích hơn `uad_source`.

### 2.2. Cách tạo case domain shift

Case domain shift được tạo theo 2 bước:

1. **Chọn pair source-target có shift lớn**
2. **Dựng split cross-machine từ dữ liệu thật**

### 2.3. Metric chọn pair

Pair được chọn bằng hướng:

- **Ben-David + JEPA latent PAD**

Cụ thể:

1. train một **global TS-JEPA-style encoder** trên source-normal windows,
2. extract latent features cho từng machine,
3. tính `PAD_latent` giữa:
   - `source normal`
   - `target normal`
4. rank **same-family directed pairs** theo `PAD_latent` giảm dần.

Trong implementation hiện tại:

- metric chính là `PAD_latent`
- target protocol là `source normal vs target normal`
- pair scope là `same-family only`

Tham chiếu:

- [ben_david_jepa_pad_method_spec.md](/D:/Papers/NAS--for--UAD/docs/docs/ben_david_jepa_pad_method_spec.md:1)
- [ben_david_jepa_pad_implementation_spec.md](/D:/Papers/NAS--for--UAD/docs/docs/ben_david_jepa_pad_implementation_spec.md:1)

## 3. Pair được chọn cho thí nghiệm

Từ pilot ranking JEPA+PAD, pair top highest shift được chọn là:

- `machine-1-1 -> machine-1-3`

Ranking pilot được lưu ở:

- [pilot_rankings.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/jepa_pad_pilot/pilot_rankings.json:1)

## 4. Chuẩn bị dữ liệu

### 4.1. Raw SMD

Raw SMD đang được lưu ở:

- [external/OmniAnomaly/ServerMachineDataset](/D:/Papers/NAS--for--UAD/external/OmniAnomaly/ServerMachineDataset)

Repo hiện tại chạy `default_nasade` qua các file windowed `.npz`, nên cần tạo cache:

- `data/smd/machine-1-1/source.npz`
- `data/smd/machine-1-1/target.npz`
- `data/smd/machine-1-3/source.npz`
- `data/smd/machine-1-3/target.npz`

### 4.2. Lệnh preprocess

```powershell
.\venv\Scripts\python.exe scripts\preprocess_smd.py --raw_root data/ServerMachineDataset --out_root data/smd --machine machine-1-1 --window 128 --stride 64
.\venv\Scripts\python.exe scripts\preprocess_smd.py --raw_root data/ServerMachineDataset --out_root data/smd --machine machine-1-3 --window 128 --stride 64
```

Lưu ý:

- `data/ServerMachineDataset` ở đây chỉ là path đầu vào logic.
- resolver của repo sẽ tự fallback sang `external/OmniAnomaly/ServerMachineDataset`.

### 4.3. Dựng split cross-machine

```powershell
.\venv\Scripts\python.exe scripts\make_uad_smd.py `
  --machine_dir data/smd/machine-1-1 `
  --target_machine_dir data/smd/machine-1-3 `
  --split_mode search `
  --shift_level hard `
  --target_pool_frac 0.2 `
  --val_frac 0.3 `
  --guard 4 `
  --out_dir data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3
```

Metadata split:

- [split_metadata.json](/D:/Papers/NAS--for--UAD/data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/split_metadata.json:1)

Thông tin chính của split:

- source machine: `machine-1-1`
- target machine: `machine-1-3`
- `train_normal_count = 443`
- `target_pool_count = 73`
- `target_pool_hidden_anomaly_ratio = 0.0685`
- `val_count = 48`
- `test_count = 111`

## 5. Thiết lập thí nghiệm

## 5.1. Budget dùng để báo cáo

Để tránh kết luận bị ảnh hưởng bởi budget pilot quá nhỏ, run báo cáo dùng:

- `epochs_pretrain = 10`
- `search_candidates = 5`
- `batch_size = 64`
- `device = cuda`

Tôi gọi đây là **paper-grade rerun** trong phạm vi repo hiện tại.

### 5.2. Lệnh chạy `uad_source`

```powershell
.\venv\Scripts\python.exe -m src.pipeline `
  --dataset_or_paths data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/train_normal.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/val_mixed.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/test_mixed.npz `
  --mode uad_source `
  --family default_nasade `
  --epochs_pretrain 10 `
  --search_candidates 5 `
  --batch_size 64 `
  --device cuda
```

Kết quả được lưu ở:

- [uad_source_results.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/uad_source_results.json:1)

### 5.3. Lệnh chạy `adaptnas_combined`

```powershell
.\venv\Scripts\python.exe -m src.pipeline `
  --dataset_or_paths data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/train_normal.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/target_pool_unlabeled.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/val_mixed.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/test_mixed.npz `
  --mode adaptnas_combined `
  --family default_nasade `
  --epochs_pretrain 10 `
  --search_candidates 5 `
  --batch_size 64 `
  --device cuda
```

Kết quả được lưu ở:

- [adaptnas_combined_results.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/adaptnas_combined_results.json:1)
- [adaptnas_combined_baselines_summary.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/adaptnas_combined_baselines_summary.json:1)

## 6. Lưu ý quan trọng khi đọc kết quả

Trong `adaptnas_combined`, repo hiện tại làm 2 việc:

1. search ra một `NAS_BestArch`
2. chạy final-only cho:
   - `Base_CNN_GRU`
   - `Base_CNN_TCN`
   - `Base_CNN_TRF`
   - `NAS_BestArch`

Sau đó top-level `metrics_uad` của combined mode lấy từ:

- `best_by_auroc` trong `baselines_summary`

Nghĩa là:

- nếu `NAS_BestArch` tốt nhất theo AUROC, top-level result sẽ là NAS
- nếu một baseline tốt hơn, top-level result sẽ là baseline đó

Trong run báo cáo này:

- `best_by_auroc = NAS_BestArch`

## 7. Kết quả so sánh

### 7.1. So sánh `uad_source` vs `adaptnas_combined`

| Metric | `uad_source` | `adaptnas_combined` | Delta |
|---|---:|---:|---:|
| AUROC | 0.5582 | 0.7158 | +0.1576 |
| AP | 0.2911 | 0.3076 | +0.0165 |
| F1_best | 0.3125 | 0.5143 | +0.2018 |
| F1_pot | 0.3030 | 0.4000 | +0.0970 |
| Precision_best | 0.2778 | 0.4286 | +0.1508 |
| Recall_best | 0.3571 | 0.6429 | +0.2857 |
| Precision_pot | 0.2632 | 0.3750 | +0.1118 |
| Recall_pot | 0.3571 | 0.4286 | +0.0714 |
| Event_F1 | 0.5714 | 0.7500 | +0.1786 |
| Event_Precision | 0.6667 | 0.7500 | +0.0833 |
| Event_Recall | 0.5000 | 0.7500 | +0.2500 |
| Delay_mean | 0.5000 | 2.0000 | -1.5000 |
| Delay_median | 0.5000 | 1.0000 | -0.5000 |

Kết luận:

- `adaptnas_combined` thắng `uad_source` trên gần như toàn bộ metric phát hiện chính
- nhưng `uad_source` có delay tốt hơn trong run này

### 7.2. So sánh các baseline nội bộ trong `adaptnas_combined`

| Model | AUROC | AP | F1_best | F1_pot | Event_F1 | Delay_mean |
|---|---:|---:|---:|---:|---:|---:|
| `Base_CNN_GRU` | 0.5162 | 0.1752 | 0.2963 | 0.2162 | 0.4000 | 0.6667 |
| `Base_CNN_TCN` | 0.4610 | 0.2582 | 0.2791 | 0.1600 | 0.4000 | 1.0000 |
| `Base_CNN_TRF` | 0.3247 | 0.0986 | 0.2240 | 0.1982 | 0.6667 | 0.0000 |
| `NAS_BestArch` | 0.7158 | 0.3076 | 0.5143 | 0.4000 | 0.7500 | 2.0000 |

Kết luận:

- với budget report này, `NAS_BestArch` là model tốt nhất theo:
  - `AUROC`
  - `AP`
  - `F1_best`
  - `F1_pot`
  - `Event_F1`
- nhưng `NAS_BestArch` không tốt nhất về delay

### 7.3. So sánh trực tiếp `Base_CNN_GRU` vs `NAS_BestArch`

| Metric | `Base_CNN_GRU` | `NAS_BestArch` | Delta |
|---|---:|---:|---:|
| AUROC | 0.5162 | 0.7158 | +0.1996 |
| AP | 0.1752 | 0.3076 | +0.1325 |
| F1_best | 0.2963 | 0.5143 | +0.2180 |
| F1_pot | 0.2162 | 0.4000 | +0.1838 |
| Precision_best | 0.3077 | 0.4286 | +0.1209 |
| Recall_best | 0.2857 | 0.6429 | +0.3571 |
| Precision_pot | 0.1333 | 0.3750 | +0.2417 |
| Recall_pot | 0.5714 | 0.4286 | -0.1429 |
| Event_F1 | 0.4000 | 0.7500 | +0.3500 |
| Event_Precision | 0.2727 | 0.7500 | +0.4773 |
| Event_Recall | 0.7500 | 0.7500 | +0.0000 |
| Delay_mean | 0.6667 | 2.0000 | -1.3333 |

Kết luận:

- `NAS_BestArch` vượt `Base_CNN_GRU` rõ rệt ở hầu hết metric chất lượng phát hiện
- `Base_CNN_GRU` vẫn có lợi thế về tốc độ phát hiện trung bình

## 8. Câu kết luận dùng để báo cáo

Có thể viết ngắn gọn như sau:

> Chúng tôi tạo case domain shift bằng cách chọn pair cross-machine cùng family có `PAD_latent` lớn nhất trong SMD, sau đó dựng split cross-machine thật từ dữ liệu gốc. Trên pair `machine-1-1 -> machine-1-3`, với budget tìm kiếm mạnh hơn, `adaptnas_combined` vượt `uad_source` theo AUROC, AP, F1 và event-level metrics. Đồng thời, trong nội bộ combined mode, `NAS_BestArch` cũng vượt các baseline cố định theo AUROC và các metric phát hiện chính, dù vẫn có trade-off về detection delay.

## 9. Caveat cần nêu khi báo cáo

1. Đây là kết quả trên **một top-shift pair**.
2. `uad_source` và `adaptnas_combined` dùng cùng outer budget, nhưng khác protocol nội tại theo thiết kế method.
3. `adaptnas_combined` vẫn có trade-off về delay.
4. Để claim mạnh hơn trong paper, nên chạy thêm:
   - 1 low-shift control pair
   - thêm 1-2 seed
   - thêm 1-2 high-shift pairs khác

## 10. Thiết kế bộ thí nghiệm mở rộng

Mục tiêu của bộ chạy mở rộng là giữ **nguyên setting** của pair đã chạy trước đó, chỉ thay:

- `source machine`
- `target machine`
- `seed`

Như vậy, mọi khác biệt trong kết quả sẽ dễ diễn giải hơn.

### 10.1. Giữ nguyên toàn bộ setting

Các run mở rộng nên giữ đúng các hyperparameter sau:

- window preprocess: `128`
- stride preprocess: `64`
- split mode: `search`
- shift level: `hard`
- `target_pool_frac = 0.2`
- `val_frac = 0.3`
- `guard = 4`
- family model: `default_nasade`
- `epochs_pretrain = 10`
- `search_candidates = 5`
- `batch_size = 64`
- `device = cuda`

Nói ngắn gọn: đây là cùng setting với pair:

- `machine-1-1 -> machine-1-3`

### 10.2. Chọn 3 case bổ sung

| Case ID | Vai trò | Pair | `PAD_latent` pilot | Lý do chọn |
|---|---|---|---:|---|
| `H1` | high-shift anchor | `machine-1-1 -> machine-1-3` | `2.0000` | pair đã chạy, dùng làm mốc so sánh |
| `L1` | low-shift control | `machine-1-3 -> machine-1-7` | `1.5847` | thấp nhất trong `machine-1-*`, giúp tạo control cùng family với `H1` |
| `H2` | high-shift extra | `machine-2-1 -> machine-2-4` | `2.0000` | kiểm tra tính khái quát ở family `machine-2-*` |
| `H3` | high-shift extra | `machine-1-1 -> machine-1-4` | `2.0000` | high-shift pair khác nhưng vẫn dựng được split hợp lệ với protocol hiện tại |

Lưu ý quan trọng:

- không chọn low-shift pair theo kiểu giữ nguyên source `machine-1-1`, vì các target thấp nhất của `machine-1-1` vẫn có `PAD_latent` khá cao:
  - `machine-1-1 -> machine-1-8`: `1.9167`
  - `machine-1-1 -> machine-1-2`: `1.9220`
- để có một control đủ "thấp shift", cần dùng:
  - `machine-1-3 -> machine-1-7`

### 10.3. Thiết kế seed

Để so sánh dễ với run hiện có, dùng bộ seed:

- `42`: seed đã dùng cho pair gốc
- `43`: seed bổ sung 1
- `44`: seed bổ sung 2

### 10.4. Matrix chạy đề xuất

#### Phương án đầy đủ

Chạy cả 4 pair với 3 seed:

| Pair set | Seeds | Modes | Số run |
|---|---|---|---:|
| `H1`, `L1`, `H2`, `H3` | `42, 43, 44` | `uad_source`, `adaptnas_combined` | `4 x 3 x 2 = 24` |

Ghi chú:

- mỗi run `adaptnas_combined` đã tự sinh block so sánh nội bộ:
  - `Base_CNN_GRU`
  - `Base_CNN_TCN`
  - `Base_CNN_TRF`
  - `NAS_BestArch`
- vì vậy không cần chạy thêm lệnh baseline riêng để so sánh trong combined mode

#### Phương án tiết kiệm compute

Nếu compute hạn chế, chạy theo 2 giai đoạn:

1. Giai đoạn A, phủ đủ các case:
   - `H1`, `L1`, `H2`, `H3`
   - chỉ seed `42`
   - tổng `4 x 1 x 2 = 8` run
2. Giai đoạn B, kiểm tra ổn định:
   - thêm seed `43`, `44` cho `H1` và `L1`
   - tổng thêm `2 x 2 x 2 = 8` run

Tổng cộng:

- `16` run

Phương án này vẫn đủ tốt để trả lời 2 câu chính:

- lợi ích adaptation có mạnh hơn ở high-shift so với low-shift không?
- kết quả ở pair gốc có ổn định theo seed không?

### 10.5. Máy cần preprocess

Để chạy bộ mở rộng này, cần preprocess các machine sau:

- `machine-1-1`
- `machine-1-3`
- `machine-1-7`
- `machine-2-1`
- `machine-2-4`
- `machine-1-4`

### 10.6. Quy ước đặt tên output

Để bảng kết quả dễ gom, nên đặt output theo format:

- `outputs/benchmarks/domain_shift_matrix/<pair_id>/seed_<seed>/<mode>_results.json`

Ví dụ:

- `outputs/benchmarks/domain_shift_matrix/H1_m1_1_to_m1_3/seed_42/uad_source_results.json`
- `outputs/benchmarks/domain_shift_matrix/H1_m1_1_to_m1_3/seed_42/adaptnas_combined_results.json`

### 10.7. Cách tổng hợp để báo cáo

Sau khi chạy xong, nên tổng hợp theo 3 bảng:

#### Bảng A: `uad_source` vs `adaptnas_combined` theo từng pair

Giữ các metric:

- `AUROC`
- `AP`
- `F1_best`
- `F1_pot`
- `Event_F1`
- `Delay_mean`

Tính thêm:

- `Delta_AUROC = AUROC_combined - AUROC_source`
- `Delta_F1_best = F1_best_combined - F1_best_source`
- `Delta_Event_F1 = Event_F1_combined - Event_F1_source`

#### Bảng B: so sánh nội bộ trong `adaptnas_combined`

Cho từng pair và seed, lấy:

- `Base_CNN_GRU`
- `Base_CNN_TCN`
- `Base_CNN_TRF`
- `NAS_BestArch`

Mục tiêu là kiểm tra:

- NAS có còn là best model khi đổi pair không
- baseline nào mạnh nhất nếu NAS không thắng

#### Bảng C: tổng hợp theo nhóm shift

Chia thành:

- `high-shift group = {H1, H2, H3}`
- `low-shift group = {L1}`

Với mỗi metric chính, báo cáo:

- mean ± std trên seed
- mean trên nhóm high-shift
- mean trên nhóm low-shift

Nên tính thêm một chỉ số kết luận:

- `Benefit_gap = mean(Delta_AUROC_high) - mean(Delta_AUROC_low)`

Nếu `Benefit_gap > 0`, đó là bằng chứng gọn và mạnh rằng:

- adaptation giúp nhiều hơn khi domain shift lớn

### 10.8. Kết luận thực nghiệm mong đợi

Nếu hypothesis của method đúng, bạn kỳ vọng:

1. `H1`, `H2`, `H3`:
   - `adaptnas_combined` thường vượt `uad_source` rõ hơn
2. `L1`:
   - khoảng cách giữa 2 mode nhỏ hơn
   - thậm chí `uad_source` có thể gần bằng hoặc thắng ở một vài metric
3. trong combined mode:
   - `NAS_BestArch` thường thắng baseline ở high-shift cases khi budget đủ mạnh

Nếu pattern này xuất hiện, claim trong paper sẽ mạnh hơn rất nhiều so với chỉ báo cáo một pair đơn lẻ.

### 10.9. Script automation đã dùng

Flow chạy và tổng hợp cho bộ case này đã được đóng gói vào:

- [run_domain_shift_case_matrix.py](/D:/Papers/NAS--for--UAD/scripts/run_domain_shift_case_matrix.py:1)

Ví dụ lệnh chạy 3 case mới và ghép luôn anchor cũ:

```powershell
.\venv\Scripts\python.exe scripts\run_domain_shift_case_matrix.py `
  --cases L1,H2,H3 `
  --include_anchor `
  --seeds 42 `
  --device cuda
```

Artifact tổng hợp được ghi ra:

- `outputs/benchmarks/domain_shift_matrix/summary.json`
- `outputs/benchmarks/domain_shift_matrix/summary_rows.csv`
- `outputs/benchmarks/domain_shift_matrix/summary.md`

## 11. File kết quả dùng trong báo cáo

- Pair selection:
  - [pilot_rankings.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/jepa_pad_pilot/pilot_rankings.json:1)
- Split:
  - [split_metadata.json](/D:/Papers/NAS--for--UAD/data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/split_metadata.json:1)
- Final report runs:
  - [uad_source_results.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/uad_source_results.json:1)
  - [adaptnas_combined_results.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/adaptnas_combined_results.json:1)
  - [adaptnas_combined_baselines_summary.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/adaptnas_combined_baselines_summary.json:1)
