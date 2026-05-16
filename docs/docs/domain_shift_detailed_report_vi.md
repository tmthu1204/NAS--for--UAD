# Báo Cáo Chi Tiết Về Thiết Lập Case Domain Shift Và Kết Quả Thí Nghiệm Trên SMD

## 1. Tóm tắt điều hành

Báo cáo này trình bày đầy đủ cách thiết lập một **case domain shift** trên SMD để kiểm tra giả thuyết:

> `adaptnas_combined` hữu ích hơn `uad_source` khi độ lệch miền giữa source và target đủ lớn.

Case domain shift được tạo bằng hướng:

- **Ben-David + JEPA latent PAD**

Cụ thể, một encoder TS-JEPA-style được dùng để học representation từ dữ liệu normal, sau đó tính `PAD_latent` giữa `source normal` và `target normal` để xếp hạng các pair cross-machine cùng family.

Thí nghiệm được chia thành 2 mức:

1. **Anchor case**: pair high-shift ban đầu `machine-1-1 -> machine-1-3`
2. **Extended matrix**: thêm 3 case để kiểm tra:
   - 1 low-shift control
   - 2 high-shift extra

Kết quả chính:

- Trên **nhóm high-shift**, `adaptnas_combined` nhìn chung tốt hơn `uad_source`
- Trên **low-shift control**, lợi ích adaptation giảm đi, thậm chí `uad_source` tốt hơn ở một số metric
- `Benefit_gap` theo `AUROC` giữa high-shift và low-shift là:
  - `+0.2094`

Điều này ủng hộ lập luận:

> adaptation mang lại lợi ích lớn hơn khi source-target mismatch thực sự mạnh.

Tuy nhiên, kết quả cũng cho thấy:

- `NAS_BestArch` **không phải lúc nào** cũng là winner trong combined mode
- ở một số case high-shift, **baseline cố định** vẫn có thể thắng NAS

Vì vậy, kết luận mạnh nhất hiện tại là:

> `adaptnas_combined` có xu hướng vượt `uad_source` rõ hơn ở high-shift cases, nhưng độ ổn định của NAS selection vẫn cần được kiểm chứng thêm bằng multi-seed.

## 2. Bối cảnh và câu hỏi nghiên cứu

### 2.1. Bài toán thực tế

Trong bài toán UAD thực tế:

- mô hình được train trên **source machine**
- sau đó deploy sang **target machine không nhãn**
- source và target thường thuộc cùng system family
- nhưng khác nhau về:
  - machine identity
  - operating condition
  - calibration
  - environment
  - measurement time
  - aging

Do đó, vấn đề cốt lõi không chỉ là:

- mô hình có hoạt động trên target hay không

mà là:

- khi nào adaptation thực sự cần thiết và có lợi

### 2.2. Câu hỏi cần trả lời

Báo cáo này tập trung trả lời 3 câu hỏi:

1. Có thể thiết lập một **case domain shift có kiểm soát** trên SMD hay không?
2. Trong case có domain shift lớn, `adaptnas_combined` có tốt hơn `uad_source` hay không?
3. Lợi ích này có giảm đi khi chuyển sang case low-shift hay không?

## 3. Nền tảng phương pháp

### 3.1. Cơ sở lý thuyết

Thiết kế này dựa trên 2 lớp ý tưởng:

1. **Ben-David domain adaptation theory**
2. **JEPA-style representation learning**

Theo Ben-David, target risk bị ảnh hưởng bởi:

- source risk
- divergence giữa source và target
- sự khác biệt của labeling function

Trong thiết kế hiện tại, thành phần được khai thác trực tiếp là:

- **distribution divergence**

### 3.2. View đang đo domain shift

Metric chính không đo trên raw signal theo kiểu `mean/std/min/max` nữa, mà đo trên:

- **representation view**
- chính xác hơn là **latent separability view**

Ý nghĩa:

- nếu source và target dễ bị tách biệt trong latent space
- thì domain shift được xem là mạnh hơn

### 3.3. Công thức thực tế đang dùng

Pipeline chọn pair làm như sau:

1. train một **global TS-JEPA-style encoder**
2. lấy latent của `source normal` và `target normal`
3. train domain classifier trên latent
4. tính:
   - `PAD_latent`

`PAD_latent` càng lớn thì pair càng có domain shift mạnh theo representation view.

### 3.4. Vì sao dùng JEPA + PAD

Thiết kế này mạnh hơn dùng riêng JEPA loss vì:

- JEPA giúp học representation giàu cấu trúc
- PAD vẫn là metric divergence chính bám sát tinh thần Ben-David hơn

Nói ngắn gọn:

- **JEPA**: representation backbone
- **PAD**: domain-shift metric chính

### 3.5. Bảng so sánh `TS-JEPA` hiện tại và `V-JEPA` chuẩn

| `TS-JEPA` hiện tại trong repo | `V-JEPA` chuẩn |
|---|---|
| Là một **JEPA-style model cho time series** | Là một **JEPA model cho video** |
| Mục tiêu chính trong repo này là **học latent để tính `PAD_latent`** | Mục tiêu chính là **học visual representation tổng quát cho downstream video/image tasks** |
| Input là **window time series** kích thước `T x C` | Input là **video clip** với cấu trúc không gian - thời gian |
| Encoder hiện tại là **1D CNN encoder** dựa trên `EncoderCNN` | Encoder chuẩn trong V-JEPA là **video transformer / ViT-style backbone** |
| Predictor hiện tại là **BiGRU + Linear + LayerNorm** | Predictor chuẩn của V-JEPA là **predictor trong latent space cho video tokens** |
| Masking hiện tại là **mask theo đoạn thời gian 1D** | Masking chuẩn là **mask các spatio-temporal target regions / tubelets** |
| `online_encoder` nhận bản input bị mask bằng cách **zero-out các time positions bị che** | Context encoder trong V-JEPA nhận **visible context tokens** theo chiến lược mask của video |
| `target_encoder` là **bản EMA/frozen copy** của encoder online | `target_encoder` của V-JEPA cũng là **target branch ổn định** để cung cấp latent mục tiêu |
| Loss hiện tại là **MSE trên latent của các vị trí bị mask** | Loss chuẩn của V-JEPA là **feature prediction loss trong latent space trên target regions** |
| Không reconstruct raw signal/pixel | Không reconstruct pixel |
| Không nhằm mô phỏng đầy đủ pipeline V-JEPA chính thức | Là implementation chính thức của Meta cho bài toán video |
| Nên được gọi chính xác là **“V-JEPA-inspired”** hoặc **“JEPA-style encoder for time series”** | Có thể gọi trực tiếp là **V-JEPA** |

Kết luận ngắn:

- `TS-JEPA` hiện tại **giống V-JEPA ở nguyên lý học bằng dự đoán latent từ ngữ cảnh**
- nhưng **khác đáng kể về modality, kiến trúc, masking và mục tiêu sử dụng**
- vì vậy trong báo cáo, cách gọi an toàn nhất là:
  - **TS-JEPA là một JEPA-style time-series adaptation lấy cảm hứng từ V-JEPA**

## 4. Dữ liệu và tiền xử lý

### 4.1. Raw dataset

Raw SMD được dùng từ:

- [external/OmniAnomaly/ServerMachineDataset](/D:/Papers/NAS--for--UAD/external/OmniAnomaly/ServerMachineDataset)

### 4.2. Processed cache

Flow `default_nasade` của repo không chạy trực tiếp trên raw text, mà qua các file `.npz`:

- `source.npz`
- `target.npz`

Mỗi machine được preprocess thành:

- `X`: tập window
- `y`: label nhị phân trên từng window

Setting preprocess dùng xuyên suốt:

- `window = 128`
- `stride = 64`
- normalization: `train_zscore`

### 4.3. Lệnh preprocess

Ví dụ:

```powershell
.\venv\Scripts\python.exe scripts\preprocess_smd.py --raw_root data/ServerMachineDataset --out_root data/smd --machine machine-1-1 --window 128 --stride 64
```

Repo hiện đã có resolver path, nên `data/ServerMachineDataset` sẽ tự fallback sang raw SMD trong `external`.

## 5. Cách dựng case domain shift

### 5.1. Pair ranking

Pair được chọn từ ranking:

- [pilot_rankings.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/jepa_pad_pilot/pilot_rankings.json:1)

Quy tắc:

- chỉ xét **same-family directed pairs**
- dùng `source normal vs target normal`
- rank theo `PAD_latent` giảm dần

### 5.2. Split protocol

Với mỗi pair, split được dựng bằng:

- `train_normal` từ source
- `target_pool_unlabeled` từ target
- `val_mixed` từ target
- `test_mixed` từ target

Setting split cố định:

- `split_mode = search`
- `shift_level = hard`
- `target_pool_frac = 0.2`
- `val_frac = 0.3`
- `guard = 4`

### 5.3. Pair anchor ban đầu

Pair high-shift đầu tiên dùng để kiểm tra hypothesis là:

- `machine-1-1 -> machine-1-3`

Split metadata:

- [split_metadata.json](/D:/Papers/NAS--for--UAD/data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/split_metadata.json:1)

## 6. Thiết lập thí nghiệm

### 6.1. Setting chung

Toàn bộ thí nghiệm dùng cùng outer budget:

- `epochs_pretrain = 10`
- `search_candidates = 5`
- `batch_size = 64`
- `device = cuda`

Đây là budget được dùng cho bản report chính.

### 6.2. Hai mode cần so sánh

#### `uad_source`

- chỉ dùng `train_normal`
- search objective là compactness trên source normal
- không dùng target pool trong optimization

#### `adaptnas_combined`

- dùng cả:
  - `train_normal`
  - `target_pool_unlabeled`
- search theo combined unlabeled objective
- final stage đánh giá cả:
  - `Base_CNN_GRU`
  - `Base_CNN_TCN`
  - `Base_CNN_TRF`
  - `NAS_BestArch`

Lưu ý quan trọng:

- top-level result của combined mode lấy từ:
  - `best_by_auroc`
- nên winner cuối cùng có thể là:
  - `NAS_BestArch`
  - hoặc một baseline cố định

## 7. Kết quả anchor pair

### 7.1. Pair `machine-1-1 -> machine-1-3`

Artifact:

- [uad_source_results.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/uad_source_results.json:1)
- [adaptnas_combined_results.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/adaptnas_combined_results.json:1)
- [adaptnas_combined_baselines_summary.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/top_pair_compare_m1_1_to_m1_3_paper_grade/adaptnas_combined_baselines_summary.json:1)

So sánh `uad_source` vs `adaptnas_combined`:

| Metric | `uad_source` | `adaptnas_combined` | Delta |
|---|---:|---:|---:|
| AUROC | 0.5582 | 0.7158 | +0.1576 |
| AP | 0.2911 | 0.3076 | +0.0165 |
| F1_best | 0.3125 | 0.5143 | +0.2018 |
| F1_pot | 0.3030 | 0.4000 | +0.0970 |
| Event_F1 | 0.5714 | 0.7500 | +0.1786 |
| Delay_mean | 0.5000 | 2.0000 | -1.5000 |

Kết luận:

- `adaptnas_combined` tốt hơn rõ ràng trên các metric phát hiện chính
- nhưng chậm hơn về detection delay

### 7.2. So sánh nội bộ combined mode

| Model | AUROC | AP | F1_best | F1_pot | Event_F1 | Delay_mean |
|---|---:|---:|---:|---:|---:|---:|
| `Base_CNN_GRU` | 0.5162 | 0.1752 | 0.2963 | 0.2162 | 0.4000 | 0.6667 |
| `Base_CNN_TCN` | 0.4610 | 0.2582 | 0.2791 | 0.1600 | 0.4000 | 1.0000 |
| `Base_CNN_TRF` | 0.3247 | 0.0986 | 0.2240 | 0.1982 | 0.6667 | 0.0000 |
| `NAS_BestArch` | 0.7158 | 0.3076 | 0.5143 | 0.4000 | 0.7500 | 2.0000 |

Kết luận:

- trên pair này, `NAS_BestArch` là winner thật sự
- điều này xác nhận rằng với budget đủ mạnh, NAS có thể vượt các baseline cố định trong case high-shift

## 8. Bộ thí nghiệm mở rộng

### 8.1. Lý do mở rộng

Nếu chỉ báo cáo một pair, claim mới dừng ở mức:

- có một case high-shift mà `adaptnas_combined` thắng

Để claim mạnh hơn, cần kiểm tra thêm:

- một low-shift control
- thêm high-shift cases khác

### 8.2. 4 case được dùng

| Case ID | Vai trò | Pair | `PAD_latent` pilot |
|---|---|---|---:|
| `H1` | high-shift anchor | `machine-1-1 -> machine-1-3` | 2.0000 |
| `L1` | low-shift control | `machine-1-3 -> machine-1-7` | 1.5847 |
| `H2` | high-shift extra | `machine-2-1 -> machine-2-4` | 2.0000 |
| `H3` | high-shift extra | `machine-1-1 -> machine-1-4` | 2.0000 |

Lưu ý:

- ban đầu dự kiến dùng một pair `machine-3-*`
- nhưng pair đó không dựng được split hợp lệ với protocol hiện tại
- vì vậy `H3` được thay bằng một high-shift pair khác vẫn thỏa protocol

### 8.3. Script automation

Flow chạy và tổng hợp được đóng gói trong:

- [run_domain_shift_case_matrix.py](/D:/Papers/NAS--for--UAD/scripts/run_domain_shift_case_matrix.py:1)

Command đã dùng:

```powershell
.\venv\Scripts\python.exe scripts\run_domain_shift_case_matrix.py `
  --cases L1,H2,H3 `
  --include_anchor `
  --seeds 42 `
  --device cuda
```

Artifact tổng hợp:

- [summary.md](/D:/Papers/NAS--for--UAD/outputs/benchmarks/domain_shift_matrix/summary.md:1)
- [summary.json](/D:/Papers/NAS--for--UAD/outputs/benchmarks/domain_shift_matrix/summary.json:1)
- [summary_rows.csv](/D:/Papers/NAS--for--UAD/outputs/benchmarks/domain_shift_matrix/summary_rows.csv:1)

## 9. Kết quả của 4 case

### 9.1. So sánh `uad_source` vs `adaptnas_combined`

| Case | Vai trò | Pair | PAD | Source AUROC | Combined AUROC | Delta AUROC | Source F1_best | Combined F1_best | Delta F1_best | Winner trong combined |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `H1` | high-shift | `machine-1-1 -> machine-1-3` | 2.0000 | 0.5582 | 0.7158 | +0.1576 | 0.3125 | 0.5143 | +0.2018 | `NAS_BestArch` |
| `H2` | high-shift | `machine-2-1 -> machine-2-4` | 2.0000 | 0.5296 | 0.5712 | +0.0416 | 0.3333 | 0.3750 | +0.0417 | `Base_CNN_TRF` |
| `H3` | high-shift | `machine-1-1 -> machine-1-4` | 2.0000 | 0.5438 | 0.7484 | +0.2046 | 0.4000 | 0.6486 | +0.2486 | `NAS_BestArch` |
| `L1` | low-shift | `machine-1-3 -> machine-1-7` | 1.5847 | 0.8273 | 0.7526 | -0.0747 | 0.4000 | 0.6667 | +0.2667 | `NAS_BestArch` |

### 9.2. Quan sát chính

#### Quan sát 1: high-shift group nhìn chung ủng hộ adaptation

Trong cả 3 high-shift case:

- `H1`: combined thắng rõ
- `H2`: combined chỉ thắng nhẹ
- `H3`: combined thắng rõ

Mean trên nhóm high-shift:

- `Delta AUROC = +0.1346`
- `Delta F1_best = +0.1640`
- `Delta Event_F1 = +0.1373`

#### Quan sát 2: low-shift control đi ngược lại một phần

Trên `L1`:

- `uad_source` tốt hơn theo `AUROC`
- `uad_source` tốt hơn theo `Event_F1`
- combined chỉ hơn rõ ở `F1_best`

Điều này rất quan trọng vì nó cho thấy:

- adaptation **không phải lúc nào cũng thắng**
- lợi ích của adaptation có liên hệ với mức shift

#### Quan sát 3: `Benefit_gap` dương

Từ `summary.md`:

- `high-shift mean Delta AUROC = +0.1346`
- `low-shift mean Delta AUROC = -0.0747`
- `Benefit_gap = +0.2094`

Đây là tín hiệu định lượng mạnh nhất của báo cáo này.

### 9.3. Quan sát về NAS vs baseline

#### `H1`

- winner: `NAS_BestArch`
- combined thắng source rõ

#### `H2`

- winner: `Base_CNN_TRF`
- `NAS_BestArch` lại có `AUROC = 0.1955`, thấp hơn rất nhiều baseline mạnh nhất

Điều này cho thấy:

- high-shift không đồng nghĩa NAS luôn thắng
- search objective hiện tại vẫn có thể chọn kiến trúc không tối ưu cho test metric cuối cùng

#### `H3`

- winner: `NAS_BestArch`
- combined thắng source mạnh

#### `L1`

- winner trong combined vẫn là `NAS_BestArch`
- nhưng combined vẫn thua source theo `AUROC` và `Event_F1`

Điều này cho thấy:

- kể cả khi NAS là winner trong combined block
- adaptation-level result vẫn chưa chắc hơn source-only

## 10. Diễn giải khoa học

### 10.1. Claim nào được hỗ trợ

Kết quả hiện tại hỗ trợ khá tốt claim sau:

> `adaptnas_combined` có xu hướng mang lại lợi ích lớn hơn `uad_source` khi domain shift lớn.

### 10.2. Claim nào chưa nên nói quá mạnh

Chưa nên nói quá mạnh rằng:

> `adaptnas_combined` luôn tốt hơn `uad_source`

vì low-shift control `L1` không ủng hộ kết luận đó.

Cũng chưa nên nói quá mạnh rằng:

> `NAS_BestArch` luôn tốt hơn mọi baseline trong combined mode

vì `H2` là phản ví dụ rõ ràng.

### 10.3. Cách diễn đạt an toàn trong báo cáo

Một cách viết phù hợp là:

> Trên các pair high-shift được chọn bằng `PAD_latent`, `adaptnas_combined` nhìn chung vượt `uad_source` theo AUROC và event-level metrics, trong khi ở low-shift control lợi ích này giảm đi hoặc mất hẳn. Điều này cho thấy adaptation đặc biệt hữu ích khi source-target mismatch đủ lớn. Tuy nhiên, lợi ích của NAS bên trong combined mode chưa hoàn toàn ổn định trên mọi high-shift case.

## 11. Hạn chế của kết quả hiện tại

### 11.1. Mới dùng một seed

Hiện tại matrix này mới dùng:

- `seed = 42`

Do đó chưa thể báo cáo:

- `mean ± std` theo seed

### 11.2. Chỉ có một low-shift control

`L1` đã rất hữu ích, nhưng vẫn chỉ là một control pair.

### 11.3. H3 là pair fallback

Một pair `machine-3-*` ban đầu không dựng được split hợp lệ, nên `H3` hiện tại là:

- `machine-1-1 -> machine-1-4`

Điều này không làm hỏng lập luận chính, nhưng cần nêu rõ để minh bạch protocol.

### 11.4. Delay vẫn là trade-off

Ở nhiều case, combined có thể tốt hơn về AUROC/F1 nhưng:

- không tốt hơn về delay

Điều này cần được nêu rõ nếu bài của bạn coi detection timeliness là quan trọng.

## 12. Khuyến nghị cho bước tiếp theo

Để chuyển báo cáo này thành claim mạnh hơn cho paper, nên làm tiếp:

1. chạy thêm `seed = 43, 44`
2. giữ nguyên 4 case hiện tại để có `mean ± std`
3. nếu còn compute:
   - thêm 1 low-shift control nữa
   - thêm 1 high-shift pair ở family khác nếu dựng split được

Thứ tự ưu tiên tôi khuyên:

1. multi-seed trước
2. rồi mới thêm pair mới

## 13. Kết luận cuối cùng

Báo cáo này cho thấy quy trình tạo case domain shift bằng:

- **JEPA latent PAD**

là đủ hiệu quả để xây dựng một benchmark so sánh `adaptnas_combined` với `uad_source`.

Kết quả thực nghiệm hiện tại cho thấy:

- trên high-shift cases, `adaptnas_combined` thường có lợi hơn
- trên low-shift control, lợi ích này giảm đi
- `Benefit_gap = +0.2094` theo AUROC là bằng chứng định lượng quan trọng nhất

Đồng thời, báo cáo cũng chỉ ra một điểm rất quan trọng:

- lợi ích của **adaptation-level method** rõ hơn lợi ích của **NAS-level architecture selection**

Nói cách khác:

- combined mode có xu hướng hữu ích khi shift lớn
- nhưng NAS bên trong combined vẫn cần được đánh giá ổn định hơn bằng multi-seed để đưa ra claim mạnh ở mức architecture search
