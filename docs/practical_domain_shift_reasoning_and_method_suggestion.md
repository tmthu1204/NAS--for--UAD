# Domain Shift Thực Tế cho NAS-based UAD: Lập luận, giao thức đo lường khuyến nghị, và danh mục đọc A/A*

## 1. Mục tiêu của tài liệu này

Tài liệu này được viết cho bối cảnh transfer thực tế của dự án:

- một detector hoặc detector được chọn bởi NAS được huấn luyện trên miền nguồn,
- sau đó được chuyển sang một miền đích không có nhãn,
- source và target thường thuộc cùng một họ hệ thống thay vì hai miền hoàn toàn xa nhau,
- nhưng target có thể khác do máy khác, chế độ vận hành khác, thời điểm khác, môi trường khác, hiệu chuẩn khác, bảo trì, hoặc lão hóa,
- và target pool là dữ liệu không nhãn, có thể chứa một lượng nhỏ anomaly bị ẩn.

Mục tiêu của tài liệu là đưa ra một câu trả lời vừa đủ chặt cho paper, vừa đủ rõ để triển khai, cho năm câu hỏi:

1. Bài toán này nên được mô tả như thế nào?
2. Khi nói “domain shift” thì nên đo cái gì?
3. View nào là chính, view nào là tùy chọn, view nào chỉ nên dùng để sanity check?
4. Những paper hội nghị A/A* nào hỗ trợ tốt nhất cho từng view?
5. Nên dùng các shift score như thế nào để hỗ trợ claim rằng `adaptnas_combined` có thể tốt hơn `uad_source` khi có mismatch giữa source và target?

Tài liệu này được viết rộng hơn SMD. Lập luận bên dưới được kỳ vọng vẫn đúng cho các bộ dữ liệu công nghiệp hoặc multivariate time series khác, chỉ thay đổi ở phần chi tiết theo từng dataset.

## 2. Cách đặc tả bài toán được khuyến nghị

### 2.1. Kết luận cốt lõi

Setting source-to-target thực tế nên được mô tả là:

**same-family deployment shift dưới temporal nonstationarity, trong đó hiệu ứng quan sát được nổi trội nhất là covariate shift, còn concept drift có thể có nhưng không được mặc định giả định**

Viết ngắn hơn:

**một deployment shift thực tế trong time series, biểu hiện chủ yếu dưới dạng covariate shift trong môi trường không dừng**

### 2.2. Nên khẳng định điều gì và không nên khẳng định điều gì

Khẳng định sau đây là phù hợp:

$$
P_s(X) \neq P_t(X)
$$

Đây là formulation chính vì source và target chủ yếu khác nhau ở phân phối đặc trưng, động học theo thời gian, và cấu trúc phụ thuộc giữa các biến.

Khẳng định sau đây có thể đúng, nhưng không nên giả định nếu chưa có bằng chứng:

$$
P_s(Y \mid X) \neq P_t(Y \mid X)
$$

Nói cách khác, concept drift có thể tồn tại, nhưng chỉ nên nêu ra khi có bằng chứng trực tiếp rằng ngữ nghĩa của normal và anomalous behavior đã thay đổi giữa hai miền.

Khẳng định sau đây không phải trọng tâm trong dự án này:

$$
P_s(Y) \neq P_t(Y)
$$

Label shift không phải formulation chính vì mối quan tâm chủ yếu là mismatch giữa hành vi normal ở source và hành vi ở target, chứ không phải sự thay đổi tần suất anomaly tự thân.

### 2.3. Không nên đóng khung độ lệch là “mild” một cách cứng nhắc

Bản trước của note thiên khá mạnh về “mild covariate shift”. Cách nói đó là quá hẹp cho một dự án có thể bao gồm nhiều bộ dữ liệu và nhiều mức độ lệch khác nhau giữa source và target.

Cách diễn đạt an toàn hơn là:

- setting này là **same-family** chứ không phải fully heterogeneous cross-domain transfer,
- hiệu ứng chính vẫn là **covariate shift dưới nonstationarity**,
- nhưng mức độ shift có thể trải từ **mild đến hard** tùy cặp máy, khoảng cách thời gian, hoặc độ khác nhau về operating regime.

Cách nói này đủ tổng quát cho cả SMD lẫn các dataset khác, đồng thời vẫn trung thực với protocol thực nghiệm.

### 2.4. Caveat quan trọng: target pool là dữ liệu không nhãn

Trong dự án này, target pool không phải là một tập tham chiếu chỉ gồm normal sạch. Nó là dữ liệu không nhãn và có thể chứa anomaly bị ẩn.

Vì vậy:

- việc đo domain shift không được ngầm giả định rằng mọi window của target đều là normal,
- nếu không, shift score sẽ trộn lẫn domain gap thật với anomaly contamination,
- và mọi claim trong paper dựa trên score này sẽ khó defend hơn nhiều.

Phát biểu đúng cần là:

**ta đo mismatch giữa source và target bằng source-normal data và một reliable target subset, hoặc tương đương là một target pool được gán trọng số theo độ tin cậy**

Đây là một quyết định thiết kế rất quan trọng.

## 3. Giao thức đo lường được khuyến nghị

### 3.1. Giao thức mặc định

Giao thức mặc định được khuyến nghị giữ hai view chính:

1. **Representation shift**
2. **Dependency shift**

Hai view khác không nằm trong protocol mặc định:

1. **Spectral shift**: tùy chọn, chỉ bật khi có lý do rõ ràng rằng drift liên quan đến tần số hoặc tính chu kỳ.
2. **Domain separability**: chỉ dùng ở vai trò phụ, để sanity check chứ không phải metric xếp hạng chính.

### 3.2. Vì sao hai view này nên là hai view chính

Lựa chọn này là cách dễ defend nhất trên nhiều dataset vì:

- representation discrepancy bắt được temporal semantics vượt ra ngoài các summary statistics thô,
- dependency discrepancy bắt được sự thay đổi trong cấu trúc liên hệ giữa các biến, điều đặc biệt quan trọng trong các hệ thống công nghiệp đa biến,
- cặp view này đủ tổng quát để dùng ngoài SMD,
- và tránh việc làm protocol trở nên quá nặng trước khi có bằng chứng rằng frequency hoặc adversarial separability thực sự tạo thêm giá trị rõ rệt.

### 3.3. Khuyến nghị về tên gọi

Để dùng được trên nhiều dataset, view cấu trúc thứ hai nên được gọi là:

**dependency shift**

chứ không nên chỉ gọi là “correlation shift” ở tầng khái niệm.

Tuy nhiên, trong protocol mặc định, view này được operationalize bằng:

**CORAL-style correlation mismatch**

Việc tách hai tầng này là quan trọng:

- **dependency shift** là khái niệm rộng hơn,
- **CORAL trên shared features** là lựa chọn đo mặc định,
- nhờ đó cách viết vẫn đủ tổng quát mà không overclaim rằng metric đang dùng đã bắt được mọi kiểu quan hệ.

## 4. View 1: Representation Shift

### 4.1. Vai trò của view này

Representation shift là view chính.

Lý do là trong transfer thực tế cho time series, mismatch giữa source và target thường không lộ ra rõ ràng nếu chỉ nhìn mean hoặc variance thô. Nó thường lộ ra rõ hơn trong một feature space mang thông tin về ngữ cảnh cục bộ, subsequence context, và động học đa biến.

### 4.2. Các paper A/A* nên đọc trước

1. **TS2Vec: Towards Universal Representation of Time Series**, AAAI 2022  
   Vì sao quan trọng: một lựa chọn mặc định mạnh để trích xuất representation có ý nghĩa theo thời gian.

2. **Contrastive Learning for Unsupervised Domain Adaptation of Time Series (CLUDA)**, ICLR 2023  
   Vì sao quan trọng: ủng hộ rất tốt cho ý tưởng rằng transfer trong time series nên được khảo sát trên contextual representation space thay vì chỉ ở không gian summary thô.

3. **Drift Doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection (D3R)**, NeurIPS 2023  
   Vì sao quan trọng: đưa ra động lực trực tiếp cho việc nonstationarity và distribution drift là vấn đề thực trong anomaly detection cho multivariate time series.

### 4.3. Cách tính được khuyến nghị

Gọi:

- $X_s^N$ là các window normal của source,
- $\widehat{X}_t^N$ là reliable target-normal subset hoặc một target pool được gán trọng số theo độ tin cậy,
- $f(\cdot)$ là một time-series encoder cố định,
- $Z_s = f(X_s^N)$ và $Z_t = f(\widehat{X}_t^N)$ là embedding tương ứng.

Khi đó định nghĩa:

$$
D_{\mathrm{rep}}(s,t) = \mathrm{MMD}(Z_s, Z_t)
$$

Lưu ý thực hành:

- Encoder nên được giữ cố định cho mọi machine pair nếu mục tiêu là xếp hạng cặp máy.
- TS2Vec là một mặc định mạnh, nhưng có thể thay bằng encoder khác nếu encoder đó được cố định và được biện minh hợp lý.
- Phía target nên được lọc hoặc gán trọng số; không khuyến nghị dùng nguyên toàn bộ unlabeled pool mà không kiểm soát.

### 4.4. View này thực sự đang đo cái gì

View này đo mismatch trong latent temporal representation space.

Nó nên được hiểu là:

- một proxy cho việc target khác source như thế nào trong temporal feature space,
- không phải là bằng chứng trực tiếp của concept drift,
- và cũng không tự nó đảm bảo rằng adaptation chắc chắn sẽ giúp.

Do đó đây là một component score mạnh, nhưng vẫn cần được đối chiếu với hành vi downstream.

## 5. View 2: Dependency Shift

### 5.1. Vai trò của view này

Dependency shift là view chính thứ hai.

Trong multivariate time series, shift không chỉ nằm ở từng biến riêng lẻ. Nó còn xuất hiện ở cách các biến đồng biến thiên, phối hợp, hoặc tách rời dưới các regime mới. Điều này đặc biệt quan trọng trong sensor systems, KPI graphs, và machine states.

### 5.2. Các paper A/A* nên đọc trước

1. **Return of Frustratingly Easy Domain Adaptation**, AAAI 2016  
   Vì sao quan trọng: cung cấp góc nhìn second-order alignment cổ điển đứng sau CORAL.

2. **CauDiTS: Causal Disentangled Domain Adaptation of Multivariate Time Series**, ICML 2024  
   Vì sao quan trọng: lập luận rằng transfer liên miền trong multivariate time series liên quan đến domain-common causal rationales và domain-specific correlations giữa các biến.

3. **SARAD: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series**, NeurIPS 2024  
   Vì sao quan trọng: hỗ trợ rất mạnh cho vai trò thực tế của inter-feature associations trong anomaly detection cho multivariate time series.

### 5.3. Cách tính được khuyến nghị

Sử dụng cùng source và target embeddings như ở trên, tính thống kê covariance hoặc correlation trong shared feature space:

$$
C_s = \mathrm{Cov}(Z_s), \qquad C_t = \mathrm{Cov}(Z_t)
$$

Sau đó định nghĩa dependency score mặc định bằng CORAL-style mismatch:

$$
D_{\mathrm{dep}}(s,t) = \frac{1}{4d^2}\lVert C_s - C_t \rVert_F^2
$$

Cách hiểu:

- ở tầng khái niệm, đây là phép đo mặc định của **dependency shift**,
- ở tầng hiện thực hóa, nó là **CORAL-style second-order discrepancy**,
- và với các dataset đa biến, cách gọi này tổng quát hơn việc chỉ nói “correlation shift”.

### 5.4. Khi nào view này không phù hợp

Nếu dataset về bản chất là univariate hoặc gần như không có cấu trúc liên hệ giữa các kênh, dependency shift nên được xem là:

- không áp dụng,
- hoặc chỉ là một score phụ tùy chọn thay vì yêu cầu mặc định.

Cách này giúp protocol dùng được trên nhiều dataset thay vì chỉ khớp với SMD.

## 6. View 3: Spectral Shift (Tùy chọn)

### 6.1. Vai trò của view này

Spectral shift không nên nằm trong protocol mặc định.

Chỉ nên thêm view này khi có lý do cụ thể để tin rằng shift thể hiện qua:

- thay đổi tính chu kỳ,
- vibration regimes,
- workload cycles,
- rotating machinery signatures,
- hoặc các pattern nhạy với tần số khác.

### 6.2. Các paper A/A* nên đọc trước

1. **Domain Adaptation for Time Series Under Feature and Label Shifts (RAINCOAT)**, ICML 2023  
   Vì sao quan trọng: chỉ ra rõ rằng time view và frequency view có thể shift theo những cách khác nhau giữa các miền.

2. **Boosting Transferability and Discriminability for Time Series Domain Adaptation (ACON)**, NeurIPS 2024  
   Vì sao quan trọng: nhấn mạnh rằng temporal features thường transferable hơn, trong khi frequency features có thể discriminative hơn trong từng domain.

### 6.3. Cách tính được khuyến nghị

Nếu kích hoạt view này:

1. tính PSD hoặc một spectral representation ổn định khác trên source-normal windows và reliable target-normal windows,
2. so sánh spectral distributions theo từng kênh hoặc trong một spectral feature space dùng chung,
3. gom mức khác biệt thành một optional score $D_{\mathrm{spec}}(s,t)$.

Lưu ý quan trọng:

- tài liệu này không ép buộc một spectral metric duy nhất cho mọi dataset,
- vì spectral representation đúng phụ thuộc khá mạnh vào ứng dụng,
- do đó spectral shift chỉ là một phần mở rộng tùy chọn, không phải trụ cột mặc định.

## 7. View phụ: Domain Separability

### 7.1. Vai trò của view này

Domain separability có giá trị, nhưng không nên là metric xếp hạng chính trong dự án này.

Vai trò tốt nhất của nó là:

- sanity check,
- hỗ trợ dựng thực nghiệm,
- và xác nhận sơ bộ rằng source với target thực sự có thể phân biệt được.

### 7.2. Paper A/A* nên đọc

1. **Unsupervised Domain Adaptation by Backpropagation (DANN)**, ICML 2015  
   Vì sao quan trọng: điểm vào thực hành tốt nhất để hiểu domain discrimination và proxy A-distance.

### 7.3. Vì sao không nên dùng nó làm score chính

Trong same-family shift, domain classification nông có thể rất dễ trở thành:

- quá thô,
- quá nhạy với các summary statistics vụn vặt,
- hoặc bão hòa ở mức discrimination accuracy rất cao.

Khi đó, nó vẫn có thể phân biệt “dễ” với “khó” ở mức tổng quát, nhưng rất kém khi cần xếp hạng tinh giữa các machine pair.

Vì vậy lập trường an toàn nhất là:

**giữ separability như một auxiliary sanity check, không dùng như bằng chứng chính cho transfer difficulty**

## 8. Reliable Target Subset và Reliability Weighting

### 8.1. Vì sao phần này quan trọng

Vì target pool không có nhãn, một protocol đo domain shift đủ chắc cần kiểm soát rõ độ tin cậy của target.

Nếu không:

- anomalous windows ở target có thể làm phồng domain gap đo được,
- score có thể không còn phản ánh deployment shift thuần túy,
- và kết quả dễ trở nên thiếu ổn định giữa các machine pair.

### 8.2. Paper A/A* nên đọc

1. **Confidence Score for Source-Free Unsupervised Domain Adaptation (CoWA-JMDS)**, ICML 2022  
   Vì sao quan trọng: dù không phải paper về time-series anomaly detection, nó ủng hộ rất mạnh cho nguyên lý tổng quát rằng các target sample trong adaptation không nên bị xem là đáng tin như nhau.

### 8.3. Quy tắc thực hành được khuyến nghị

Hãy dùng một trong hai cách sau:

1. **Reliable target-normal subset**  
   Chỉ giữ lại các window có pseudo-normal confidence cao hoặc anomaly score thấp.

2. **Reliability weighting**  
   Giữ toàn bộ target windows nhưng gán trọng số theo độ tin cậy khi tính shift.

Quy tắc chọn cụ thể có thể thay đổi tùy dataset, nhưng tài liệu nên luôn nêu rõ rằng target reliability được kiểm soát tường minh.

## 9. Có nên gộp thành một combined score duy nhất không?

### 9.1. Câu trả lời mặc định được khuyến nghị

Với protocol chính, lựa chọn an toàn nhất là:

**báo cáo riêng từng component score**

Tức là báo cáo:

- $D_{\mathrm{rep}}(s,t)$
- $D_{\mathrm{dep}}(s,t)$

Cách này dễ defend hơn so với việc chốt cứng một weighted score quá sớm.

### 9.2. Khi nào combined score là chấp nhận được

Combined score là chấp nhận được nếu mục tiêu là:

- xếp hạng các machine pair ứng viên,
- chọn một benchmark subset,
- hoặc tạo một one-number summary mang tính exploratory.

Trong trường hợp đó:

1. chuẩn hóa từng component score trên toàn bộ tập machine pair, tốt nhất bằng percentile rank,
2. chỉ gộp sau khi đã chuẩn hóa,
3. và nói rõ rằng combined score được dùng cho mục đích ranking chứ không phải như một “định luật” cơ bản của transfer.

### 9.3. Công thức khuyến nghị nếu thật sự cần một score duy nhất

Gọi $\widetilde{D}_{\mathrm{rep}}$ và $\widetilde{D}_{\mathrm{dep}}$ là các score đã được chuẩn hóa. Khi đó:

$$
D_{\mathrm{pair}}(s,t) = \alpha \widetilde{D}_{\mathrm{rep}}(s,t) + (1-\alpha)\widetilde{D}_{\mathrm{dep}}(s,t)
$$

Tuy nhiên, khác với phiên bản trước của note, tài liệu này **không** khuyến nghị chốt $\alpha = 0.7$ như một paper default.

### 9.4. Nên chọn $\alpha$ như thế nào

Cách chọn được khuyến nghị là empirical:

1. chọn một target evaluation metric $M$, ưu tiên **AP** hoặc **AUROC**,
2. định nghĩa downstream gain cho từng pair:

$$
G(i,j) = M_{\mathrm{adaptnas\_combined}}(i,j) - M_{\mathrm{uad\_source}}(i,j)
$$

3. chọn $\alpha$ sao cho tối đa hóa rank correlation giữa $D_{\mathrm{pair}}^{(\alpha)}(i,j)$ và $G(i,j)$ trên một tập held-out hoặc pilot pairs.

Correlation nên ưu tiên:

- Spearman
- hoặc Kendall

Nếu chưa có bước validation như vậy, thì:

- hãy dùng các component score trực tiếp trong phân tích chính,
- và coi mọi combined score chỉ mang tính exploratory.

## 10. Nên dùng các shift score như thế nào để hỗ trợ claim trong paper

### 10.1. Điều gì cần được kiểm chứng

Nếu mục tiêu là hỗ trợ claim rằng `adaptnas_combined` có thể tốt hơn `uad_source` khi có domain shift, thì shift score không chỉ cần “lớn”. Nó cần hữu ích trong việc giải thích hành vi downstream.

Có hai giả thuyết đáng để kiểm:

1. **Shift cao hơn thường làm source-only transfer kém hơn**

$$
D(i,j) \uparrow \quad \Rightarrow \quad M_{\mathrm{uad\_source}}(i,j) \downarrow
$$

2. **Shift cao hơn làm tăng tiềm năng của adaptation**

$$
D(i,j) \uparrow \quad \Rightarrow \quad G(i,j) \uparrow
$$

trong đó $G(i,j)$ là phần gain của `adaptnas_combined` so với `uad_source`.

### 10.2. Nên chọn metric nào cho $M$

Main metric nên được chốt trước.

Với dự án này, các lựa chọn an toàn nhất thường là:

1. **AP**
2. **AUROC**

Nếu dùng một F1 metric phụ thuộc threshold, tài liệu nên giải thích rõ vì sao thresholding protocol đó là protocol đúng cho paper.

### 10.3. Cách báo cáo được khuyến nghị

Với mỗi machine pair, hãy báo cáo:

- source-only performance,
- combined-model performance,
- performance gain,
- representation shift score,
- dependency shift score,
- và nếu cần thì thêm combined ranking score.

Sau đó báo cáo:

- rank correlation giữa shift và source-only performance,
- rank correlation giữa shift và adaptation gain,
- và một phần thảo luận định tính nhỏ cho các ca thành công mạnh nhất và thất bại rõ nhất.

## 11. Thứ tự ưu tiên đọc các paper A/A*

Nếu muốn một reading list ngắn nhưng khớp nhất với tài liệu này, hãy đọc theo thứ tự:

1. **TS2Vec**, AAAI 2022
2. **CLUDA**, ICLR 2023
3. **D3R**, NeurIPS 2023
4. **CORAL**, AAAI 2016
5. **RAINCOAT**, ICML 2023
6. **CauDiTS**, ICML 2024
7. **SARAD**, NeurIPS 2024
8. **ACON**, NeurIPS 2024
9. **CoWA-JMDS**, ICML 2022
10. **DEV**, ICML 2019

Nếu chỉ có thời gian đọc rất ít, hãy ưu tiên:

- TS2Vec
- CLUDA
- CORAL
- CauDiTS
- SARAD
- CoWA-JMDS

## 12. Kết luận cuối cùng

### 12.1. Quan điểm chốt

Bài toán nên được đóng khung là:

**same-family deployment shift dưới temporal nonstationarity, biểu hiện chủ yếu qua covariate shift**

chứ không phải một generic large cross-domain problem, và cũng không nên mặc định rằng mọi setting đều chỉ là mild shift.

### 12.2. Gợi ý protocol chốt

Protocol mặc định nên là:

1. **Representation shift** là score chính
2. **Dependency shift** là score chính thứ hai
3. **Spectral shift** chỉ dùng khi có bằng chứng rõ ràng liên quan đến tần số
4. **Domain separability** chỉ dùng như một auxiliary sanity check

### 12.3. Hướng dẫn cuối cùng về combined score

Lựa chọn an toàn nhất cho paper là báo cáo riêng các component score.

Nếu cần một ranking score duy nhất, chỉ nên gộp các normalized score sau khi chọn trọng số bằng thực nghiệm đối chiếu với downstream gain. Không nên trình bày một manual weight cố định như thể nó được literature hỗ trợ trực tiếp.

### 12.4. Một đoạn có thể dùng lại trong paper

We characterize the practical source-to-target transfer setting as same-family deployment shift under temporal nonstationarity, where the dominant observable effect is covariate shift. Accordingly, we measure source-target mismatch primarily through representation discrepancy and dependency discrepancy in a shared time-series feature space, while spectral shift is treated as optional and domain separability is used only as an auxiliary sanity check. Because the target pool is unlabeled and may contain hidden anomalies, target reliability is explicitly controlled when computing shift scores.

Viết ngắn hơn:

Trong deployment setting thực tế của chúng tôi, domain shift được đo chủ yếu qua representation mismatch và dependency mismatch, đồng thời có kiểm soát tường minh độ tin cậy của target pool thay vì giả định rằng target không nhãn là sạch.
