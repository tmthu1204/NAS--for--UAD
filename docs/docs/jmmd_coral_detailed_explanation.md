# Diễn giải chi tiết JMMD và CORAL cho representation shift và dependency shift

## 1. Mục tiêu của file này

File này giải thích một cách chi tiết nhưng dễ hiểu cách đi từ hai paper gốc sau đến hai công thức dùng trong bài toán domain shift của bạn:

1. JMMD từ paper *Deep Transfer Learning with Joint Adaptation Networks* (ICML 2017).
2. CORAL từ paper *Return of Frustratingly Easy Domain Adaptation* (AAAI 2016), kèm cách nhìn theo Deep CORAL khi áp dụng trong deep feature space.

Mục tiêu không phải chỉ chép lại công thức, mà là trả lời rõ:

- công thức xuất phát từ ý tưởng nào,
- từng ký hiệu có nghĩa gì,
- vì sao JMMD phù hợp cho representation shift,
- vì sao CORAL phù hợp cho dependency shift,
- và cách chuyển hai công thức đó vào bài toán multivariate time series anomaly detection.

---

## 2. JMMD: từ MMD đến representation shift

### 2.1. Bắt đầu từ câu hỏi ta thực sự muốn đo cái gì

Giả sử mỗi window time series `x` được đưa qua một encoder và cho ra embedding:

$$
z = f(x).
$$

Khi đó, nếu source và target khác nhau trong không gian biểu diễn, điều ta muốn nói thực ra là:

$$
P_s(z) \neq P_t(z).
$$

Nói bằng lời:

- source và target sinh ra các embedding khác nhau,
- nên phân phối của các điểm trong latent space khác nhau,
- và khoảng cách giữa hai phân phối đó chính là representation shift.

### 2.2. MMD thông thường đi từ đâu ra

Ý tưởng của MMD là: thay vì so trực tiếp hai phân phối rất khó xử lý, ta đưa dữ liệu vào một không gian đặc trưng khác thông qua một ánh xạ `phi`, rồi so trung bình của hai phân phối trong không gian đó.

Mean embedding của source và target là:

$$
\mu_s = \mathbb{E}_{z \sim P_s}[\phi(z)],
\qquad
\mu_t = \mathbb{E}_{z \sim P_t}[\phi(z)].
$$

Khi đó MMD là:

$$
\mathrm{MMD}^2(P_s, P_t) = \|\mu_s - \mu_t\|_{\mathcal H}^2.
$$

Ý nghĩa rất trực quan:

- mỗi mẫu embedding `z` được biến đổi thành `phi(z)`;
- ta lấy trung bình các điểm của source và target trong không gian đó;
- nếu hai trung bình xa nhau, thì hai phân phối khác nhau;
- khoảng cách đó chính là discrepancy giữa source và target.

Nếu viết empirical form với kernel `k`, ta có:

$$
\mathrm{MMD}^2(Z_s,Z_t)
= \frac{1}{n_s^2}\sum_{i,i'} k(z_{i,s},z_{i',s})
+ \frac{1}{n_t^2}\sum_{j,j'} k(z_{j,t},z_{j',t})
- \frac{2}{n_sn_t}\sum_{i,j} k(z_{i,s},z_{j,t}).
$$

Trong đó:

- `n_s` là số sample source,
- `n_t` là số sample target,
- `k` thường là Gaussian kernel.

Ba hạng trên nên được hiểu như sau:

- hạng 1: source giống source đến mức nào,
- hạng 2: target giống target đến mức nào,
- hạng 3: source giống target đến mức nào.

Nếu source và target thực sự giống nhau, hạng chéo sẽ lớn và toàn bộ MMD nhỏ. Nếu source và target khác nhau, hạng chéo giảm tương đối và MMD tăng.

### 2.3. Vì sao JAN nói MMD một tầng là chưa đủ

Paper JAN chỉ ra rằng trong deep network, domain shift không chỉ xuất hiện ở một embedding cuối cùng. Nó có thể tồn tại đồng thời ở nhiều tầng biểu diễn, đặc biệt là các tầng domain-specific gần head của mạng.

Nếu chỉ đo MMD ở một tầng cuối, ta mới đang so:

$$
P_s(z^{L}) \text{ và } P_t(z^{L}),
$$

tức là chỉ một marginal distribution của một tầng.

Nhưng nếu cùng một mẫu đi qua nhiều tầng `l1, l2, ..., lm`, thì thực chất ta có một bộ đặc trưng:

$$
Z = (z^{\ell_1}, z^{\ell_2}, \dots, z^{\ell_m}).
$$

Điều cần so không chỉ là từng tầng riêng lẻ, mà là joint distribution của toàn bộ bộ đặc trưng đó:

$$
P_s(Z) \neq P_t(Z).
$$

Đây chính là động cơ của JMMD.

### 2.4. Từ joint distribution đến JMMD

Để so joint distribution của nhiều tầng, JAN dùng tensor-product RKHS. Thay vì ánh xạ một tầng bằng `phi(z)`, ta ánh xạ đồng thời nhiều tầng bằng:

$$
\Phi(Z) = \bigotimes_{\ell \in \mathcal L} \phi^{\ell}(z^{\ell}).
$$

Trong đó:

- `mathcal{L}` là tập các tầng được chọn,
- `z^ell` là feature tại tầng `ell`,
- `phi^ell` là feature map của kernel ở tầng đó,
- `otimes` là tensor product, tức là ghép các ánh xạ của nhiều tầng lại thành một ánh xạ joint.

Mean embedding của joint distribution khi đó là:

$$
\mu_P = \mathbb{E}_{Z \sim P}\left[\bigotimes_{\ell \in \mathcal L} \phi^{\ell}(z^{\ell})\right].
$$

Vì vậy, khoảng cách giữa source và target trong joint space là:

$$
D_{\mathrm{rep}}^{\mathrm{JMMD}}(P_s,P_t)
=
\left\|
\mathbb{E}_{Z_s \sim P_s}\left[\bigotimes_{\ell \in \mathcal L} \phi^{\ell}(z_s^{\ell})\right]
-
\mathbb{E}_{Z_t \sim P_t}\left[\bigotimes_{\ell \in \mathcal L} \phi^{\ell}(z_t^{\ell})\right]
\right\|^2.
$$

Đây là công thức population-level của JMMD.

### 2.5. Cách hiểu trực quan của JMMD

So với MMD thông thường:

- MMD chỉ so trung bình của source và target ở một không gian đặc trưng;
- JMMD so trung bình joint của source và target qua nhiều tầng cùng lúc.

Vì vậy JMMD bắt được những domain gap kiểu:

- hai domain trông có vẻ giống nhau ở embedding cuối, nhưng khác nhau ở các tầng ẩn;
- hai domain khác nhau không chỉ ở vị trí điểm trong latent space, mà còn ở cách các mức biểu diễn cùng biến thiên với nhau.

Đây là lý do JMMD hợp lý hơn plain MMD khi bạn muốn dùng nó như representation shift chính.

### 2.6. Dạng empirical của JMMD

Do kernel của joint space có thể viết thành tích các kernel theo từng tầng:

$$
K(Z, Z') = \prod_{\ell \in \mathcal L} k^{\ell}(z^{\ell}, z'^{\ell}),
$$

nên empirical JMMD có dạng:

$$
\widehat D_{\mathrm{rep}}^{\mathrm{JMMD}}
=
\frac{1}{n_s^2}\sum_{i,i'}
\prod_{\ell \in \mathcal L} k^{\ell}(z_{i,s}^{\ell}, z_{i',s}^{\ell})
+
\frac{1}{n_t^2}\sum_{j,j'}
\prod_{\ell \in \mathcal L} k^{\ell}(z_{j,t}^{\ell}, z_{j',t}^{\ell})
-
\frac{2}{n_sn_t}\sum_{i,j}
\prod_{\ell \in \mathcal L} k^{\ell}(z_{i,s}^{\ell}, z_{j,t}^{\ell}).
$$

Ý nghĩa của từng phần vẫn giống MMD, chỉ khác ở chỗ mỗi phép so sánh bây giờ là phép so sánh joint qua nhiều tầng.

### 2.7. Linear-time estimator mà JAN dùng khi huấn luyện

Vì bản đầy đủ ở trên tốn `O(n^2)`, JAN dùng bản estimator tuyến tính theo minibatch để train hiệu quả hơn:

$$
\widehat D_{\mathrm{rep},L}^{\mathrm{JMMD}}
=
\frac{2}{n}
\sum_{i=1}^{n/2}
\Big[
\prod_{\ell\in\mathcal L} k^{\ell}(z_{2i-1,s}^{\ell}, z_{2i,s}^{\ell})
+
\prod_{\ell\in\mathcal L} k^{\ell}(z_{2i-1,t}^{\ell}, z_{2i,t}^{\ell})
-
\prod_{\ell\in\mathcal L} k^{\ell}(z_{2i-1,s}^{\ell}, z_{2i,t}^{\ell})
-
\prod_{\ell\in\mathcal L} k^{\ell}(z_{2i-1,t}^{\ell}, z_{2i,s}^{\ell})
\Big].
$$

Trong báo cáo của bạn, nếu mục tiêu là đo shift chứ không phải train adaptation end-to-end, bản empirical đầy đủ thường dễ giải thích hơn. Còn nếu muốn cài đặt trong quá trình huấn luyện hoặc tính trên minibatch lớn, linear-time estimator là phiên bản thực dụng hơn.

### 2.8. Cách chuyển JMMD thành representation shift cho bài toán của bạn

Trong bài toán multivariate time series anomaly detection, hãy hiểu như sau:

1. Cắt source và target thành các window normal.
2. Đưa từng window qua encoder.
3. Lấy đặc trưng ở nhiều tầng `mathcal{L}`.
4. Xem mỗi window là một điểm joint feature `Z`.
5. Tính JMMD giữa source và target.

Khi đó representation shift được định nghĩa là:

$$
D_{\mathrm{rep}}(s,t) = \widehat D_{\mathrm{rep}}^{\mathrm{JMMD}}(P_s, P_t).
$$

Diễn giải:

- `D_rep` lớn: source và target khác nhau mạnh trong không gian biểu diễn;
- `D_rep` nhỏ: source và target gần nhau về contextual representation.

Một lưu ý quan trọng:

- nếu bạn chỉ có đúng một embedding cuối, thì về bản chất bạn đang dùng MMD chứ chưa khai thác hết JMMD;
- muốn đúng tinh thần JAN, bạn cần đặc trưng của ít nhất hai tầng hoặc hai mức biểu diễn.

---

## 3. CORAL: từ covariance alignment đến dependency shift

### 3.1. CORAL muốn bắt loại khác biệt nào

Nếu JMMD tập trung vào khác biệt của toàn bộ phân phối biểu diễn, thì CORAL tập trung vào khác biệt trong cấu trúc tương quan giữa các chiều đặc trưng.

Giả sử ta có feature matrix:

$$
D_s \in \mathbb{R}^{n_s \times d},
\qquad
D_t \in \mathbb{R}^{n_t \times d},
$$

trong đó:

- mỗi hàng là một sample,
- mỗi cột là một chiều đặc trưng,
- `d` là số chiều feature.

Khi đó source và target có thể khác nhau không chỉ ở mean, mà còn ở cách các chiều feature cùng biến thiên với nhau. Phần này được thể hiện qua covariance matrix.

### 3.2. Covariance được tính như thế nào

Covariance matrix của source là:

$$
C_s
=
\frac{1}{n_s-1}
\left(
D_s^\top D_s
-
\frac{1}{n_s}(\mathbf 1^\top D_s)^\top (\mathbf 1^\top D_s)
\right).
$$

Tương tự, covariance của target là:

$$
C_t
=
\frac{1}{n_t-1}
\left(
D_t^\top D_t
-
\frac{1}{n_t}(\mathbf 1^\top D_t)^\top (\mathbf 1^\top D_t)
\right).
$$

Trong đó:

- `mathbf 1` là vector toàn số 1,
- phần trừ phía sau dùng để loại ảnh hưởng của mean,
- kết quả `C_s, C_t` là hai ma trận `d x d`.

Mỗi phần tử của covariance matrix nói cho ta biết hai chiều đặc trưng có xu hướng cùng tăng cùng giảm hay không. Vì vậy covariance không đo từng chiều riêng lẻ, mà đo dependency structure giữa các chiều.

### 3.3. CORAL gốc đặt bài toán như thế nào

Paper CORAL gốc không bắt đầu bằng việc định nghĩa một score. Nó bắt đầu bằng bài toán adaptation:

hãy tìm một phép biến đổi tuyến tính `A` trên source sao cho covariance của source sau biến đổi gần covariance của target nhất.

Công thức là:

$$
\min_A \|A^\top C_s A - C_t\|_F^2.
$$

Đây là công thức lõi của CORAL.

Ý nghĩa:

- `A` biến đổi feature source;
- covariance mới của source sau biến đổi là `A^T C_s A`;
- ta muốn ma trận này giống với `C_t` của target;
- chuẩn Frobenius đo độ lệch giữa hai covariance matrices.

### 3.4. Vì sao whitening và recoloring xuất hiện

Để giải bài toán trên, paper dùng trực giác rất mạnh:

1. trước tiên làm source mất tương quan cũ, tức là whitening;
2. sau đó áp cấu trúc tương quan của target vào, tức là recoloring.

Viết gọn, source sau biến đổi có thể được biểu diễn theo dạng:

$$
\widetilde D_s = D_s C_s^{-1/2} C_t^{1/2}.
$$

Ý nghĩa của hai thừa số:

- `C_s^{-1/2}`: xóa correlation structure hiện tại của source;
- `C_t^{1/2}`: gắn correlation structure của target vào source.

Nên CORAL về bản chất là một phép covariance matching.

### 3.5. Từ CORAL adaptation sang dependency shift score

Trong bài toán của bạn, ta không nhất thiết phải thật sự biến đổi source thành target. Ta chỉ cần một đại lượng nói rằng dependency structure khác nhau nhiều hay ít.

Khi đó, cách tự nhiên nhất là lấy chính độ lệch giữa hai covariance matrices làm score:

$$
D_{\mathrm{dep}}(s,t) = \|C_s - C_t\|_F^2.
$$

Điểm số này có nghĩa rất rõ:

- nếu `C_s` và `C_t` gần nhau, quan hệ phụ thuộc giữa các chiều đặc trưng ở source và target gần nhau;
- nếu hai covariance matrices khác xa nhau, dependency shift lớn.

### 3.6. Deep CORAL thêm chuẩn hóa như thế nào

Khi đưa CORAL vào deep learning, Deep CORAL dùng loss:

$$
L_{\mathrm{CORAL}} = \frac{1}{4d^2} \|C_s - C_t\|_F^2.
$$

Trong đó:

- `d` là số chiều feature,
- hệ số `1 / (4d^2)` giúp loss ổn định hơn về scale,
- nhờ đó có thể cộng loss này với classification loss trong huấn luyện.

Với mục đích dùng như shift score, ta có thể giữ nguyên công thức này:

$$
D_{\mathrm{dep}}(s,t) = \frac{1}{4d^2} \|C_s - C_t\|_F^2.
$$

Đây là phiên bản gọn, sạch, và phù hợp nhất để dùng trong báo cáo.

### 3.7. Vì sao CORAL chính là dependency shift

Dependency shift không phải là khác biệt của toàn bộ phân phối, mà là khác biệt trong quan hệ phụ thuộc giữa các chiều feature.

Covariance matrix chính là đối tượng tự nhiên để đo chuyện đó, vì nó ghi nhận:

- chiều nào đồng biến với chiều nào,
- cường độ đồng biến mạnh hay yếu,
- cấu trúc second-order của feature space.

Vì vậy:

- JMMD trả lời câu hỏi: hai phân phối representation có khác nhau không;
- CORAL trả lời câu hỏi: cấu trúc phụ thuộc giữa các chiều representation có khác nhau không.

Đó là lý do CORAL phù hợp cho dependency shift.

### 3.8. Cách chuyển CORAL thành dependency shift cho bài toán của bạn

Quy trình áp dụng rất đơn giản:

1. Lấy embedding cuối của source và target:

$$
Z_s = \{z_{1,s}, \dots, z_{n_s,s}\},
\qquad
Z_t = \{z_{1,t}, \dots, z_{n_t,t}\}.
$$

2. Tính covariance của hai tập embedding:

$$
C_s = \mathrm{Cov}(Z_s),
\qquad
C_t = \mathrm{Cov}(Z_t).
$$

3. Định nghĩa dependency shift bằng CORAL distance:

$$
D_{\mathrm{dep}}(s,t)=\frac{1}{4d^2}\|C_s-C_t\|_F^2.
$$

Diễn giải:

- `D_dep` lớn: coupling giữa các chiều embedding ở source và target khác nhau mạnh;
- `D_dep` nhỏ: source và target gần nhau về second-order dependency structure.

---

## 4. So sánh vai trò của JMMD và CORAL

### 4.1. JMMD bắt cái gì

JMMD bắt sự khác nhau của phân phối representation qua nhiều tầng.

Nó phù hợp khi bạn muốn đo:

- khác biệt tổng thể trong latent geometry,
- khác biệt ngữ cảnh thời gian qua nhiều mức biểu diễn,
- mismatch giữa source và target trong contextual feature space.

### 4.2. CORAL bắt cái gì

CORAL bắt sự khác nhau của second-order statistics.

Nó phù hợp khi bạn muốn đo:

- khác biệt trong tương quan giữa các chiều feature,
- thay đổi cấu trúc coupling giữa sensor hoặc latent channels,
- dependency mismatch giữa source và target.

### 4.3. Vì sao nên dùng cả hai

Hai metric này không thay nhau hoàn toàn:

- JMMD mạnh ở mức phân phối biểu diễn tổng thể;
- CORAL mạnh ở mức dependency structure.

Trong multivariate time series anomaly detection, domain shift thường không chỉ nằm ở “điểm nằm đâu trong latent space” mà còn ở “các chiều latent phối hợp với nhau như thế nào”. Vì vậy dùng cả hai là hợp lý hơn dùng một metric đơn lẻ.

---

## 5. Công thức chốt để dùng trong bài toán của bạn

### 5.1. Representation shift

Nếu encoder có nhiều tầng đặc trưng, dùng:

$$
D_{\mathrm{rep}}(s,t)
=
\left\|
\mathbb{E}_{Z_s \sim P_s}\left[\bigotimes_{\ell \in \mathcal L}\phi^{\ell}(z_s^{\ell})\right]
-
\mathbb{E}_{Z_t \sim P_t}\left[\bigotimes_{\ell \in \mathcal L}\phi^{\ell}(z_t^{\ell})\right]
\right\|^2.
$$

Nếu chỉ có một embedding cuối, có thể dùng MMD như approximation đơn giản hơn.

### 5.2. Dependency shift

Tính trên embedding cuối:

$$
C_s = \mathrm{Cov}(Z_s),
\qquad
C_t = \mathrm{Cov}(Z_t),
$$

$$
D_{\mathrm{dep}}(s,t)=\frac{1}{4d^2}\|C_s-C_t\|_F^2.
$$

---

## 6. Kết luận ngắn gọn

Nếu tóm tắt thật gọn, toàn bộ logic của hai paper có thể viết lại như sau:

1. JMMD xuất phát từ việc muốn đo khác biệt giữa joint distributions của nhiều tầng biểu diễn, nên nó phù hợp cho representation shift.
2. CORAL xuất phát từ việc muốn align covariance giữa source và target, nên nó phù hợp cho dependency shift.
3. Trong bài toán của bạn, JMMD đo mức lệch của contextual representations, còn CORAL đo mức lệch của dependency structure giữa các chiều embedding.
4. Vì hai metric nhìn domain shift từ hai góc khác nhau, kết hợp cả hai sẽ thuyết phục hơn khi giải thích source--target mismatch.

## 7. Reference gốc

1. Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan. *Deep Transfer Learning with Joint Adaptation Networks*. ICML 2017.
2. Baochen Sun, Jiashi Feng, and Kate Saenko. *Return of Frustratingly Easy Domain Adaptation*. AAAI 2016.
3. Baochen Sun and Kate Saenko. *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. 2016.