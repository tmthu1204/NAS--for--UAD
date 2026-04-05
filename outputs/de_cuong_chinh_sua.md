# Bản đề cương sau chỉnh sửa

## 2. NỘI DUNG THỰC HIỆN

### 2.1 Giới thiệu về đề tài

Phát hiện bất thường trên chuỗi thời gian đa biến là một bài toán quan trọng trong giám sát hệ thống máy chủ, hạ tầng công nghệ thông tin, hệ thống công nghiệp và an ninh mạng. Về bản chất, bài toán này hướng đến việc nhận diện các mẫu dữ liệu bất thường có hành vi lệch đáng kể so với quy luật vận hành bình thường của hệ thống [1]. Trong các bối cảnh thực tế, dữ liệu thu thập từ cảm biến và các bản ghi vận hành thường có quy mô lớn, mức độ biến thiên cao và rất khó được gán nhãn đầy đủ. Vì vậy, phát hiện bất thường không giám sát (unsupervised anomaly detection) đã trở thành một hướng nghiên cứu có ý nghĩa cả về học thuật lẫn ứng dụng [1].

Trong những năm gần đây, nhiều công trình tiêu biểu đã được đề xuất cho bài toán này. USAD [2] là một phương pháp phát hiện bất thường không giám sát dựa trên cơ chế tái thiết đối kháng (adversarial reconstruction), qua đó làm nổi bật sai khác giữa mẫu bình thường và mẫu bất thường. OmniAnomaly [3] là một mô hình tái phát ngẫu nhiên cho chuỗi thời gian đa biến, kết hợp với cơ chế biến đổi chuẩn hóa (normalizing flow) để mô hình hóa tốt hơn các động lực phức tạp của dữ liệu. DAGMM [4] là một mô hình kết hợp học biểu diễn với mô hình hỗn hợp Gaussian nhằm ước lượng cấu trúc phân bố trong không gian tiềm ẩn. Gần đây hơn, Anomaly Transformer [8] và TranAD [9] là hai hướng tiếp cận dựa trên cơ chế chú ý (attention), cho thấy khả năng phát hiện bất thường hiệu quả trên các chuỗi thời gian có quan hệ phụ thuộc phức tạp. Ở hướng tự động hóa thiết kế mô hình, công trình PASTA [10], một phương pháp tìm kiếm kiến trúc mạng nơ-ron (Neural Architecture Search - NAS) cho phát hiện bất thường, cho thấy việc tự động lựa chọn kiến trúc có thể giúp cải thiện hiệu quả mô hình so với cách thiết kế thủ công.

Mặc dù đã đạt được nhiều kết quả tích cực, các nghiên cứu hiện có vẫn đối mặt với ba thách thức chính. Thứ nhất, kết quả phát hiện thường biến động đáng kể theo lựa chọn kiến trúc, trong khi nhiều mô hình vẫn phụ thuộc mạnh vào kinh nghiệm thiết kế thủ công [2], [3], [10]. Thứ hai, phân phối dữ liệu có thể thay đổi rõ rệt giữa các hệ thống hoặc giữa các giai đoạn vận hành khác nhau, từ đó gây ra hiện tượng lệch miền (domain shift) và làm suy giảm năng lực khái quát hóa của mô hình. Thứ ba, nhiều kỹ thuật tìm kiếm kiến trúc mạng nơ-ron và thích nghi miền (domain adaptation) được phát triển chủ yếu cho bài toán phân lớp có nhãn, nên chưa thực sự phù hợp với bối cảnh phát hiện bất thường không giám sát, trong khi nhãn của dữ liệu thuộc miền đích thường rất hạn chế hoặc không sẵn có [7], [10].

Xuất phát từ bối cảnh đó, đề tài đề xuất hướng tiếp cận NAS-ADE (Neural Architecture Search for Anomaly Detection Enhancement). Ý tưởng cốt lõi của đề tài là kết hợp học biểu diễn tự giám sát cho chuỗi thời gian, mô hình học một lớp cho dữ liệu bình thường và cơ chế tìm kiếm kiến trúc có xét đến khác biệt giữa miền nguồn và miền đích. Cách tiếp cận này hướng đến việc tự động lựa chọn kiến trúc phù hợp hơn cho bài toán phát hiện bất thường, đồng thời nâng cao tính bền vững của mô hình khi dữ liệu vận hành xuất hiện sự thay đổi về phân phối dữ liệu.

### 2.2 Mục tiêu đề tài

Mục tiêu tổng quát của đề tài là đề xuất một phương pháp có khả năng tự động tìm kiếm kiến trúc phù hợp cho bài toán phát hiện bất thường trên chuỗi thời gian đa biến trong bối cảnh thiếu nhãn ở miền đích. Cụ thể, đề tài hướng tới ba mục tiêu chính: (i) học được biểu diễn có tính khái quát tốt từ dữ liệu không nhãn; (ii) xây dựng cơ chế đánh giá độ tin cậy của dữ liệu thuộc miền đích nhằm hạn chế ảnh hưởng của nhiễu và bất thường trong quá trình thích nghi; và (iii) lựa chọn được kiến trúc mạng nơ-ron có hiệu quả cao, ổn định hơn so với các cấu hình được thiết kế hoàn toàn thủ công.

### 2.3 Phạm vi của đề tài

Đề tài tập trung vào bài toán phát hiện bất thường không giám sát trên chuỗi thời gian đa biến, trong đó dữ liệu đầu vào được tổ chức dưới dạng các cửa sổ thời gian có độ dài cố định. Để bảo đảm tính khách quan trong đánh giá, đề tài dự kiến thực nghiệm trên ba bộ dữ liệu chuẩn đánh giá (benchmark datasets) tiêu biểu là SMD [3], SMAP [11] và MSL [11]. Trong đó, SMD là bộ dữ liệu đa biến thu thập từ các máy chủ của một công ty Internet quy mô lớn, phù hợp với bối cảnh giám sát hạ tầng công nghệ thông tin; SMAP và MSL là hai bộ dữ liệu telemetry của NASA, lần lượt gắn với vệ tinh Soil Moisture Active Passive và hệ thống Mars Science Laboratory, phản ánh các môi trường vận hành có cấu trúc cảm biến và dạng bất thường khác biệt [11]. Việc lựa chọn ba bộ dữ liệu này giúp đánh giá phương pháp trên nhiều bối cảnh ứng dụng hơn, đồng thời giảm thiên lệch khi chỉ kết luận từ một bộ dữ liệu duy nhất.

Phạm vi nghiên cứu được giới hạn trong kịch bản mà miền nguồn chủ yếu bao gồm dữ liệu vận hành bình thường, trong khi miền đích bao gồm dữ liệu không nhãn phục vụ thích nghi và dữ liệu có nhãn chỉ được sử dụng ở bước đánh giá thực nghiệm. Đề tài không hướng tới bài toán phát hiện bất thường có giám sát đầy đủ, không tập trung vào sinh dữ liệu tổng hợp, và chưa đi sâu vào triển khai thời gian thực hoặc tối ưu hóa tài nguyên tính toán cho hệ thống sản xuất quy mô lớn.



### 2.4 Cách tiếp cận dự kiến

Đề tài dự kiến tiếp cận bài toán theo hướng kết hợp ba thành phần bổ trợ lẫn nhau. Thứ nhất, đề tài kế thừa công trình TS-TCC [5], tức phương pháp học biểu diễn chuỗi thời gian thông qua đối sánh theo thời gian và ngữ cảnh (Time-Series Representation Learning via Temporal and Contextual Contrasting - TS-TCC), để học biểu diễn tự giám sát từ dữ liệu chuỗi thời gian không nhãn. Thành phần này giúp mô hình thu được đặc trưng có khả năng khái quát tốt hơn, giảm phụ thuộc vào nhãn và tạo điểm khởi đầu thuận lợi cho giai đoạn tìm kiếm kiến trúc.

Thứ hai, đề tài vận dụng DeepSVDD [6], tức phương pháp mô tả dữ liệu bằng véc-tơ hỗ trợ sâu (Deep Support Vector Data Description - DeepSVDD), để mô hình hóa vùng biểu diễn của dữ liệu bình thường trong không gian tiềm ẩn. Thay vì xem mọi dữ liệu thuộc miền đích đều có vai trò tương đương, đề tài dự kiến xây dựng một điểm số độ tin cậy dựa trên mức độ gần hoặc xa của dữ liệu miền đích so với vùng biểu diễn bình thường đã học được. Theo đó, những mẫu thuộc miền đích có khả năng phản ánh hành vi bình thường cao hơn sẽ được ưu tiên trong quá trình thích nghi, qua đó hạn chế nguy cơ làm sai lệch quá trình học do ảnh hưởng của nhiễu hoặc bất thường.

Thứ ba, đề tài kế thừa khung tối ưu hai tầng của AdaptNAS [7], tức phương pháp thích nghi kiến trúc mạng nơ-ron giữa các miền (Adapting Neural Architectures Between Domains - AdaptNAS), để xây dựng cơ chế tìm kiếm kiến trúc có xét đến khác biệt phân phối dữ liệu. Tuy nhiên, khác với bối cảnh phân lớp có nhãn trong nghiên cứu gốc, đề tài điều chỉnh cơ chế này cho phù hợp với bài toán phát hiện bất thường không giám sát. Cụ thể, tín hiệu hướng dẫn từ miền đích không được rút ra trực tiếp từ nhãn lớp, mà được suy ra từ độ tin cậy của dữ liệu miền đích trong không gian biểu diễn. Trên cơ sở đó, quá trình tìm kiếm kiến trúc sẽ ưu tiên những cấu hình vừa học tốt trên dữ liệu bình thường ở miền nguồn, vừa thích nghi tốt hơn với dữ liệu thuộc miền đích có độ tin cậy cao.

Không gian tìm kiếm kiến trúc của đề tài được thiết kế xoay quanh ba nhóm lựa chọn chính: thành phần trích chọn đặc trưng, thành phần mô hình hóa phụ thuộc theo thời gian, và thành phần sinh điểm số bất thường. Thay vì mở rộng không gian tìm kiếm quá lớn, đề tài chủ động giới hạn các lựa chọn trong những nhóm kiến trúc đã được sử dụng hiệu quả trong xử lý chuỗi thời gian, nhằm bảo đảm tính khả thi của quá trình tối ưu, đồng thời vẫn duy trì đủ mức độ linh hoạt để tìm ra các cấu hình phù hợp cho từng bộ dữ liệu.

So với các nghiên cứu trước, đóng góp dự kiến của đề tài nằm ở việc chuyển hóa cơ chế tìm kiếm kiến trúc có thích nghi miền thành một phương pháp phù hợp hơn với phát hiện bất thường không giám sát. Nếu TS-TCC [5] tập trung vào học biểu diễn, DeepSVDD [6] tập trung vào mô hình hóa tính bình thường, và AdaptNAS [7] tập trung vào thích nghi kiến trúc giữa các miền, thì NAS-ADE hướng tới việc tích hợp ba thành phần này thành một khung phương pháp thống nhất cho bài toán phát hiện bất thường trên chuỗi thời gian đa biến. Trong quá trình đánh giá, đề tài sẽ đối chiếu với các mô hình đối chứng tiêu biểu như USAD [2], OmniAnomaly [3], Anomaly Transformer [8], TranAD [9] và PASTA [10] để làm rõ giá trị bổ sung của cách tiếp cận đề xuất.

### 2.5 Kết quả dự kiến của đề tài

Kết quả dự kiến quan trọng nhất của đề tài là xây dựng được một khung phương pháp NAS-ADE có khả năng tự động lựa chọn kiến trúc phù hợp cho bài toán phát hiện bất thường trên chuỗi thời gian đa biến trong bối cảnh lệch miền và thiếu nhãn. Kết quả này dự kiến được thể hiện thông qua một hệ phương pháp thực nghiệm hoàn chỉnh, có tính lặp lại và có khả năng so sánh với các mô hình đối chứng tiêu biểu.

Về phương diện đánh giá định lượng, đề tài kỳ vọng mô hình đề xuất đạt kết quả cạnh tranh hoặc tốt hơn các mô hình đối chứng trên các chỉ số phổ biến như AUROC, Average Precision, F1-score và Event-F1. Bên cạnh giá trị tuyệt đối của từng chỉ số, đề tài cũng hướng đến một kết quả quan trọng hơn về mặt nghiên cứu, đó là khả năng duy trì hiệu quả ổn định khi chuyển từ miền nguồn sang miền đích có sự thay đổi phân phối dữ liệu.

Về ý nghĩa khoa học, đề tài góp phần kết nối ba hướng nghiên cứu đang được quan tâm, gồm học biểu diễn tự giám sát (self-supervised representation learning) cho chuỗi thời gian [5], phát hiện bất thường không giám sát [2], [3], [8], [9], và tìm kiếm kiến trúc mạng nơ-ron có thích nghi miền [7], [10]. Nếu kết quả thực nghiệm thuận lợi, đề tài có thể là nền tảng cho việc phát triển thành báo cáo khoa học hoặc công bố học thuật trong giai đoạn tiếp theo.

Về ý nghĩa thực tiễn, kết quả của đề tài có thể tạo nền tảng cho các hệ thống giám sát tự động trong những môi trường vận hành biến đổi liên tục, nơi nhãn bất thường rất khó thu thập nhưng yêu cầu phát hiện sớm, ổn định và đáng tin cậy lại đặc biệt cao.

### 2.6 Kế hoạch thực hiện

Kế hoạch thực hiện dự kiến được tổ chức theo các giai đoạn sau:

| Thời gian | Nội dung công việc | Người phụ trách chính | Kết quả dự kiến |
| --- | --- | --- | --- |
| 04/2025 - 06/2025 | Khảo sát tài liệu về phát hiện bất thường trên chuỗi thời gian, học biểu diễn tự giám sát, học một lớp và tìm kiếm kiến trúc mạng nơ-ron có thích nghi miền; xác định bài toán và hướng tiếp cận tổng quát | Cả nhóm | Cơ sở lý thuyết và định hướng nghiên cứu cho đề tài |
| 07/2025 - 09/2025 | Chuẩn hóa dữ liệu, xác lập giao thức thực nghiệm trên các bộ dữ liệu lựa chọn và hoàn thiện bước tiền xử lý | Mai Đức Vân | Bộ dữ liệu và giao thức đánh giá sẵn sàng cho nghiên cứu |
| 10/2025 - 12/2025 | Nghiên cứu và xây dựng thành phần học biểu diễn tự giám sát cho dữ liệu chuỗi thời gian | Tạ Minh Thư | Thành phần biểu diễn có khả năng khởi tạo cho mô hình phát hiện bất thường |
| 01/2026 - 02/2026 | Xây dựng không gian tìm kiếm kiến trúc và các mô hình đối chứng tham chiếu | Tạ Minh Thư | Bộ cấu hình kiến trúc ứng viên và các mô hình đối chứng |
| 03/2026 - 04/2026 | Xây dựng cơ chế đánh giá độ tin cậy của dữ liệu thuộc miền đích dựa trên biểu diễn dữ liệu bình thường | Mai Đức Vân | Cơ chế trọng số dữ liệu miền đích phục vụ thích nghi |
| 05/2026 - 06/2026 | Tích hợp các thành phần thành khung NAS-ADE, chạy thực nghiệm, thực hiện thí nghiệm triệt tiêu và phân tích kết quả | Cả nhóm | Phương pháp đề xuất và bảng kết quả thực nghiệm |
| 07/2026 | Hoàn thiện báo cáo khóa luận, chỉnh sửa bản thảo và chuẩn bị bảo vệ | Cả nhóm | Báo cáo hoàn chỉnh và tài liệu bảo vệ |

## Tài liệu tham khảo

[1] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Computing Surveys*, vol. 41, no. 3, pp. 15:1-15:58, 2009, doi: 10.1145/1541880.1541882.

[2] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, "USAD: UnSupervised Anomaly Detection on Multivariate Time Series," in *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2020, pp. 3395-3404, doi: 10.1145/3394486.3403392.

[3] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network," in *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2019, pp. 2828-2837, doi: 10.1145/3292500.3330672.

[4] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen, "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection," in *International Conference on Learning Representations*, 2018.

[5] E. Eldele, M. Ragab, Z. Chen, M. Wu, C. K. Kwoh, X. Li, and C. Guan, "Time-Series Representation Learning via Temporal and Contextual Contrasting," in *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence*, 2021, pp. 2352-2359, doi: 10.24963/ijcai.2021/324.

[6] L. Ruff, R. Vandermeulen, N. Goernitz, L. Deecke, S. A. Siddiqui, A. Binder, E. Muller, and M. Kloft, "Deep One-Class Classification," in *Proceedings of the 35th International Conference on Machine Learning*, 2018, pp. 4393-4402.

[7] Y. Li, Z. Yang, Y. Wang, and C. Xu, "Adapting Neural Architectures Between Domains," in *Advances in Neural Information Processing Systems 33*, 2020.

[8] J. Xu, H. Wu, J. Wang, and M. Long, "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy," in *International Conference on Learning Representations*, 2022.

[9] S. Tuli, G. Casale, and N. R. Jennings, "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data," *Proceedings of the VLDB Endowment*, vol. 15, no. 6, pp. 1201-1214, 2022, doi: 10.14778/3514061.3514067.

[10] P. Trirat and J.-G. Lee, "PASTA: Neural Architecture Search for Anomaly Detection in Multivariate Time Series," *IEEE Transactions on Emerging Topics in Computational Intelligence*, vol. 9, no. 4, pp. 2924-2939, 2025, doi: 10.1109/TETCI.2024.3508845.

[11] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," in *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2018, pp. 387-395, doi: 10.1145/3219819.3219845.
