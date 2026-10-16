
# Benchmark & Tối Ưu Hóa Nhân Ma Trận (Matrix Multiplication)
## 1. TỔNG QUAN DỰ ÁN

Dự án này tập trung vào nghiên cứu, hiện thực hóa, tối ưu hóa và đánh giá hiệu năng của bài toán **Nhân hai ma trận vuông** ($C = A \times B$) với các số thực độ chính xác kép (`double`) trên nhiều mô hình tính toán song song khác nhau:

1. **Chuẩn tham chiếu (Baseline / Ground Truth)**: Thư viện phần cứng tối ưu **OpenBLAS** (`cblas_dgemm`).
2. **Tuần tự (Sequential)**: Thuật toán Naive 3 vòng lặp kinh điển ($ijk$) và thuật toán tối ưu bộ nhớ đệm cache ($ikj$).
3. **Song song bộ nhớ chia sẻ (Shared Memory)**: Đa luồng sử dụng **OpenMP** (`naive_omp`).
4. **Song song bộ nhớ phân tán (Distributed Memory)**: Đa tiến trình sử dụng **MPI** (OpenMPI / MPICH) với kỹ thuật chia dải hàng (`naive_mpi`).
5. **Song song lai ghép (Hybrid Parallelism)**: Kết hợp cả **MPI + OpenMP** (`naive_hybrid`), trong đó MPI truyền thông giữa các node/process và OpenMP tính toán đa luồng trên từng node.
6. **Thuật toán chia để trị nâng cao (Advanced)**: Thuật toán **Strassen** ($O(n^{2.807})$) tuần tự và song song hóa bằng OpenMP Tasks.

---

## 2. CẤU TRÚC THƯ MỤC VÀ Ý NGHĨA CÁC FILE

```text
pc/
├── include/
│   └── common.h             # Header chung: định nghĩa kiểu dữ liệu, struct, nguyên mẫu hàm
├── src/
│   ├── common/
│   │   └── common.cpp       # Triển khai các tiện ích: căn chỉnh RAM, PRNG, Freivalds, CSV, CLI
│   ├── naive/
│   │   ├── naive_seq.cpp    # Nhân ma trận tuần tự (ijk, ikj, ikj_rows)
│   │   ├── naive_omp.cpp    # Nhân ma trận đa luồng OpenMP
│   │   ├── naive_mpi.cpp    # Nhân ma trận phân tán MPI (Scatterv / Bcast / Gatherv)
│   │   └── naive_hybrid.cpp # Nhân ma trận lai ghép MPI + OpenMP
│   └── ref/
│       └── blas_ref.cpp     # Bản chuẩn OpenBLAS dùng so sánh hiệu năng và độ chính xác
├── scripts/
│   ├── sysinfo.sh           # Thu thập thông số phần cứng, OS, compiler và test OpenMP
│   └── run_verify.sh        # Tự động build và test tính đúng đắn trên nhiều kích cỡ (n, threads)
├── tests/
│   ├── test_common.cpp      # Unit test cho hạ tầng tiện ích trong common.cpp
│   ├── test_naive.sh        # Script kiểm thử đúng đắn cho naive_seq và naive_omp
│   ├── test_mpi.sh          # Script kiểm thử đúng đắn cho naive_mpi
│   └── test_hybrid.sh       # Script kiểm thử đúng đắn cho naive_hybrid
├── results/
│   ├── shared/              # Lưu sysinfo.txt, log kiểm thử và dữ liệu CSV đo đạc
│   └── cluster/             # Lưu kết quả chạy trên hệ thống phân tán / cluster
├── reports/                 # Chứa báo cáo khoa học, biểu đồ speedup, efficiency
├── bin/                     # Thư mục chứa các file thực thi sau khi biên dịch (binary)
├── setup.md                 # Hướng dẫn cài đặt môi trường trên macOS / Linux
├── README.md                # Tài liệu hướng dẫn thao tác lệnh nhanh
└── scripts.md               # [Tài liệu này] Toàn cảnh dự án, chi tiết từng file & roadmap thực hiện
```

---

### Chi Tiết Ý Nghĩa Từng File

#### A. Tầng Cấu Trúc Chung (`include/` và `src/common/`)
* **`include/common.h`**:
  * Định nghĩa kiểu số thực `using real_t = double;`.
  * `struct Args`: Chứa tham số cấu hình truyền qua CLI (`-n`, `-t`, `-r`, `-s`, `--verify`, `--cutoff`, `--depth`, `--kernel`, `--warmup`).
  * `struct RunInfo`: Chứa dữ liệu một lượt đo lường (`algo`, `model`, `n`, `P`, `T`, `t_total`, `t_compute`, `t_comm`, `verified`, `rel_err`).
  * Khai báo các nguyên mẫu hàm cấp phát bộ nhớ, sinh ma trận, bấm giờ, kiểm chứng và các kernel toán học.
* **`src/common/common.cpp`**:
  * `alloc_matrix`: Dùng `posix_memalign` căn chỉnh 64-byte (trùng kích thước CPU cache line và hỗ trợ vector hóa SIMD).
  * `gen_matrix`: Sinh ma trận ngẫu nhiên dựa trên hàm băm `splitmix64`. Giá trị phần tử tại tọa độ $(i, j)$ hoàn toàn xác định theo `seed` và chỉ số toàn cục (tính nhất quán giữa các tiến trình độc lập).
  * `now`: Đo thời gian thực (*wall-clock time*) bằng `omp_get_wtime()`.
  * `verify_freivalds`: Thuật toán xác suất kiểm tra $A \times B \stackrel{?}{=} C$ trong thời gian $O(n^2)$ và $O(n)$ bộ nhớ phụ thông qua 3 vector ngẫu nhiên $r \in \{0, 1\}^n$.
  * `verify_full`: So sánh từng phần tử với ma trận tham chiếu $C_{ref}$, tính chuẩn sai số Frobenius tương đối.
  * `parse_args` & `print_csv`: Phân tích cờ dòng lệnh và in kết quả ra định dạng CSV chuẩn kèm hostname và git commit.

#### B. Tầng Thuật Toán Nhân Ma Trận (`src/`)
* **`src/ref/blas_ref.cpp`**:
  * Gọi hàm `cblas_dgemm` từ thư viện tối ưu hóa phần cứng OpenBLAS.
  * Đạt hiệu năng tiệm cận giới hạn lý thuyết phần cứng (GFLOPS cực cao), dùng làm thước đo chuẩn so sánh và kiểm chứng kết quả.
* **`src/naive/naive_seq.cpp`**:
  * `matmul_ijk`: 3 vòng lặp kinh điển $i \to j \to k$, truy cập $B$ nhảy cóc theo cột $\to$ cache miss cao.
  * `matmul_ikj`: Đảo thứ tự vòng lặp sang $i \to k \to j$, cả $B$ và $C$ được duyệt liên tục theo hàng $\to$ tận dụng spatial locality, tự động vector hóa SIMD.
  * `matmul_ikj_rows`: Tính toán trên từng khối hàng cục bộ (phục vụ MPI).
* **`src/naive/naive_omp.cpp`**:
  * Song song hóa các hàng ma trận $C$ trên nhiều CPU core bằng chỉ thị `#pragma omp parallel for schedule(static)`.
  * Hỗ trợ cả 2 kernel: `matmul_ijk_omp` và `matmul_ikj_omp`.
* **`src/naive/naive_mpi.cpp`**:
  * Tiến trình Rank 0 làm master: chia các dải hàng của ma trận $A$ bằng `MPI_Scatterv` (hỗ trợ số hàng không chia hết cho $P$), broadcast ma trận $B$ cho tất cả các tiến trình qua `MPI_Bcast`.
  * Các rank tính toán độc lập dải hàng kết quả $C$ cục bộ bằng `matmul_ikj_rows`.
  * Rank 0 thu gom các dải hàng của $C$ qua `MPI_Gatherv`.
  * Tách bạch rõ ràng `t_compute` và `t_comm`.
* **`src/naive/naive_hybrid.cpp`**:
  * Phân tán bộ nhớ giữa các máy/tiến trình qua MPI, bên trong từng tiến trình lại mở rộng đa luồng tính toán qua OpenMP (`MPI_THREAD_FUNNELED`).
  * Khai thác tối đa kiến trúc cụm máy tính đa nhân hiện đại (Cluster of Multi-core Nodes).

#### C. Tầng Tự Động Hóa & Kiểm Thử (`scripts/` & `tests/`)
* **`scripts/sysinfo.sh`**: Thu thập thông tin phần cứng máy tính (Apple Silicon / Intel / AMD), số nhân vật lý/logic, dung lượng RAM, phiên bản compiler (`g++-16`, `clang++`, `mpicxx`) và chạy thử nghiệm OpenMP.
* **`scripts/run_verify.sh`**: Tự động build và chạy kiểm tra tính đúng đắn trên 7 kích thước ma trận đặc thù ($n=1, 2, 3, 7, 16, 100, 257$) và các số luồng ($T=1, 2, 4$).
* **`tests/test_common.cpp`**: Kiểm thử các tính năng nền tảng trong `common.cpp` (bộ nhớ, tái lập seed, Freivalds vs Naive).
* **`tests/test_naive.sh`, `test_mpi.sh`, `test_hybrid.sh`**: Các bộ test bash tự động cho từng thành phần thuật toán.

---

## 3. ROADMAP

### Giai Đoạn 1: Chuẩn Bị & Cấu Hình Môi Trường
1. **Cài đặt công cụ bắt buộc** (trên macOS qua Homebrew):
   ```bash
   brew install gcc libomp open-mpi openblas
   ```
2. **Kiểm tra và dọn dẹp script**:
   * Kiểm tra các script `sysinfo.sh` và `run_verify.sh` (đảm bảo không bị dính lệnh `cat > ... <<'EOF'`).
   * Cấp quyền thực thi:
     ```bash
     chmod +x scripts/*.sh tests/*.sh
     ```
3. **Thu thập thông tin hệ thống**:
   ```bash
   mkdir -p results/shared bin
   ./scripts/sysinfo.sh | tee results/shared/sysinfo.txt
   ```

---

### Giai Đoạn 2: Biên Dịch Toàn Bộ Các Thành Phần

Dùng các lệnh sau từ thư mục gốc dự án:

```bash
OPENBLAS_PREFIX="$(brew --prefix openblas)"

# 1. Biên dịch bộ chuẩn OpenBLAS
g++-16 -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude -I"$OPENBLAS_PREFIX/include" \
  src/common/common.cpp src/ref/blas_ref.cpp \
  -L"$OPENBLAS_PREFIX/lib" -Wl,-rpath,"$OPENBLAS_PREFIX/lib" \
  -lopenblas -o bin/blas_ref

# 2. Biên dịch bản Tuần tự (Sequential)
g++-16 -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude src/common/common.cpp src/naive/naive_seq.cpp \
  -o bin/naive_seq

# 3. Biên dịch bản Song song đa luồng (OpenMP)
g++-16 -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude src/common/common.cpp src/naive/naive_omp.cpp \
  -o bin/naive_omp

# 4. Biên dịch bản Phân tán (MPI)
mpicxx -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude src/common/common.cpp src/naive/naive_mpi.cpp \
  -o bin/naive_mpi

# 5. Biên dịch bản Lai ghép (Hybrid MPI + OpenMP)
mpicxx -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude src/common/common.cpp src/naive/naive_hybrid.cpp \
  -o bin/naive_hybrid
```

---

### Giai Đoạn 3: Kiểm Thử Tính Đúng Đắn (Correctness Verification)

Chạy các bộ kiểm tra tự động để đảm bảo thuật toán không bị sai logic hay race condition:

```bash
# 1. Kiểm tra bộ thư viện dùng chung common
g++-16 -std=c++17 -O3 -fopenmp -Iinclude src/common/common.cpp tests/test_common.cpp -o bin/test_common
./bin/test_common

# 2. Kiểm thử tự động tổng thể qua script run_verify
BUILD=0 ./scripts/run_verify.sh

# 3. Kiểm thử phân tán MPI
./tests/test_mpi.sh

# 4. Kiểm thử lai ghép Hybrid
./tests/test_hybrid.sh
```
*Tiêu chí đạt:* Tất cả các test đều phải in ra `[PASS]` và mã thoát bằng 0.

---

### Giai Đoạn 4: Đo Lường Hiệu Năng (Benchmarking & Data Collection)

Thu thập số liệu đo lường thực tế trên nhiều dải kích thước ma trận và cấu hình CPU:

1. **So sánh 2 kernel tuần tự ($ijk$ vs $ikj$)**:
   * Chạy với $n \in \{500, 1000, 1500, 2000\}$.
   * Minh chứng hiệu quả vượt trội của tối ưu cache locality trong $ikj$.
2. **Khảo sát Tăng tốc mạnh (Strong Scaling)**:
   * Cố định kích thước ma trận lớn (ví dụ: $n = 2000$).
   * Tăng số luồng OpenMP $T \in \{1, 2, 4, 8\}$.
   * Tăng số process MPI $P \in \{1, 2, 4\}$.
   * Tính Hệ số tăng tốc: $S_p = \frac{T_1}{T_p}$ và Hiệu suất song song: $E_p = \frac{S_p}{p}$.
3. **Phân tích truyền thông trong MPI / Hybrid**:
   * So sánh tỷ trọng giữa thời gian tính toán (`t_compute`) và truyền dữ liệu (`t_comm`).
   * Nhận xét điểm nghẽn đường truyền khi số process tăng lên.
4. **Đối chiếu với OpenBLAS**:
   * Tính toán chỉ số GFLOPS: $\text{GFLOPS} = \frac{2 \times n^3}{\text{thời gian (giây)} \times 10^9}$.
   * Xem code tự viết đạt bao nhiêu % hiệu năng so với OpenBLAS.

---

### Giai Đoạn 5: Mở Rộng Thuật Toán Strassen (Nâng Cao - TV2)
* Hiện thực hàm `strassen_seq` và `strassen_omp` đã được khai báo trong `common.h`.
* Chia ma trận thành 4 khối con $n/2 \times n/2$, tính 7 phép nhân ma trận con $M_1 \dots M_7$.
* Sử dụng OpenMP Tasking: `#pragma omp task` cho các nhánh đệ quy độc lập với độ sâu $\le$ `depth`.
* Khi kích thước $n \le$ `cutoff`, chuyển sang gọi kernel `matmul_ikj` để tránh overhead phân chia đệ quy.

---

### Giai Đoạn 6: Viết Báo Cáo Kỹ Thuật (Final Report)
Lưu kết quả phân tích vào thư mục `reports/`:
1. **Bảng số liệu tổng hợp**: Dữ liệu từ các file CSV của từng mô hình tính toán.
2. **Biểu đồ trực quan hóa**:
   * Biểu đồ thời gian thực thi (Execution Time vs Matrix Dimension $n$).
   * Biểu đồ Speedup và Efficiency theo số CPU cores / threads.
   * Biểu đồ GFLOPS so sánh `naive_seq` vs `naive_omp` vs `naive_mpi` vs `naive_hybrid` vs `OpenBLAS`.
3. **Nhận xét & Kết luận**:
   * Ảnh hưởng của kiến trúc bộ nhớ (Hierarchical Cache vs Distributed Memory).
   * Trade-off giữa chi phí truyền thông mạng và tốc độ tính toán song song.
