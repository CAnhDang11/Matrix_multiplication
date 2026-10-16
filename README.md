Chạy và lưu báo cáo:
```bash
./scripts/sysinfo.sh | tee results/shared/sysinfo.txt
```

Khi đã có `naive_seq.cpp` và `naive_omp.cpp` cùng phần triển khai kernel, chạy:

```bash
chmod +x ./scripts/run_verify.sh
./scripts/run_verify.sh
```

Nếu chỉ muốn chạy các binary đã build, không build lại:
```bash
BUILD=0 ./scripts/run_verify.sh
```

Chỉ kiểm thử một số thread cụ thể:
```bash
TEST_THREADS="1 2" ./scripts/run_verify.sh
```


Nếu tên hoặc vị trí binary khác cấu hình mặc định:
```bash
BUILD=0 VERIFY_BINS="bin/naive_seq bin/naive_omp" \
  ./scripts/run_verify.sh
```

**Lưu ý:** `--verify` chỉ hoạt động khi mỗi executable thực sự gọi `verify_freivalds()` hoặc `verify_full()` và trả mã thoát khác 0 khi kiểm chứng thất bại. Script trên là khung tự động hóa kiểm thử; nó không thay thế việc viết kernel và logic xác minh trong các chương trình `C++`.

Sau khi tạo xong, chạy:
```bash
./scripts/sysinfo.sh
bash -n scripts/sysinfo.sh
bash -n scripts/run_verify.sh
```

Nếu hai lệnh bash `-n` không in lỗi, cú pháp shell đã hợp lệ.

* Biên dịch blas_ref
Trên macOS dùng Homebrew:
```bash
OPENBLAS_PREFIX="$(brew --prefix openblas)"

g++-16 -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude \
  -I"$OPENBLAS_PREFIX/include" \
  src/common/common.cpp \
  src/ref/blas_ref.cpp \
  -L"$OPENBLAS_PREFIX/lib" \
  -Wl,-rpath,"$OPENBLAS_PREFIX/lib" \
  -lopenblas \
  -o bin/blas_ref
```

Nếu thành công, kiểm tra:
```bash
ls -lh bin/blas_ref
```

Nếu báo không tìm thấy cblas.h, kiểm tra:
```bash
find "$(brew --prefix openblas)" -name cblas.h
```

5. Chạy kiểm thử
Thử ma trận nhỏ trước:
```bash
OPENBLAS_NUM_THREADS=1 ./bin/blas_ref \
  -n 16 -t 1 -r 2 -s 42 --verify
```

Sau đó thử BLAS nhiều thread:
```bash
OPENBLAS_NUM_THREADS=4 ./bin/blas_ref \
  -n 500 -t 4 -r 3 -s 42 --verify
```

`OPENBLAS_NUM_THREADS` điều khiển số thread thực tế của OpenBLAS; `-t` trong chương trình hiện dùng để ghi nhận cấu hình vào CSV, nên hai giá trị phải khớp nhau. Chương trình thực hiện kiểm chứng Freivalds ở mọi kích thước và kiểm tra đầy đủ với phép nhân ijk khi $(n\leq1000)$.

- **Lưu ý**: `t_total` hiện là thời gian gọi dgemm, chưa bao gồm khởi tạo ma trận và kiểm chứng. Điều này phù hợp để so sánh thời gian tính toán kernel, nhưng khi viết report cần ghi rõ quy ước đo thời gian.
Sau khi chạy thử, nếu biên dịch hoặc kiểm chứng thất bại, gửi mình nguyên văn lỗi để mình sửa theo môi trường macOS của bạn.

**Biên dịch** : 
Vì `common.cpp` dùng OpenMP timer, cần bật `-fopenmp` ngay cả với phiên bản tuần tự:
```bash
g++-16 -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude \
  src/common/common.cpp \
  src/naive/naive_seq.cpp \
  -o bin/naive_seq
```

Kiểm tra binary đã được tạo:
```bash
ls -lh bin/naive_seq
```

Hoặc chạy trực tiếp lệnh cơ bản:
```bash
./bin/naive_seq -n 16 -t 1 -r 2 -s 42 --kernel ikj --verify
```

* **Biên dịch**
```bash
g++-16 -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude \
  src/common/common.cpp \
  src/naive/naive_omp.cpp \
  -o bin/naive_omp
```

Chạy kiểm thử với nhiều thread

Chạy lần lượt:
```bash
./bin/naive_omp -n 16 -t 1 -r 2 -s 42 --kernel ikj --verify
./bin/naive_omp -n 100 -t 2 -r 2 -s 42 --kernel ikj --verify
./bin/naive_omp -n 257 -t 4 -r 2 -s 42 --kernel ikj --verify
```

Để so sánh hai kernel:
```bash
./bin/naive_omp -n 500 -t 4 -r 3 --kernel ijk --verify
./bin/naive_omp -n 500 -t 4 -r 3 --kernel ikj --verify
```

- Kết quả phải có `verified=true`. Với cùng `n` và `seed`, hai kernel cần tạo ra kết quả tương đương trong sai số cho phép.
- Một điểm quan trọng: bản trên đã song song hóa đúng theo hàng, nhưng nếu muốn kiến trúc code sạch cho cả nhóm, bước tiếp theo nên tách các kernel ra file `naive_kernels.cpp` và đặt `main()` riêng cho từng executable. Điều đó giúp `MPI/Hybrid` tái sử dụng kernel mà không bị xung đột `main()`.

- **Biên dịch**

Từ thư mục gốc của project, chạy:
```bash
mkdir -p bin

mpicxx -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude \
  src/common/common.cpp \
  src/naive/naive_mpi.cpp \
  src/naive/naive_seq.cpp \
  -o bin/naive_mpi
```

- **Lưu ý:** Không dùng lệnh trên nếu `naive_seq.cpp` có hàm `main()` như phiên bản trước — khi đó sẽ bị lỗi multiple definition of main. Chỉ cần biên dịch `common.cpp` và `naive_mpi.cpp`, vì `naive_mpi.cpp` đã tự triển khai phép nhân tham chiếu và gọi `matmul_ikj_rows()`. Do đó, lệnh đúng là:
```bash
mpicxx -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude \
  src/common/common.cpp \
  src/naive/naive_mpi.cpp \
  -o bin/naive_mpi
```

Biên dịch trên macOS
Nếu dùng `g++-16` và `OpenMP` từ `Homebrew`, hãy dùng `mpicxx` để biên dịch MPI:
```bash
mkdir -p bin

mpicxx -std=c++17 -O3 -march=native -fopenmp \
  -Iinclude \
  src/common/common.cpp \
  src/naive/naive_hybrid.cpp \
  -o bin/naive_hybrid
```

- Chạy kiểm thử
Bắt đầu với ma trận nhỏ để xác nhận tính đúng đắn:
```bash
# 1 process, 1 thread
mpirun -np 1 ./bin/naive_hybrid \
  -n 16 -t 1 -r 2 --verify

# 2 processes, mỗi process 2 threads
mpirun -np 2 ./bin/naive_hybrid \
  -n 16 -t 2 -r 2 --verify

# 3 processes, mỗi process 4 threads; n không chia hết cho P
mpirun -np 3 ./bin/naive_hybrid \
  -n 7 -t 4 -r 2 --verify

# Ma trận lớn hơn
mpirun -np 4 ./bin/naive_hybrid \
  -n 100 -t 2 -r 2 --verify
```

- Nếu OpenMPI báo lỗi về số slot khi chạy nhiều process hoặc thread, hãy kiểm tra cấu hình runtime trước; không nên mặc định thêm --oversubscribe vào các benchmark chính thức vì nó có thể làm sai lệch kết quả hiệu năng.

* Kiểm tra hiệu năng

Sau khi các bài test đúng, so sánh MPI và hybrid với cùng ma trận và số process:
```bash
mpirun -np 2 ./bin/naive_mpi \
  -n 1000 -r 3 --verify

mpirun -np 2 ./bin/naive_hybrid \
  -n 1000 -t 4 -r 3 --verify
```

Trong đó:
- P là số MPI process.
- T là số OpenMP thread trên mỗi process.
- t_compute là thời gian tính toán lớn nhất trong các process.
- t_comm là thời gian truyền dữ liệu lớn nhất được đo tại mỗi process.
- t_total bao gồm phân phối dữ liệu, broadcast, tính toán và thu kết quả.

**Lưu ý:** 
- `t_compute + t_comm` không nhất thiết bằng `t_total`, do có thêm đồng bộ và các phần chi phí khác.
- Một điểm cần kiểm tra trong `common.cpp`: nếu `print_csv()` hiện in dòng tiêu đề mỗi lần gọi, hãy sửa để chỉ in header một lần. Nếu không, file CSV benchmark sẽ có các dòng tiêu đề xen giữa dữ liệu.
Khi `naive_mpi` và `naive_hybrid` đều biên dịch và vượt qua kiểm thử, phần triển khai chính của TV1 sẽ gần hoàn chỉnh; tiếp theo là kiểm tra `blas_ref`, chạy script xác minh và chuẩn hóa kết quả benchmark.

