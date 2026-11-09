# HƯỚNG DẪN SỬ DỤNG VÀ KHẢO SÁT CỤM CLUSTER (TV3 · D1 & D6)
**Người phụ trách**: Tô Thế Hưng (TV3) — Điều phối viên Cluster  
**Phạm vi**: Dùng chung cho toàn nhóm (TV1, TV2, TV3) khi chạy các bài toán MPI và Hybrid.

---

## 1. THÔNG SỐ KHẢO SÁT HỆ THỐNG CLUSTER (Nhiệm vụ D1 - Đã hoàn thành khảo sát)

*Thông số khảo sát thực tế trên cụm SuperNodeXP (HPCC ĐHBK TP.HCM) ngày 09/11/2026 bởi Tô Thế Hưng (s2211384):*

| Hạng mục | Chi tiết thực tế khảo sát | Ý nghĩa / Ghi chú cho nhóm |
| :--- | :--- | :--- |
| **Số lượng Node** | **21 compute nodes** (`node[1-4,6-7,11-23,25]`, `node9`) | Quản lý bởi Slurm (`partition: compute`, `highmem`) |
| **CPU mỗi node** | **Dual Socket: 2 × Intel(R) Xeon(R) CPU E5-2630L v4 @ 1.80GHz** | Hỗ trợ tập lệnh AVX2, FMA, SSE4.2 (rất mạnh cho SIMD) |
| **Số Core / Threads** | **20 cores vật lý / 40 threads** (10 cores/socket, 2 sockets, HT=2) | Có 2 vùng NUMA (NUMA0: 0-9, 20-29; NUMA1: 10-19, 30-39) |
| **Bộ nhớ Cache** | L1: 640KB, L2: 5MB, **L3: 50MB** | Cache L3 rất lớn, tối ưu hóa mạnh cho thuật toán block/Strassen |
| **Dung lượng RAM** | **125 GiB** (trống khả dụng ~87 GiB) | Dư sức chạy ma trận $n = 10.000$ (chỉ tốn ~2.4 GB RAM) |
| **Hệ điều hành (OS)** | **Red Hat Enterprise Linux / Rocky Linux** (Kernel x86_64) | Chuẩn cho tính toán khoa học |
| **Trình biên dịch C++** | **GCC 11.5.0** (`gcc (GCC) 11.5.0 20240719 (Red Hat 11.5.0-14)`) | Hỗ trợ C++17 và OpenMP 4.5/5.0 đầy đủ |
| **Phiên bản MPI** | **OpenMPI 5.0** (`openmpi/5.0`) & Intel MPI 2021 | Nạp qua lệnh: `module load openmpi/5.0` |
| **Thư viện BLAS** | **Intel OneAPI MKL 2026** (`intel-oneapi-mkl/2026`) | Thư viện BLAS chuẩn công nghiệp nhanh nhất cho CPU Intel |

---

## 2. QUY TRÌNH NẠP MÔI TRƯỜNG & CHUẨN BỊ (BẮT BUỘC)

Trên cụm SuperNodeXP, các công cụ được quản lý qua hệ thống **Lmod (Environment Modules)**. Để sử dụng `mpicxx` và `mpirun`, mỗi khi mở terminal mới **bắt buộc phải chạy**:

```bash
module load openmpi/5.0
```
*(Nếu làm việc với BLAS của TV1, nạp thêm: `module load intel-oneapi-mkl/2026`)*.

Kiểm tra sau khi load:
```bash
which mpicxx
mpicxx --version
```

### 2.3. Tạo và kiểm tra file danh sách các Node (`hostfile` / `hosts`)
Tạo file `~/mpi_hosts` chứa danh sách node và số slot (CPU cores):
```text
node01 slots=8
node02 slots=8
node03 slots=8
node04 slots=8
```
Kiểm tra kết nối MPI không cần mật khẩu giữa các node:
```bash
mpirun --hostfile ~/mpi_hosts -np 4 hostname
```
*(Yêu cầu: In ra tên của 4 node khác nhau mà không báo lỗi SSH)*.

---

## 3. BIÊN DỊCH MÃ NGUỒN TRÊN CLUSTER

Từ thư mục gốc dự án:
```bash
mkdir -p bin results/cluster/logs

# 1. Biên dịch Naive MPI & Naive Hybrid (phục vụ TV1 & TV3)
mpicxx -std=c++17 -O3 -march=native -fopenmp -Iinclude \
  src/common/common.cpp src/naive/naive_mpi.cpp -o bin/naive_mpi

mpicxx -std=c++17 -O3 -march=native -fopenmp -Iinclude \
  src/common/common.cpp src/naive/naive_hybrid.cpp -o bin/naive_hybrid

# 2. Biên dịch Strassen MPI & Strassen Hybrid (TV3)
mpicxx -std=c++17 -O3 -march=native -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_mpi.cpp -o bin/strassen_mpi

mpicxx -std=c++17 -O3 -march=native -fopenmp -Iinclude \
  src/common/common.cpp src/strassen/strassen.cpp \
  src/strassen/strassen_hybrid.cpp -o bin/strassen_hybrid
```

---

## 4. QUY TẮC ĐIỀU PHỐI LỊCH CHẠY BENCHMARK (Nhiệm vụ D6)

Để tránh tình trạng 2 thành viên cùng đo đạc một lúc làm tụt hiệu năng (tranh chấp CPU và nghẽn băng thông mạng):

1. **Nguyên tắc số 1**: Tất cả các lượt chạy nặng ($n \ge 2000$) **bắt buộc** phải đăng ký vào Bảng lịch chung của nhóm trước ít nhất 1 giờ.
2. **Nguyên tắc số 2**: Trong lúc một thành viên đang chạy benchmark chính thức, các thành viên khác **chỉ được biên dịch hoặc chạy test kiểm thử nhỏ** ($n \le 1000$).
3. **Mẫu bảng đăng ký lịch (Google Sheets / Wiki nhóm)**:
   * **Cột**: `Ngày` | `Khung giờ (Slot)` | `Người chạy (TV1/TV3)` | `Nhiệm vụ / Binary` | `Ghi chú`
   * *Ví dụ slot khuyến nghị*:
     * Khung sáng: 08:00 – 12:00
     * Khung chiều: 13:00 – 17:00
     * Khung đêm (chạy mẻ lớn $n=10000$): 22:00 – 06:00 sáng hôm sau

---

## 5. HƯỚNG DẪN CHẠY BENCHMARK BẰNG `run_cluster.sh`

### 5.1. Chạy trong phiên `tmux` hoặc `screen` (Tránh rớt mạng làm ngắt lệnh)
```bash
tmux new -s bench_cluster
```

### 5.2. Chạy thử nhanh kiểm tra hệ thống (Sanity Check)
```bash
HOSTFILE=~/mpi_hosts scripts/run_cluster.sh --quick
```

### 5.3. Chạy benchmark chính thức (Full Run)
```bash
# Chạy toàn bộ cấu hình MPI & Hybrid:
HOSTFILE=~/mpi_hosts scripts/run_cluster.sh

# Hoặc chỉ chạy riêng cho Strassen (D7):
HOSTFILE=~/mpi_hosts scripts/run_cluster.sh --only strassen_mpi,strassen_hybrid
```

*Kết quả đo được tự động ghi vào:*
* File dữ liệu: `results/cluster/<hostname>_<YYYYMMDD>.csv`
* File log chi tiết: `results/cluster/logs/<hostname>_<YYYYMMDD>.log`

