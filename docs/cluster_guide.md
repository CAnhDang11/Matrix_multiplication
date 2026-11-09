# HƯỚNG DẪN SỬ DỤNG VÀ KHẢO SÁT CỤM CLUSTER (TV3 · D1 & D6)
**Người phụ trách**: Tô Thế Hưng (TV3) — Điều phối viên Cluster  
**Phạm vi**: Dùng chung cho toàn nhóm (TV1, TV2, TV3) khi chạy các bài toán MPI và Hybrid.

---

## 1. THÔNG SỐ KHẢO SÁT HỆ THỐNG CLUSTER (Nhiệm vụ D1)

*Điền các thông số thực tế của cluster vào bảng này sau khi SSH vào node master:*

| Hạng mục | Chi tiết thực tế khảo sát | Lệnh kiểm tra |
| :--- | :--- | :--- |
| **Số lượng Node** | *(Ví dụ: 4 nodes: node01..node04)* | `sinfo` hoặc xem hostfile |
| **CPU mỗi node** | *(Ví dụ: Intel Xeon E5-2680 v4 @ 2.40GHz)* | `lscpu \| grep "Model name"` |
| **Số Core vật lý/Logic** | *(Ví dụ: 14 cores vật lý / 28 threads)* | `lscpu \| grep -E "Socket\|Core\|Thread"` |
| **Dung lượng RAM** | *(Ví dụ: 64 GB DDR4)* | `free -h` |
| **Hệ điều hành (OS)** | *(Ví dụ: Ubuntu 22.04 LTS / CentOS 7)* | `cat /etc/os-release` |
| **Trình biên dịch C++** | *(Ví dụ: GCC 11.4.0 hoặc GCC 12)* | `g++ --version` |
| **Phiên bản MPI** | *(Ví dụ: OpenMPI 4.1.2 / MPICH 4.0)* | `mpicxx --version` & `mpirun --version` |
| **Thư viện BLAS** | *(Có sẵn OpenBLAS / Intel MKL không)* | `ldconfig -p \| grep blas` |
| **Kiến trúc đồng nhất** | Các node có cùng CPU/RAM hay không? | So sánh `lscpu` giữa các node |

---

## 2. QUY TRÌNH ĐĂNG NHẬP VÀ CHUẨN BỊ MÔI TRƯỜNG

### 2.1. Đăng nhập SSH
```bash
ssh <username>@<cluster_ip_or_domain>
```

### 2.2. Kiểm tra hoặc nạp Modules (nếu cluster dùng Environment Modules / Lmod)
```bash
module avail               # Xem các module có sẵn
module load gcc/11.2.0     # Nạp GCC mới nếu mặc định quá cũ
module load openmpi        # Nạp OpenMPI
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
