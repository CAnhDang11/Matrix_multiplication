* **Bước 1**: Kiểm tra compiler hiện tại
```bash
uname -s
which g++
g++ --version
which clang++
```
* **Bước 2**: Cài GNU GCC và OpenMP
```bash
brew install gcc libomp
```
Kiểm tra phiên bản GCC vừa cài:
```bash
ls "$(brew --prefix)"/bin/g++-*
```
Trên Apple Silicon, GCC có thể xuất hiện dưới tên g++-15, g++-16 hoặc phiên bản khác. Chọn đúng tên được liệt kê.
```bash
g++-16 -fopenmp /tmp/test_omp.cpp -o /tmp/test_omp
/tmp/test_omp
```

* Cài đặt mpi: 
```bash
brew install open-mpi
```

Sau đó kiểm tra:
```bash
which mpicxx
mpicxx --version
```

Cài OpenBLAS trên macOS
Chạy:
```bash
brew install openblas
```

Lấy đường dẫn cài đặt:
```bash
brew --prefix openblas
```

Trên Apple Silicon, thường là /opt/homebrew/opt/openblas.
