#!/usr/bin/env bash
# =============================================================================
# scripts/run_shared.sh — benchmark shared-memory trên 1 máy (TV2 · S5)
# Tuân theo plan v1.1 §3.9 (kích thước, số lần chạy, quét thread) và §3.10.
#
# Dùng chung cho cả nhóm: TV1 (laptop 1), TV2 (laptop 2), TV3 (laptop 3 + 1 node
# cluster). Chạy:  scripts/run_shared.sh --help
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

usage() {
cat <<'EOF'
Cách dùng: scripts/run_shared.sh [tuỳ chọn]

  (không tham số)   Benchmark shared-memory đầy đủ theo plan §3.9:
                    naive_seq (ikj + ijk), naive_omp, strassen_seq,
                    strassen_omp, blas_ref
  --mpi             Chạy naive/strassen _mpi và _hybrid với np <= số lõi vật lý
  --cutoff-sweep    Quét cutoff Strassen (S3) ở n = 1000, 2000, 4000
  --quick           Chạy thử nhanh (n = 100, 500, 1000; 2 lần lặp)
  --dry             Chỉ in các lệnh sẽ chạy
  --only a,b,c      Chỉ chạy các chương trình trong danh sách:
                    naive_seq naive_ijk naive_omp strassen_seq strassen_omp
                    blas_ref naive_mpi naive_hybrid strassen_mpi strassen_hybrid
  --n "1000 2000"   Ghi đè danh sách n
  -h, --help        Hiện trợ giúp này

Biến môi trường (đặt trước lệnh, ví dụ:  CUTOFF=256 scripts/run_shared.sh):
  CUTOFF=128  DEPTH=2  SEED=42       tham số Strassen / seed ma trận
  TIMEOUT=2400                       giây tối đa cho MỖI cấu hình
  MPIRUN=mpirun                      hoặc mpiexec
  CORES_PHYS=4  CORES_LOGICAL=8      ghi đè số lõi tự phát hiện
  MPI_BIND=auto|none                 none = không truyền cờ binding cho mpirun
  NO_VERIFY=1                        không truyền --verify
  RESUME=0                           chạy lại cả cấu hình đã xong hôm nay
  FORCE_MEM=1                        bỏ kiểm tra RAM (dễ bị OOM-kill)
  MEM_FRACTION_PCT=85                % RAM khả dụng cho phép dùng
  STRASSEN_SEQ_MEM_FACTOR=14  STRASSEN_OMP_MEM_FACTOR=40
                                     hệ số bộ nhớ (xem ghi chú trong script)

Kết quả: results/shared/<host>_<yyyymmdd>.csv (header ghi 1 lần, ghi nối tiếp)
Log:     results/shared/logs/<host>_<yyyymmdd>.log
EOF
}

# ---------------------------------------------------------------- tham số CLI
MODE=shared
QUICK=0
DRY=0
ONLY=""
N_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mpi)          MODE=mpi ;;
    --cutoff-sweep) MODE=cutoff ;;
    --quick)        QUICK=1 ;;
    --dry)          DRY=1 ;;
    --only)         [[ $# -ge 2 ]] || { echo "--only cần giá trị" >&2; exit 2; }
                    ONLY="$2"; shift ;;
    --n)            [[ $# -ge 2 ]] || { echo "--n cần giá trị" >&2; exit 2; }
                    N_OVERRIDE="$2"; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "Tuỳ chọn không hợp lệ: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------ biến môi trường
CUTOFF="${CUTOFF:-128}"
DEPTH="${DEPTH:-2}"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-2400}"
MPIRUN="${MPIRUN:-mpirun}"
MPI_BIND="${MPI_BIND:-auto}"
RESUME="${RESUME:-1}"
MEM_FRACTION_PCT="${MEM_FRACTION_PCT:-85}"
VERIFY_FLAG="--verify"
[[ "${NO_VERIFY:-0}" == 1 ]] && VERIFY_FLAG=""

# Biến này từng được export để chạy test đúng/sai; khi benchmark thì KHÔNG được
# cho phép oversubscribe, nếu không số liệu sai.
unset OMPI_MCA_rmaps_base_oversubscribe

# -------------------------------------------------------------------- máy/host
OS="$(uname -s)"
HOST="$(hostname -s 2>/dev/null || hostname)"
DATE="$(date +%Y%m%d)"
OUT_DIR="results/shared"
CSV="$OUT_DIR/${HOST}_${DATE}.csv"
LOG="$OUT_DIR/logs/${HOST}_${DATE}.log"
DONE="$OUT_DIR/logs/${HOST}_${DATE}.done"
HEADER="algo,model,n,P,T,run,t_total,t_compute,t_comm,verified,rel_err,cutoff,host,commit"

detect_cpu() {
  local phys="${CORES_PHYS:-}" logi="${CORES_LOGICAL:-}"
  if [[ "$OS" == Darwin ]]; then
    [[ -n "$phys" ]] || phys="$(sysctl -n hw.physicalcpu)"
    [[ -n "$logi" ]] || logi="$(sysctl -n hw.logicalcpu)"
  else
    [[ -n "$logi" ]] || logi="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc --all)"
    if [[ -z "$phys" ]]; then
      phys=$(( $(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l) ))
      (( phys >= 1 )) || phys="$logi"
    fi
  fi
  CORES_PHYS="$phys"
  CORES_LOGICAL="$logi"
}
detect_cpu

avail_kb() {   # RAM khả dụng (kB); 0 nếu không xác định được
  if [[ -r /proc/meminfo ]]; then
    awk '/^MemAvailable:/ {print $2}' /proc/meminfo
  elif [[ "$OS" == Darwin ]]; then
    echo $(( $(sysctl -n hw.memsize) / 1024 * 7 / 10 ))
  else
    echo 0
  fi
}

# -------------------------------------------------- danh sách n / thread / run
if [[ "$QUICK" == 1 ]]; then
  N_LIST="100 500 1000"; SWEEP_N="1000"
  MPI_N_LIST="500 1000"
  CS_N_LIST="500 1000"; CUTOFF_LIST="64 128"
else
  N_LIST="100 500 1000 2000 4000 6000 8000 10000"
  SWEEP_N="2000 4000 8000"          # chỉ ở các n này mới quét đủ số thread (§3.9)
  MPI_N_LIST="500 1000 2000 4000"   # quy mô nhỏ hơn cho MPI trên laptop
  CS_N_LIST="1000 2000 4000"
  CUTOFF_LIST="32 64 128 256 512"
fi
[[ -n "$N_OVERRIDE" ]] && { N_LIST="$N_OVERRIDE"; MPI_N_LIST="$N_OVERRIDE"; CS_N_LIST="$N_OVERRIDE"; }

pow2_upto() {  # "1 2 4 ..." các lũy thừa của 2 không vượt quá $1
  local t=1 out=""
  while (( t <= $1 )); do out="$out $t"; t=$(( t * 2 )); done
  echo $out
}
uniq_sorted() { tr ' ' '\n' | grep -v '^$' | sort -nu | tr '\n' ' '; }
thread_sweep() { echo "$(pow2_upto "$CORES_PHYS") $CORES_PHYS $CORES_LOGICAL" | uniq_sorted; }
is_sweep_n() { case " $SWEEP_N " in *" $1 "*) return 0 ;; esac; return 1; }
threads_for() {
  if is_sweep_n "$1"; then thread_sweep; else echo "1 $CORES_PHYS" | uniq_sorted; fi
}

# §3.9: 5 lần khi n <= 4000; 3 lần khi n >= 6000; 1 lần cho cấu hình tuần tự
# (hoặc T=1/P=1) khi n >= 8000. $1 = n, $2 = mức song song (T hoặc P*T).
runs_for() {
  local n="$1" par="$2"
  if [[ "$QUICK" == 1 ]]; then echo 2
  elif (( n >= 8000 && par == 1 )); then echo 1
  elif (( n <= 4000 )); then echo 5
  else echo 3; fi
}
# Khi chỉ chạy 1 lần ở n lớn thì bỏ warm-up để không nhân đôi thời gian.
warm_for() { if (( $1 >= 8000 && $2 == 1 )); then echo 0; else echo 1; fi; }

# --------------------------------------------------------------- kiểm tra RAM
# Hệ số = (bộ nhớ đỉnh) / (n^2 * 8 byte). ƯỚC LƯỢNG THÔ, cần hiệu chỉnh bằng
# /usr/bin/time -v sau khi tối ưu bộ nhớ Strassen:
#   naive_seq/omp, blas_ref : A, B, C                        -> 3.5
#   strassen_seq            : 3 + 31 buffer (n/2)^2 cộng dồn qua các tầng
#   strassen_omp            : nhiều nhánh task chạy đồng thời -> rất lớn
#   *_mpi, *_hybrid         : rank 0 giữ A,B,C; mỗi rank giữ thêm B
mem_factor() {  # $1 = chương trình, $2 = P
  case "$1" in
    naive_seq|naive_ijk|naive_omp|blas_ref) echo 3.5 ;;
    strassen_seq)    echo "${STRASSEN_SEQ_MEM_FACTOR:-14}" ;;
    strassen_omp)    echo "${STRASSEN_OMP_MEM_FACTOR:-40}" ;;
    naive_mpi|naive_hybrid)       awk -v p="$2" 'BEGIN{print p+5}' ;;
    strassen_mpi|strassen_hybrid) awk -v p="$2" 'BEGIN{print p+20}' ;;
    *) echo 10 ;;
  esac
}
need_mb() {  # $1 = chương trình, $2 = n, $3 = P
  awk -v f="$(mem_factor "$1" "$3")" -v n="$2" 'BEGIN{printf "%.0f", f*n*n*8/1048576}'
}
fits_memory() {
  [[ "${FORCE_MEM:-0}" == 1 ]] && return 0
  local avail; avail="$(avail_kb)"
  (( avail > 0 )) || return 0
  awk -v need="$(need_mb "$1" "$2" "$3")" -v avail="$avail" -v pct="$MEM_FRACTION_PCT" \
    'BEGIN{ exit !(need*1024 <= avail*pct/100) }'
}

# ------------------------------------------------------- git / điều kiện đo
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi
export GIT_COMMIT
GIT_TAG="$(git describe --tags --exact-match 2>/dev/null || true)"

# §3.10: ghim thread. Với MPI + binding thì OpenMP kế thừa mặt nạ affinity của
# rank; nếu không ghim bằng mpirun thì phải bỏ ghim OpenMP, tránh các rank
# chồng lên cùng một nhóm lõi.
export OMP_PROC_BIND=close OMP_PLACES=cores
if [[ "$MPI_BIND" == auto ]]; then
  if [[ "$OS" == Darwin ]]; then MPI_BIND=none; else MPI_BIND=bind; fi
fi
if [[ "$MODE" == mpi && "$MPI_BIND" == none ]]; then
  unset OMP_PROC_BIND OMP_PLACES
fi

if command -v timeout >/dev/null 2>&1; then TO_BIN=timeout
elif command -v gtimeout >/dev/null 2>&1; then TO_BIN=gtimeout
else TO_BIN=""; fi
with_timeout() { if [[ -n "$TO_BIN" ]]; then "$TO_BIN" "$TIMEOUT" "$@"; else "$@"; fi; }

# ------------------------------------------------------- chương trình có sẵn
ALL_BINS="naive_seq naive_omp naive_mpi naive_hybrid strassen_seq strassen_omp strassen_mpi strassen_hybrid blas_ref"
AVAILABLE=" "
for b in $ALL_BINS; do [[ -x "bin/$b" ]] && AVAILABLE="${AVAILABLE}${b} "; done
have() { case "$AVAILABLE" in *" $1 "*) return 0 ;; esac; return 1; }
want() {
  [[ -z "$ONLY" ]] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}
use() { want "$1" && have "${2:-$1}"; }   # use <tên> [binary]

# ---------------------------------------------------------------- chạy 1 cấu hình
N_OK=0; N_FAIL=0; N_SKIP=0
run_cfg() {
  # run_cfg <key> <chương trình> <P cho ước lượng RAM> <n> <lệnh...>
  local key="$1" prog="$2" mem_p="$3" n="$4"; shift 4

  if [[ "$RESUME" == 1 && "$DRY" == 0 ]] && grep -qxF "$key" "$DONE" 2>/dev/null; then
    echo "[done ] $key"; N_SKIP=$(( N_SKIP + 1 )); return 0
  fi
  if ! fits_memory "$prog" "$n" "$mem_p"; then
    local msg
    msg="[SKIP ] $key — cần ≈ $(need_mb "$prog" "$n" "$mem_p") MB, khả dụng ≈ $(( $(avail_kb) / 1024 )) MB"
    echo "$msg"; [[ "$DRY" == 0 ]] && echo "$msg" >> "$LOG"
    N_SKIP=$(( N_SKIP + 1 )); return 0
  fi
  if [[ "$DRY" == 1 ]]; then echo "[dry  ] $key :: $*"; return 0; fi

  echo "[run  ] $key"
  echo "### $(date '+%F %T') $key :: $*" >> "$LOG"
  local tmp t0 rc dt
  tmp="$(mktemp)"; t0=$SECONDS
  with_timeout "$@" > "$tmp" 2>> "$LOG"; rc=$?
  grep -v '^algo,' "$tmp" >> "$CSV"
  rm -f "$tmp"
  dt=$(( SECONDS - t0 ))

  case "$rc" in
    0)   echo "        ok (${dt}s)"; echo "$key" >> "$DONE"; N_OK=$(( N_OK + 1 )) ;;
    124) echo "        TIMEOUT sau ${TIMEOUT}s — bỏ qua cấu hình này (tăng TIMEOUT hoặc giảm số lần chạy)"
         echo "TIMEOUT $key" >> "$LOG"; N_FAIL=$(( N_FAIL + 1 )) ;;
    1)   echo "        FAIL: verify không đạt (xem $LOG)"
         echo "VERIFY-FAIL $key" >> "$LOG"; N_FAIL=$(( N_FAIL + 1 )) ;;
    *)   echo "        FAIL: mã thoát $rc (xem $LOG)"
         echo "EXIT-$rc $key" >> "$LOG"; N_FAIL=$(( N_FAIL + 1 )) ;;
  esac
  return 0
}

# ------------------------------------------------------------- chế độ shared
mode_shared() {
  local n t r w
  for n in $N_LIST; do
    if use naive_seq; then
      r=$(runs_for "$n" 1); w=$(warm_for "$n" "$r")
      run_cfg "naive_seq|ikj|n=$n" naive_seq 1 "$n" \
        ./bin/naive_seq -n "$n" -t 1 -r "$r" -s "$SEED" --warmup "$w" --kernel ikj $VERIFY_FLAG
    fi
    # ijk thuần chỉ làm mốc "naive sách giáo khoa" cho n <= 2000 (§3.9)
    if (( n <= 2000 )) && use naive_ijk naive_seq; then
      r=$(runs_for "$n" 1); w=$(warm_for "$n" "$r")
      run_cfg "naive_seq|ijk|n=$n" naive_ijk 1 "$n" \
        ./bin/naive_seq -n "$n" -t 1 -r "$r" -s "$SEED" --warmup "$w" --kernel ijk $VERIFY_FLAG
    fi
    if use strassen_seq; then
      r=$(runs_for "$n" 1); w=$(warm_for "$n" "$r")
      run_cfg "strassen_seq|c=$CUTOFF|n=$n" strassen_seq 1 "$n" \
        ./bin/strassen_seq -n "$n" -t 1 -r "$r" -s "$SEED" --warmup "$w" --cutoff "$CUTOFF" $VERIFY_FLAG
    fi

    for t in $(threads_for "$n"); do
      r=$(runs_for "$n" "$t"); w=$(warm_for "$n" "$t")
      if use naive_omp; then
        run_cfg "naive_omp|T=$t|n=$n" naive_omp 1 "$n" \
          ./bin/naive_omp -n "$n" -t "$t" -r "$r" -s "$SEED" --warmup "$w" $VERIFY_FLAG
      fi
      if use strassen_omp; then
        run_cfg "strassen_omp|c=$CUTOFF|d=$DEPTH|T=$t|n=$n" strassen_omp 1 "$n" \
          ./bin/strassen_omp -n "$n" -t "$t" -r "$r" -s "$SEED" --warmup "$w" \
            --cutoff "$CUTOFF" --depth "$DEPTH" $VERIFY_FLAG
      fi
      if use blas_ref; then
        run_cfg "blas_ref|T=$t|n=$n" blas_ref 1 "$n" \
          env OPENBLAS_NUM_THREADS="$t" OMP_NUM_THREADS="$t" \
          ./bin/blas_ref -n "$n" -t "$t" -r "$r" -s "$SEED" --warmup "$w" $VERIFY_FLAG
      fi
    done
  done
}

# ------------------------------------------------------------ chế độ cutoff (S3)
mode_cutoff() {
  local n c r w
  for n in $CS_N_LIST; do
    for c in $CUTOFF_LIST; do
      if use strassen_seq; then
        r=$(runs_for "$n" 1); w=$(warm_for "$n" "$r")
        run_cfg "strassen_seq|c=$c|n=$n" strassen_seq 1 "$n" \
          ./bin/strassen_seq -n "$n" -t 1 -r "$r" -s "$SEED" --warmup "$w" --cutoff "$c" $VERIFY_FLAG
      fi
      if use strassen_omp; then
        r=$(runs_for "$n" "$CORES_PHYS"); w=$(warm_for "$n" "$CORES_PHYS")
        run_cfg "strassen_omp|c=$c|d=$DEPTH|T=$CORES_PHYS|n=$n" strassen_omp 1 "$n" \
          ./bin/strassen_omp -n "$n" -t "$CORES_PHYS" -r "$r" -s "$SEED" --warmup "$w" \
            --cutoff "$c" --depth "$DEPTH" $VERIFY_FLAG
      fi
    done
  done
}

# --------------------------------------------------------------- chế độ MPI
bind_opts() {  # $1 = số thread mỗi rank
  [[ "$MPI_BIND" == none ]] && return 0
  if (( $1 > 1 )); then echo "--map-by slot:PE=$1 --bind-to core"; else echo "--bind-to core"; fi
}
hybrid_configs() {  # "P:T ..." với P>=2, T>=2, P*T = số lõi vật lý
  local p=2 out=""
  while (( p * 2 <= CORES_PHYS )); do
    (( CORES_PHYS % p == 0 )) && out="$out $p:$(( CORES_PHYS / p ))"
    p=$(( p * 2 ))
  done
  echo $out
}
mode_mpi() {
  local n p t cfg r w extra
  local plist; plist="$(echo "$(pow2_upto "$CORES_PHYS") $CORES_PHYS" | uniq_sorted)"
  for n in $MPI_N_LIST; do
    for prog in naive_mpi strassen_mpi; do
      use "$prog" || continue
      extra=""; [[ "$prog" == strassen_mpi ]] && extra="--cutoff $CUTOFF"
      for p in $plist; do
        r=$(runs_for "$n" "$p"); w=$(warm_for "$n" "$p")
        run_cfg "$prog|P=$p|T=1|n=$n" "$prog" "$p" "$n" \
          "$MPIRUN" -np "$p" $(bind_opts 1) ./bin/"$prog" \
            -n "$n" -t 1 -r "$r" -s "$SEED" --warmup "$w" $extra $VERIFY_FLAG
      done
    done
    for prog in naive_hybrid strassen_hybrid; do
      use "$prog" || continue
      extra=""; [[ "$prog" == strassen_hybrid ]] && extra="--cutoff $CUTOFF --depth $DEPTH"
      for cfg in $(hybrid_configs); do
        p="${cfg%%:*}"; t="${cfg##*:}"
        r=$(runs_for "$n" $(( p * t ))); w=$(warm_for "$n" $(( p * t )))
        run_cfg "$prog|P=$p|T=$t|n=$n" "$prog" "$p" "$n" \
          "$MPIRUN" -np "$p" $(bind_opts "$t") ./bin/"$prog" \
            -n "$n" -t "$t" -r "$r" -s "$SEED" --warmup "$w" $extra $VERIFY_FLAG
      done
    done
  done
}

# ------------------------------------------------------------ tóm tắt đầu chạy
echo "=============================================================="
echo " run_shared.sh — chế độ: $MODE$([[ $QUICK == 1 ]] && echo ' (quick)')$([[ $DRY == 1 ]] && echo ' (dry)')"
echo " host            : $HOST ($OS)"
echo " số lõi          : $CORES_PHYS vật lý / $CORES_LOGICAL logic   <-- KIỂM TRA, sai thì đặt CORES_PHYS=…"
echo " RAM khả dụng    : $(( $(avail_kb) / 1024 )) MB (cho phép dùng ${MEM_FRACTION_PCT}%)"
echo " commit          : $GIT_COMMIT   tag: ${GIT_TAG:-<không có>}"
echo " cutoff / depth  : $CUTOFF / $DEPTH      verify: ${VERIFY_FLAG:-tắt}      timeout: ${TIMEOUT}s/cấu hình"
case "$MODE" in
  shared) echo " n               : $N_LIST"
          echo " quét thread ở n : $SWEEP_N  ->  T = $(thread_sweep)" ;;
  cutoff) echo " n               : $CS_N_LIST    cutoff: $CUTOFF_LIST" ;;
  mpi)    echo " n               : $MPI_N_LIST    P = $(echo "$(pow2_upto "$CORES_PHYS") $CORES_PHYS" | uniq_sorted)"
          echo " hybrid (P:T)    : $(hybrid_configs)    binding: $MPI_BIND    launcher: $MPIRUN" ;;
esac
echo " ghi vào         : $CSV"
echo "=============================================================="

# Cảnh báo điều kiện đo
if [[ "$GIT_TAG" != "v1.0-bench" ]]; then
  echo "[!] HEAD không phải tag v1.0-bench -> số liệu chỉ để thử, KHÔNG dùng cho báo cáo (plan §3.9)."
fi
case "$ROOT" in /mnt/*) echo "[!] Repo đang nằm trên /mnt/... (ổ Windows): I/O chậm, nên để trong ~ của WSL." ;; esac
[[ "$GIT_COMMIT" == *-dirty ]] && echo "[!] Có thay đổi chưa commit: cột commit trong CSV sẽ có hậu tố -dirty."
echo "[i] Cắm sạc, chế độ hiệu năng cao, đóng ứng dụng nền (plan §3.10)."

# Báo chương trình thiếu (chỉ những cái chế độ hiện tại cần)
case "$MODE" in
  shared) NEEDED="naive_seq naive_omp strassen_seq strassen_omp blas_ref" ;;
  cutoff) NEEDED="strassen_seq strassen_omp" ;;
  mpi)    NEEDED="naive_mpi naive_hybrid strassen_mpi strassen_hybrid" ;;
esac
for b in $NEEDED; do
  have "$b" || echo "[miss ] bin/$b chưa build -> bỏ qua (xem README để biết lệnh build)"
done
if [[ "$MODE" == mpi ]] && ! command -v "$MPIRUN" >/dev/null 2>&1; then
  echo "Không tìm thấy '$MPIRUN' (thử MPIRUN=mpiexec)." >&2; exit 1
fi
echo

# ------------------------------------------------------------------ chạy thật
if [[ "$DRY" == 0 ]]; then
  mkdir -p "$OUT_DIR/logs"
  [[ -s "$CSV" ]] || echo "$HEADER" > "$CSV"
fi
trap 'echo; echo "Đã dừng giữa chừng. Chạy lại cùng lệnh để tiếp tục (RESUME=1)."; exit 130' INT TERM

case "$MODE" in
  shared) mode_shared ;;
  cutoff) mode_cutoff ;;
  mpi)    mode_mpi ;;
esac

echo
echo "=============================================================="
echo " Xong: $N_OK ok · $N_FAIL lỗi · $N_SKIP bỏ qua/đã có"
[[ "$DRY" == 0 ]] && echo " CSV: $CSV   ·   log: $LOG"
echo " Nhớ chạy scripts/sysinfo.sh để lưu cấu hình máy cạnh file CSV."
echo "=============================================================="
(( N_FAIL == 0 ))
