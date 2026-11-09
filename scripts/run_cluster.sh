#!/usr/bin/env bash
# =============================================================================
# scripts/run_cluster.sh — Benchmark phan tan MPI/Hybrid tren cluster (TV3 · D5)
# Dung chung cho TV1 (naive_mpi, naive_hybrid) va TV3 (strassen_mpi, strassen_hybrid).
#
# Chay:  scripts/run_cluster.sh [--help]
# Vi du: scripts/run_cluster.sh --quick
#        scripts/run_cluster.sh --only naive_mpi,naive_hybrid
#        HOSTFILE=~/mpi_hosts P_LIST="2 4 8" scripts/run_cluster.sh
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

ulimit -n 4096 2>/dev/null || true
export OMPI_MCA_opal_set_max_sys_limits=1
export PRTE_MCA_prte_set_max_sys_limits=1

# ---------------------------------------------------------------------------
# Tro giup
# ---------------------------------------------------------------------------
usage() {
cat <<'EOF'
Cach dung: scripts/run_cluster.sh [tuy chon]

  (khong tham so)   Benchmark day du tren cluster (P x T, tat ca binary)
  --quick           Chay thu nhanh (n=500 1000, P=1 2, 2 lan lap)
  --dry             Chi in lenh, khong thuc thi
  --only a,b,c      Chi chay cac binary trong danh sach:
                    naive_mpi naive_hybrid strassen_mpi strassen_hybrid
  --n "1000 2000"   Ghi de danh sach kich thuoc ma tran
  -h, --help        Hien tro giup nay

Bien moi truong:
  HOSTFILE=path          File chua danh sach host cho mpirun (mac dinh: khong dung)
  MPIRUN=mpirun          Lenh MPI su dung (mpirun / mpiexec / srun)
  P_LIST="1 2 4 8"       Danh sach so process MPI can quet
  T_LIST="1 2 4"         Danh sach so thread OpenMP moi process
  CUTOFF=128             Strassen cutoff
  DEPTH=2                Strassen OMP task depth
  SEED=42                Seed sinh ma tran
  TIMEOUT=1800           Giay toi da moi cau hinh (mac dinh 30 phut)
  RUNS=3                 So lan chay moi cau hinh (mac dinh 3)
  WARMUP=1               So lan khoi dong (mac dinh 1)
  NO_VERIFY=1            Bo qua kiem chung (nhanh hon khi n lon)
  RESUME=1               Bo qua cau hinh da chay hom nay (mac dinh bat)
  CORES_PER_NODE=        So loi moi node (de tinh oversubscribe; 0=tu phat hien)

Ket qua: results/cluster/<host>_<yyyymmdd>.csv (ghi noi tiep)
Log:     results/cluster/logs/<host>_<yyyymmdd>.log
EOF
}

# ---------------------------------------------------------------------------
# Phan tich tham so dong lenh
# ---------------------------------------------------------------------------
QUICK=0
DRY=0
ONLY=""
N_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)  QUICK=1 ;;
    --dry)    DRY=1 ;;
    --only)   [[ $# -ge 2 ]] || { echo "--only can gia tri" >&2; exit 2; }
              ONLY="$2"; shift ;;
    --n)      [[ $# -ge 2 ]] || { echo "--n can gia tri" >&2; exit 2; }
              N_OVERRIDE="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Tuy chon khong hop le: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Bien moi truong (co the ghi de tu ngoai)
# ---------------------------------------------------------------------------
MPIRUN="${MPIRUN:-mpirun}"
CUTOFF="${CUTOFF:-128}"
DEPTH="${DEPTH:-2}"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-1800}"
WARMUP="${WARMUP:-1}"
VERIFY_FLAG="--verify"
[[ "${NO_VERIFY:-0}" == 1 ]] && VERIFY_FLAG=""
RESUME="${RESUME:-1}"
CORES_PER_NODE="${CORES_PER_NODE:-0}"

# Hostfile (tuy chon)
HOSTFILE_FLAG=""
if [[ -n "${HOSTFILE:-}" && -f "$HOSTFILE" ]]; then
  HOSTFILE_FLAG="--hostfile $HOSTFILE"
  echo "[i] Dung hostfile: $HOSTFILE"
fi

# ---------------------------------------------------------------------------
# Thu muc ket qua
# ---------------------------------------------------------------------------
OS="$(uname -s)"
HOST="$(hostname -s 2>/dev/null || hostname)"
DATE="$(date +%Y%m%d)"
OUT_DIR="results/cluster"
mkdir -p "$OUT_DIR/logs"

CSV="$OUT_DIR/${HOST}_${DATE}.csv"
LOG="$OUT_DIR/logs/${HOST}_${DATE}.log"
DONE_FILE="$OUT_DIR/logs/${HOST}_${DATE}.done"
HEADER="algo,model,n,P,T,run,t_total,t_compute,t_comm,verified,rel_err,cutoff,host,commit"

# Ghi header CSV mot lan
if [[ ! -f "$CSV" ]]; then
  echo "$HEADER" > "$CSV"
fi

# ---------------------------------------------------------------------------
# Danh sach n, P, T, RUNS
# ---------------------------------------------------------------------------
if [[ "$QUICK" == 1 ]]; then
  N_LIST="${N_OVERRIDE:-500 1000}"
  P_LIST="${P_LIST:-1 2}"
  T_LIST="${T_LIST:-1}"
  RUNS="${RUNS:-2}"
else
  N_LIST="${N_OVERRIDE:-500 1000 2000 4000 8000 10000}"
  P_LIST="${P_LIST:-1 2 4 8}"
  T_LIST="${T_LIST:-1 2 4}"
  RUNS="${RUNS:-3}"
fi

# ---------------------------------------------------------------------------
# Danh sach binary se chay
# ---------------------------------------------------------------------------
ALL_BINS="naive_mpi naive_hybrid strassen_mpi strassen_hybrid"

if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a BIN_LIST <<< "$ONLY"
else
  read -r -a BIN_LIST <<< "$ALL_BINS"
fi

# ---------------------------------------------------------------------------
# Ham tien ich
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Kiem tra binary ton tai va co the chay
check_bin() {
  local bin="$1"
  if [[ ! -x "bin/$bin" ]]; then
    log "[SKIP] bin/$bin khong ton tai hoac khong co quyen thuc thi"
    return 1
  fi
  return 0
}

# Xac dinh thong so MPI (binding, oversubscribe)
mpi_flags() {
  local np="$1" nt="$2"
  local flags=""
  [[ -n "$HOSTFILE_FLAG" ]] && flags="$HOSTFILE_FLAG"
  # Tranh oversubscribe khi do hieu nang
  flags="$flags --map-by socket:PE=$nt"
  echo "$flags"
}

# Kiem tra xem cau hinh nay da chay chua (RESUME mode)
already_done() {
  local key="$1"
  [[ "$RESUME" == 1 ]] && grep -qF "$key" "$DONE_FILE" 2>/dev/null
}

mark_done() {
  local key="$1"
  echo "$key" >> "$DONE_FILE"
}

# Chuyen doi mode model (de print_csv dung dung)
model_of() {
  local bin="$1"
  case "$bin" in
    *hybrid*) echo "hybrid" ;;
    *mpi*)    echo "mpi" ;;
    *)        echo "seq" ;;
  esac
}

# Tinh so runs theo n (theo plan §3.9)
runs_for() {
  local n="$1"
  if   (( n >= 6000 )); then echo 3
  elif (( n >= 1000 )); then echo "$RUNS"
  else                       echo "$RUNS"
  fi
}

# ---------------------------------------------------------------------------
# Bat dau benchmark
# ---------------------------------------------------------------------------
log "====== run_cluster.sh ======"
log "Host: $HOST | Ngay: $DATE"
log "BINS: ${BIN_LIST[*]}"
log "N: $N_LIST"
log "P: $P_LIST | T: $T_LIST"
log "TIMEOUT: ${TIMEOUT}s | WARMUP: $WARMUP | RUNS: $RUNS"
log "CSV: $CSV | LOG: $LOG"
echo ""

TOTAL_CONFIGS=0
PASS_CONFIGS=0
SKIP_CONFIGS=0
FAIL_CONFIGS=0
TIMEOUT_CONFIGS=0

for bin in "${BIN_LIST[@]}"; do
  check_bin "$bin" || continue
  local_model=$(model_of "$bin")

  for n in $N_LIST; do
    for P in $P_LIST; do

      # Strassen: bo qua n nho hon nguu'ng cutoff*2 voi nhieu rank
      if [[ "$bin" == strassen_* ]] && (( n < CUTOFF * 2 && P > 1 )); then
        log "[SKIP] $bin n=$n P=$P: n qua nho voi Strassen phan tan"
        continue
      fi

      for T in $T_LIST; do
        # Hybrid: can T > 1 de co y nghia; MPI thuong: T=1
        if [[ "$local_model" == "mpi" ]] && (( T > 1 )); then
          continue
        fi

        TOTAL_CONFIGS=$(( TOTAL_CONFIGS + 1 ))
        local_runs=$(runs_for "$n")
        local_key="${bin}_n${n}_P${P}_T${T}"

        if already_done "$local_key"; then
          log "[SKIP][RESUME] $local_key da chay hom nay"
          SKIP_CONFIGS=$(( SKIP_CONFIGS + 1 ))
          continue
        fi

        # Xay dung lenh chay
        local_flags=""
        if [[ -n "$HOSTFILE_FLAG" ]]; then
          local_flags="$HOSTFILE_FLAG"
        fi

        CMD="$MPIRUN -np $P $local_flags \
          ./bin/$bin \
          -n $n -t $T -r $local_runs \
          -s $SEED --cutoff $CUTOFF --depth $DEPTH \
          --warmup $WARMUP $VERIFY_FLAG"

        log ">>> $local_key | runs=$local_runs"
        if [[ "$DRY" == 1 ]]; then
          echo "    [DRY] $CMD"
          SKIP_CONFIGS=$(( SKIP_CONFIGS + 1 ))
          continue
        fi

        # Chay voi timeout; noi ket qua vao CSV
        set +e
        timeout "$TIMEOUT" bash -c "$CMD >> \"$CSV\" 2>> \"$LOG\""
        exit_status=$?
        set -e

        if (( exit_status == 124 )); then
          log "[TIMEOUT] $local_key sau ${TIMEOUT}s"
          TIMEOUT_CONFIGS=$(( TIMEOUT_CONFIGS + 1 ))
        elif (( exit_status != 0 )); then
          log "[FAIL] $local_key thoat voi ma $exit_status"
          FAIL_CONFIGS=$(( FAIL_CONFIGS + 1 ))
        else
          log "[OK]   $local_key"
          PASS_CONFIGS=$(( PASS_CONFIGS + 1 ))
          mark_done "$local_key"
        fi

      done  # T
    done    # P
  done      # n
done        # bin

# ---------------------------------------------------------------------------
# Tom tat
# ---------------------------------------------------------------------------
echo ""
log "====== Ket qua ======"
log "Tong cau hinh : $TOTAL_CONFIGS"
log "Thanh cong    : $PASS_CONFIGS"
log "Bo qua        : $SKIP_CONFIGS"
log "That bai      : $FAIL_CONFIGS"
log "Timeout       : $TIMEOUT_CONFIGS"
log "CSV output    : $CSV"
log "Log           : $LOG"

# Canh bao neu khong chay tu commit tag v1.0-bench
if git describe --exact-match --match 'v1.0-bench' HEAD >/dev/null 2>&1; then
  log "[OK] HEAD la tag v1.0-bench — so lieu chinh thuc."
else
  log "[!] HEAD KHONG phai tag v1.0-bench — so lieu chi de thu, KHONG dung cho bao cao."
fi

echo ""
log "[i] Cam sac, che do hieu nang cao, dong ung dung nen (plan §3.10)."
log "[i] Ket qua CSV: $CSV"

[[ "$FAIL_CONFIGS" -eq 0 && "$TIMEOUT_CONFIGS" -eq 0 ]]
