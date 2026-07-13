#!/usr/bin/env bash
#
# gpu-oc-validate.sh — interactive NVIDIA VRAM overclock validator.
#
# Run with NO arguments. It shows a menu: pick a GPU, then pick an action
# (baseline / test an offset / soak / read-only check). It validates BOTH
# axes that matter:
#   1) STABILITY -> zero memory-pattern errors AND no new kernel Xid faults
#   2) BENEFIT   -> bandwidth did not regress vs a saved stock baseline
#                   (catches GDDR7's silent error-correction slowdown)
#
# A tiny CUDA memtest+bandwidth probe is compiled on first run (nvcc+gcc).
# Offsets are read/set via NVML (ctypes, no pip). Any applied offset is ALWAYS
# reverted before returning to the menu, so a bad OC never persists.
set -uo pipefail

NVCC="${NVCC:-/opt/cuda/bin/nvcc}"
REAL_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"
CACHE="${REAL_HOME:-$HOME}/.cache/gpu-oc-validate"
SRC="$CACHE/probe.cu"
BIN="$CACHE/probe"
BASELINE="$CACHE/baseline.tsv"   # <pci_bus_id>\t<bandwidth_gbs>
mkdir -p "$CACHE"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

# ---- NVML offset helpers (ctypes) -------------------------------------------
nvml_get() {  # $1=index -> prints "<core> <mem>"
  python3 - "$1" <<'PY'
import ctypes as C, sys
n = C.CDLL("libnvidia-ml.so.1"); n.nvmlInit_v2()
h = C.c_void_p(); n.nvmlDeviceGetHandleByIndex_v2(int(sys.argv[1]), C.byref(h))
g = C.c_int(); m = C.c_int()
n.nvmlDeviceGetGpcClkVfOffset(h, C.byref(g)); n.nvmlDeviceGetMemClkVfOffset(h, C.byref(m))
print(g.value, m.value)
PY
}
nvml_set() {  # $1=index $2=core $3=mem -> nonzero on failure
  python3 - "$1" "$2" "$3" <<'PY'
import ctypes as C, sys
n = C.CDLL("libnvidia-ml.so.1"); n.nvmlInit_v2()
h = C.c_void_p(); n.nvmlDeviceGetHandleByIndex_v2(int(sys.argv[1]), C.byref(h))
rc1 = n.nvmlDeviceSetGpcClkVfOffset(h, int(sys.argv[2]))
rc2 = n.nvmlDeviceSetMemClkVfOffset(h, int(sys.argv[3]))
sys.exit(0 if rc1 == 0 and rc2 == 0 else 1)
PY
}

# ---- compile probe on first use ---------------------------------------------
compile_probe() {
  [[ -x "$BIN" && "$0" -ot "$BIN" ]] && return 0
  [[ -x "$NVCC" ]] || { echo "nvcc not found at $NVCC (set NVCC=...)" >&2; return 2; }
  cat > "$SRC" <<'CU'
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  printf("FATAL\n"); exit(2);} }while(0)
__global__ void fillK(uint64_t* p,size_t n,uint64_t pat,int addr){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=s) p[i]=addr?((uint64_t)i^pat):pat; }
__global__ void checkK(uint64_t* p,size_t n,uint64_t pat,int addr,unsigned long long* e){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=s){uint64_t x=addr?((uint64_t)i^pat):pat; if(p[i]!=x) atomicAdd(e,1ULL);} }
__global__ void bwK(uint64_t* d,const uint64_t* src,size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=s) d[i]=src[i]+1; }
int main(int argc,char**argv){
  double frac=argc>1?atof(argv[1]):0.85; int iters=argc>2?atoi(argv[2]):3;
  CK(cudaSetDevice(0));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0)); printf("DEVICE %s\n",pr.name);
  size_t freeB,totB; CK(cudaMemGetInfo(&freeB,&totB));
  size_t bytes=(size_t)(freeB*frac); bytes&=~((size_t)(64ull<<20)-1);
  size_t n=bytes/sizeof(uint64_t);
  if(bytes==0){printf("MEMTEST_MB 0\nERRORS 0\nBANDWIDTH_GBS 0.0\n");
    fprintf(stderr,"not enough free VRAM to memtest\n"); return 0;}
  uint64_t* buf; CK(cudaMalloc(&buf,bytes));
  unsigned long long* dErr; CK(cudaMalloc(&dErr,sizeof(*dErr))); CK(cudaMemset(dErr,0,sizeof(*dErr)));
  const int B=256,G=4096;
  const uint64_t pats[4]={0ull,~0ull,0xAAAAAAAAAAAAAAAAull,0x5555555555555555ull};
  for(int it=0;it<iters;++it){
    for(int p=0;p<4;++p){fillK<<<G,B>>>(buf,n,pats[p],0); checkK<<<G,B>>>(buf,n,pats[p],0,dErr);}
    fillK<<<G,B>>>(buf,n,0xD15EA5Eull,1); checkK<<<G,B>>>(buf,n,0xD15EA5Eull,1,dErr);
    CK(cudaDeviceSynchronize());
  }
  unsigned long long errs=0; CK(cudaMemcpy(&errs,dErr,sizeof(errs),cudaMemcpyDeviceToHost));
  printf("MEMTEST_MB %zu\n",bytes>>20); printf("ERRORS %llu\n",errs);
  cudaFree(buf); cudaFree(dErr);
  CK(cudaMemGetInfo(&freeB,&totB));
  size_t bb=(size_t)(freeB*0.40); if(bb>(2ull<<30)) bb=(2ull<<30); bb&=~((size_t)(64ull<<20)-1);
  size_t bn=bb/sizeof(uint64_t);
  uint64_t *src,*dst; CK(cudaMalloc(&src,bb)); CK(cudaMalloc(&dst,bb)); CK(cudaMemset(src,1,bb));
  cudaEvent_t a,z; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&z));
  const int R=50; bwK<<<G,B>>>(dst,src,bn); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a)); for(int r=0;r<R;++r) bwK<<<G,B>>>(dst,src,bn);
  CK(cudaEventRecord(z)); CK(cudaEventSynchronize(z));
  float ms=0; CK(cudaEventElapsedTime(&ms,a,z));
  printf("BANDWIDTH_GBS %.1f\n",(2.0*(double)bb*R)/(ms/1e3)/1e9);
  return 0;
}
CU
  echo ">> Compiling probe (first run)..."
  "$NVCC" -O3 -std=c++20 -allow-unsupported-compiler -o "$BIN" "$SRC" \
    || { echo "compile failed" >&2; return 2; }
}

# ---- one probe run + verdict ------------------------------------------------
# args: $1=gpu $2=frac $3=iters $4=mode(baseline|test)
# sets global VERDICT=PASS|FAIL
run_and_verdict() {
  local gpu="$1" frac="$2" iters="$3" mode="$4"
  local pci start out mb errs gbs xid_lines xid_n base
  local clkfile sampler mem_now mem_max mem_pct
  pci="$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader)"
  start="$(date '+%Y-%m-%d %H:%M:%S')"
  echo ">> Probing GPU $gpu (frac=$frac iters=$iters)..."
  # sample the REAL memory clock while the probe runs, so we can report the
  # card's actual memory speed (proves it hit full clock / wasn't throttling)
  clkfile="$(mktemp)"
  ( while :; do
      nvidia-smi -i "$gpu" --query-gpu=clocks.current.memory --format=csv,noheader,nounits 2>/dev/null
      sleep 0.2
    done ) >"$clkfile" &
  sampler=$!
  out="$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$BIN" "$frac" "$iters" 2>/dev/null)" \
    || { kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null; rm -f "$clkfile"
         echo "  !! probe crashed (likely unstable OC)"; VERDICT=FAIL; return; }
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  mem_now="$(grep -E '^[0-9]+$' "$clkfile" | sort -n | tail -1)"; rm -f "$clkfile"
  mem_max="$(nvidia-smi -i "$gpu" --query-gpu=clocks.max.memory --format=csv,noheader,nounits 2>/dev/null)"
  mb="$(awk '/^MEMTEST_MB/{print $2}' <<<"$out")"
  errs="$(awk '/^ERRORS/{print $2}' <<<"$out")"
  gbs="$(awk '/^BANDWIDTH_GBS/{print $2}' <<<"$out")"
  xid_lines="$(journalctl -k --since "$start" 2>/dev/null | grep -iE 'Xid' | grep -i "${pci#0000:}" || true)"
  xid_n="$(printf '%s' "$xid_lines" | grep -c . || true)"

  echo "   VRAM tested ${mb} MB | errors ${errs} | bandwidth ${gbs} GB/s | new Xid ${xid_n}"
  if [[ -n "${mem_now:-}" && "${mem_max:-0}" =~ ^[0-9]+$ && "${mem_max:-0}" -gt 0 ]]; then
    mem_pct="$(awk -v a="$mem_now" -v b="$mem_max" 'BEGIN{printf "%d",a*100/b}')"
    if [[ "$mem_pct" -ge 95 ]]; then
      echo "   memory ran at ${mem_now} MHz of ${mem_max} MHz max (${mem_pct}%) — full speed, not throttled"
    else
      echo "   memory ran at ${mem_now} MHz of ${mem_max} MHz max (${mem_pct}%) — LOW, card was throttling"
    fi
  else
    echo "   memory clock: could not read from nvidia-smi"
  fi

  base=""; [[ -f "$BASELINE" ]] && base="$(awk -F'\t' -v k="$pci" '$1==k{print $2}' "$BASELINE" | tail -1)"

  if [[ "$mode" == "baseline" ]]; then
    local tmp; tmp="$(mktemp)"
    { [[ -f "$BASELINE" ]] && grep -vP "^\Q$pci\E\t" "$BASELINE" || true; printf '%s\t%s\n' "$pci" "$gbs"; } > "$tmp"
    mv "$tmp" "$BASELINE"
    echo "   baseline saved: ${gbs} GB/s (stock reference for $pci)"
  fi

  VERDICT=PASS
  [[ "${mb:-0}" -eq 0 ]] && { echo "   INCONCLUSIVE: 0 MB tested — idle the GPU to free VRAM"; VERDICT=FAIL; }
  [[ "${errs:-1}" != "0" ]] && { echo "   FAIL: ${errs} memory errors (unstable)"; VERDICT=FAIL; }
  [[ "${xid_n:-0}" -gt 0 ]] && { echo "   FAIL: Xid fault(s):"; echo "$xid_lines" | sed 's/^/     /'; VERDICT=FAIL; }
  if [[ "$mode" == "test" && -n "$base" ]]; then
    if [[ "$(awk -v g="$gbs" -v b="$base" 'BEGIN{print (g<b*0.99)?1:0}')" == "1" ]]; then
      echo "   FAIL: bandwidth ${gbs} < baseline ${base} GB/s — no benefit (error-correction slowdown)"; VERDICT=FAIL
    else
      echo "   OK: bandwidth ${gbs} vs baseline ${base} (+$(awk -v g="$gbs" -v b="$base" 'BEGIN{printf "%.1f",(g-b)/b*100}')%)"
    fi
  elif [[ "$mode" == "test" && -z "$base" ]]; then
    echo "   NOTE: no baseline for this GPU — set one to enable regression checks"
  fi
}

# ---- apply offsets, validate, always revert ---------------------------------
# args: $1=gpu $2=core $3=mem $4=iters $5=soak_minutes(0=single)
apply_validate() {
  local gpu="$1" core="$2" mem="$3" iters="$4" soak="$5" prev_c prev_m deadline cycles
  if [[ $EUID -ne 0 ]]; then
    echo "!! Applying offsets needs root. Re-run:  sudo $0"; return
  fi
  read -r prev_c prev_m < <(nvml_get "$gpu")
  echo ">> Applying core=${core} mem=${mem} MHz (was core=${prev_c} mem=${prev_m})"
  # revert on normal return AND on Ctrl-C
  trap 'nvml_set "'"$gpu"'" "'"$prev_c"'" "'"$prev_m"'" >/dev/null 2>&1; echo; echo ">> reverted"; trap - INT; return' INT
  nvml_set "$gpu" "$core" "$mem" || { echo "!! failed to apply offsets"; trap - INT; return; }

  if [[ "$soak" -gt 0 ]]; then
    deadline=$(( $(date +%s) + soak*60 )); cycles=0
    echo ">> Soaking ${soak} min (Ctrl-C to abort)..."
    while [[ $(date +%s) -lt $deadline ]]; do
      cycles=$((cycles+1)); echo "-- cycle $cycles --"
      run_and_verdict "$gpu" 0.85 "$iters" test
      [[ "$VERDICT" != "PASS" ]] && break
    done
    echo ">> Soak ended after $cycles cycle(s): last verdict $VERDICT"
  else
    run_and_verdict "$gpu" 0.85 "$iters" test
  fi

  nvml_set "$gpu" "$prev_c" "$prev_m" >/dev/null 2>&1 && echo ">> reverted to core=${prev_c} mem=${prev_m}"
  trap - INT
  if [[ "$VERDICT" == "PASS" ]]; then
    echo ">> PASS. To keep core=${core} mem=${mem} (minus a safety margin), set it via LACT or a boot service."
  fi
}

# ---- menus ------------------------------------------------------------------
pick_gpu() {
  local n idx
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader)
  echo "GPUs:"
  for i in "${!GPUS[@]}"; do echo "  $i) ${GPUS[$i]}"; done
  n=${#GPUS[@]}
  while :; do
    read -rp "Select GPU [0-$((n-1))]: " idx
    [[ "$idx" =~ ^[0-9]+$ && "$idx" -lt "$n" ]] && { GPU="${GPUS[$idx]%%,*}"; return; }
    echo "invalid."
  done
}

action_menu() {
  local gc gm core mem iters soak
  while :; do
    read -r gc gm < <(nvml_get "$GPU")
    echo
    echo "=== GPU $GPU  [$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)] ==="
    echo "    current offsets: core=${gc}MHz mem=${gm}MHz | root=$([[ $EUID -eq 0 ]] && echo yes || echo NO)"
    echo "    validate a VRAM overclock: memory errors + no bandwidth regression"
    echo "  1) Set stock baseline    - record reference bandwidth at offset 0"
    echo "  2) Test a memory OC       - apply offset, validate, auto-revert"
    echo "  3) Soak test             - repeat the test for N minutes"
    echo "  4) Validate current      - read-only check at current offsets"
    echo "  5) Pick a different GPU"
    echo "  6) Quit"
    read -rp "Choice: " c
    case "$c" in
      1) run_and_verdict "$GPU" 0.85 3 baseline ;;
      2) read -rp "  memory offset MHz (e.g. 1000): " mem
         read -rp "  core offset MHz [0]: " core; core="${core:-0}"
         read -rp "  memtest passes [3]: " iters; iters="${iters:-3}"
         apply_validate "$GPU" "$core" "$mem" "$iters" 0 ;;
      3) read -rp "  memory offset MHz: " mem
         read -rp "  core offset MHz [0]: " core; core="${core:-0}"
         read -rp "  soak minutes [10]: " soak; soak="${soak:-10}"
         apply_validate "$GPU" "$core" "$mem" 3 "$soak" ;;
      4) run_and_verdict "$GPU" 0.85 3 test ;;
      5) return ;;
      6) exit 0 ;;
      *) echo "invalid." ;;
    esac
  done
}

compile_probe || exit 2
while :; do pick_gpu; action_menu; done
