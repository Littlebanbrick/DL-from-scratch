#!/usr/bin/env bash
# 学习率扫描。固定步数、关掉早停，只比较同一时刻的 dev NLL。
set -u
cd "$(dirname "$0")"

PY=../../.venv/bin/python
STEPS=2000
PATIENCE=999
LRS="1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1"
LOG=sweep_lr.log

printf "%-8s %-10s %-10s %-10s\n" lr dev test last_train | tee "$LOG"

for lr in $LRS; do
    out=$($PY split.py --lr "$lr" --max-steps "$STEPS" --patience "$PATIENCE" \
                       --ckpt "/tmp/sweep_${lr}.pt" 2>&1)
    dev=$(echo "$out" | grep "mlp (3-char" | awk '{print $4}')
    test=$(echo "$out" | grep "mlp (3-char" | awk '{print $5}')
    train=$(echo "$out" | grep -E "^step" | tail -1 | awk '{print $4}')
    printf "%-8s %-10s %-10s %-10s\n" "$lr" "${dev:-NA}" "${test:-NA}" "${train:-NA}" | tee -a "$LOG"
done

rm -f /tmp/sweep_*.pt
echo
echo "View results at $LOG"
