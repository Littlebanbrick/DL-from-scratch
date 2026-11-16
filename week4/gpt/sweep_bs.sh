for bs in 8 32 64 128; do
  python train.py --block-size $bs --steps 2000 --eval-every 100 --ckpt /tmp/ck_$bs.pt
done