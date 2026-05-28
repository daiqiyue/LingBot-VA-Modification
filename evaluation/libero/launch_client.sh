START=0
END=1

PORT="${PORT:-$((29056 + (${SLURM_JOB_ID:-0} % 1000)))}"

CAMERA_ARGS=${CAMERA_ARGS:-}
PYTHONPATH=. python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port "${PORT}" \
    --test-num 10 \
    --task-range $START $END \
    --out-dir outputs/libero/task0_camera_init_pos_0.2 \
    --eef-delta 0.00 0.20 0.00 \
    #--eef-delta 0.00 0.30 0.00 \




# Text_distractor_2:
#  put both the alphabet soup and the tomato sauce in the basket. the cream cheese, ketchup, orange juice, milk, and butter are also on the table.
