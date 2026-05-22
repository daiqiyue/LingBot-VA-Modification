START=0
END=1


CAMERA_ARGS=${CAMERA_ARGS:-}
PYTHONPATH=. python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port 29056 \
    --test-num 10 \
    --task-range $START $END \
    --out-dir outputs/libero/task0_y_0.3 \
    --prompt "put both the alphabet soup and the tomato sauce in the basket." \
    --agentview-camera-rotate-deg 45 \




# Text_distractor_2:
#  put both the alphabet soup and the tomato sauce in the basket. the cream cheese, ketchup, orange juice, milk, and butter are also on the table.
