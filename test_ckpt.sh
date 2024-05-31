incrcp=$1
diff=$2
naive_incre=$3


ckpt_dir="/mnt/ssd/deepfm" # directory to save checkpoints
dataset_path="/mnt/ssd/dataset/kaggle/train_sample.txt" # sampled kaggle dataset path
check_freq=10 # checkpoint frequency: number of iterations
num_batches=1500  # numebr of total training iterations
result_path=/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/deepfm # output path

if [ $incrcp = 1 ]; then
  mkdir -p $ckpt_dir/incrcp
  echo "start testing incrcp..."
  python run_deepfm_mp.py --num-batches=$num_batches \
    --ckpt-method="incrcp" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --eperc=0 \
    --concat=1 \
    --incrcp-reset-thres=80 \
    --perf-out-path="$result_path/incrcp.json"

fi

if [ $diff = 1 ]; then
  mkdir -p $ckpt_dir/diff
  echo "start testing diff..."
  python run_deepfm_mp.py --num-batches=$num_batches \
    --ckpt-method="diff" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --perf-out-path="$result_path/diff.json"

fi

if [ $naive_incre = 1 ]; then
  mkdir -p $ckpt_dir/naive_incre
  echo "start testing naive_incre..."
  python run_deepfm_mp.py --num-batches=$num_batches \
    --ckpt-method="naive_incre" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --perf-out-path="$result_path/naive_incre.json"

fi
