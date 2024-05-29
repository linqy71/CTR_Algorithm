incrcp=$1
diff=$2
naive_incre=$3


ckpt_dir="/mnt/3dx/pnn"

dataset_path="/mnt/ssd/dataset/kaggle/train_sample.txt"

check_freq=10
num_batches=1000

if [ $incrcp = 1 ]; then
  mkdir -p $ckpt_dir/incrcp
  echo "start testing incrcp..."
  python run_pnn_mp.py --num-batches=$num_batches \
    --ckpt-method="incrcp" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --eperc=0.02 \
    --concat=1 \
    --incrcp-reset-thres=80 \
    --perf-out-path="/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/pnn/incrcp.3dx.json"

fi

if [ $diff = 1 ]; then
  mkdir -p $ckpt_dir/diff
  echo "start testing diff..."
  python run_pnn_mp.py --num-batches=$num_batches \
    --ckpt-method="diff" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --perf-out-path="/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/pnn/diff.3dx.json"

fi

if [ $naive_incre = 1 ]; then
  mkdir -p $ckpt_dir/naive_incre
  echo "start testing naive_incre..."
  python run_pnn_mp.py --num-batches=$num_batches \
    --ckpt-method="naive_incre" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --perf-out-path="/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/pnn/naive_incre.3dx.json"

fi