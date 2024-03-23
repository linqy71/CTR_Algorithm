lsecp=1
diff=1
incre=1
rocksdb=1

ckpt_dir="/mnt/ssd/pnn"

dataset_path="/mnt/ssd/dataset/kaggle/train_sample.txt"

check_freq=10
num_batches=1000

if [ $lsecp=1 ]; then
  mkdir -p $ckpt_dir/lsecp

  python run_pnn_mp.py --num-batches=$num_batches \
    --ckpt-method="lsecp" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --lsecp-eperc=0.01 \
    --lsecp-clen=10 \
    --perf-out-path="./lsecp.json"

fi

if [ $diff=1 ]; then
  mkdir -p $ckpt_dir/diff

  python run_pnn_mp.py --num-batches=$num_batches \
    --ckpt-method="diff" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --lsecp-eperc=0.01 \
    --lsecp-clen=10 \
    --perf-out-path="./diff.json"

fi

if [ $incre=1 ]; then
  mkdir -p $ckpt_dir/incre

  python run_pnn_mp.py --num-batches=$num_batches \
    --ckpt-method="incre" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --lsecp-eperc=0.01 \
    --lsecp-clen=10 \
    --perf-out-path="./incre.json"

fi

if [ $rocksdb=1 ]; then
  mkdir -p $ckpt_dir/rocksdb

  python run_pnn_mp.py --num-batches=$num_batches \
    --ckpt-method="rocksdb" \
    --dataset-path=$dataset_path \
    --ckpt-freq=$check_freq \
    --ckpt-dir=$ckpt_dir \
    --lsecp-eperc=0.01 \
    --lsecp-clen=10 \
    --perf-out-path="./rocksdb.json"

fi