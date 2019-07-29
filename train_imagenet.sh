set -x
apt update
apt install wget
pip install hdfs
pip3 install -U torch torchvision
touch ~/.hdfscli.cfg
echo "[dev.alias]" > ~/.hdfscli.cfg
echo "url = http://10.151.40.179:50070" >> ~/.hdfscli.cfg
hdfscli download --alias=dev /v_yugzh/imagenet /root
python3 main.py /root/imagenet -j 4 --epochs 20 -a efficientnet-b0 --pretrained --num-classes 1000 --batch-size 64 --optimizer rmsprop