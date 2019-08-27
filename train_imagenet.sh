set -x
apt update
apt install --assume-yes nfs-common cifs-utils sshpass wget git
mkdir --parents /mnt/data
mount -t cifs //10.151.41.13/data/yugzh /mnt/data -o vers=3.0,username=paismbuser,password=paismbpwd,domain=WORKGROUP
git clone https://github.com/ultmaster/EfficientNet-PyTorch
cd EfficientNet-PyTorch
cp examples/imagenet/main.py .
python3 main.py /mnt/data/imagenet -j 1 -a efficientnet --batch-size 32 --lr 0.032 --wd 1e-5 --epochs 5 --request-from-nni
