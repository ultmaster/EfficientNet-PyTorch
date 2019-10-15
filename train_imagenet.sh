set -x
apt update
apt install --assume-yes nfs-common cifs-utils sshpass wget git
mkdir --parents /mnt/data
mount -t cifs //path/to/folder/contains/imagenet -o vers=3.0,username=username,password=password,domain=WORKGROUP  # mount imagenet
git clone https://github.com/ultmaster/EfficientNet-PyTorch
cd EfficientNet-PyTorch
python3 main.py /mnt/data/imagenet -j 1 -a efficientnet --batch-size 32 --lr 0.032 --wd 1e-5 --epochs 5 --request-from-nni
