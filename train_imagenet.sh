apt update
apt install --assume-yes nfs-common cifs-utils sshpass wget git
umask 000
declare -a MOUNTPOINTS=()
mkdir --parents /mnt/data
mount -t cifs //10.151.40.4/data/yugzh /mnt/data -o vers=3.0,username=paismbuser,password=paismbpwd,domain=WORKGROUP
pip install tensorboardx tensorboard tensorflow
pip install -U torch torchvision

git clone https://github.com/ultmaster/EfficientNet-PyTorch
cd EfficientNet-PyTorch

(tensorboard --logdir /tmp/summary --port $PAI_PORT_LIST_main_0_tensorboard &)
python3 main.py /mnt/data/imagenet -j 16 --epochs 20 -a efficientnet-b0 --num-classes 1000 --batch-size 64 --optimizer rmsprop --lr 0.064 --wd 1e-5
