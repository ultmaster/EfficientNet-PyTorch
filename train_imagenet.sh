apt update
apt install --assume-yes nfs-common cifs-utils sshpass wget git
mkdir --parents /mnt/data
mount -t cifs //10.151.40.4/data/yugzh /mnt/data -o vers=3.0,username=paismbuser,password=paismbpwd,domain=WORKGROUP

# (tensorboard --logdir /tmp/summary --port $PAI_PORT_LIST_main_0_tensorboard &)
python3 main.py /mnt/data/imagenet -j 16 --epochs 5 -a efficientnet --num-classes 1000 --batch-size 64 --optimizer rmsprop --lr 0.064 --wd 1e-5 --request-from-nni
