touch ~/.hdfscli.cfg
echo "[dev.alias]" > ~/.hdfscli.cfg
echo "url = http://10.151.40.179:50070" >> ~/.hdfscli.cfg
hdfscli download --alias=dev /v_yugzh/imagenet /home/v-yugzh
python3 main.py /home/v-yugzh/imagenet -j 4 --epochs 20 -a efficientnet-b0 --pretrained --num-classes 1000 --batch-size 64 --optimizer rmsprop
