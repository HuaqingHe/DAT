scp -P 30657 -r root@10.255.0.125:/simple/HuaqingHe/.cache/.cache /root
scp -P 30859 -r root@10.255.0.125:/simple/HuaqingHe/conda/envs/Real-ESRGAN /opt/conda/envs


python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_S_x4.yml --launcher pytorch