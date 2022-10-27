CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 VAM/train.py --train-list-file=data/train_data_list.txt --image-folder=/disk6/shiyao.wsy/www/images --batch-size=192 
