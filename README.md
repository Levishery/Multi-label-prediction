# Multi-label-prediction
Multi-label prediction using inception on CelebA

## prepare data to TF Records

first seperate the train and vali images and texts
then:

python build_image_data_multi.py \
  --train_directory=".../CelebA_train" \
  --validation_directory=".../CelebA_vali" \
  --output_directory=".../CelebA_TF_40" \
  --vali_labels_file=".../vali_label.txt"\
  --train_labels_file=".../train_label.txt"\
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8i
  
 ## train
 
 CUDA_VISIBLE_DEVICES=2 python CelebA_train.py \
  --train_dir=".../CelebA_ckpt" \
  --data_dir=".../CelebA_TF_40"
  
 ## validation
 
 CUDA_VISIBLE_DEVICES=2 python CelebA_eval.py \
  --eval_dir=".../CelebA_eval" \
  --data_dir=".../CelebA_TF_40" \
  --subset=validation \
  --num_examples=64 \
  --checkpoint_dir=".../CelebA_ckpt" \
  --input_queue_memory_factor=1 \
  --run_once
  
  ## results
  
  loss:
![output](https://github.com/Levishery/Multi-label-prediction/blob/master/loss.png)
![output](https://github.com/Levishery/Multi-label-prediction/blob/master/prediction.png)
