data_path=../Data/res15/
save_path=./savemodel/res15/

SEED=1000

for Z in 0.1 0.3 0.5 0.7 1
do 
CUDA_VISIBLE_DEVICES=1 python3 -u Train.py \
	--seed $SEED \
	--data_path $data_path \
	--model_dir $save_path \
	--span_pruned_threshold $Z \
	--epochs 60 ;
done;

