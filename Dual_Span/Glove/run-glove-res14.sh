data_path=../Data/res14/
save_path=./savemodel/res14/

for SEED in {1..100}
do 
CUDA_VISIBLE_DEVICES=1 python3 -u Train.py \
	--seed $SEED \
	--data_path $data_path \
	--model_dir $save_path \
	--epochs 60 ;
done;

