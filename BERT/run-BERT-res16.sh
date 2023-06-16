data_path=../Data/res16/
save_path=./savemodel/res16/

for SEED in {1..100}
do 
CUDA_VISIBLE_DEVICES=1 python3 -u BERT-Train.py \
	--seed $SEED \
	--data_path $data_path \
	--model_dir $save_path \
	--epochs 20 ;
done;

