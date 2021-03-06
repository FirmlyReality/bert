CUDA_VISIBLE_DEVICES=1
BERT_BASE_DIR=../chinese_L-12_H-768_A-12
MY_DATASET=../../filter_data1

CUDA_VISIBLE_DEVICES=1 python3 -u run_classifier.py --task_name=reply --do_eval=false --do_mypredict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=../best_save/model.ckpt-700 \
	--iterations_per_loop=1000 ---max_seq_length=256 --predict_batch_size=16 --learning_rate=1e-5 --output_dir=../rp_predict_output/
