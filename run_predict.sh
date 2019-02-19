CUDA_VISIBLE_DEVICES=0
BERT_BASE_DIR=../chinese_L-12_H-768_A-12
MY_DATASET=../../filter_data1

CUDA_VISIBLE_DEVICES=0 python3 -u run_classifier.py --task_name=guba --do_eval=false --do_mypredict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=../best_save/model.ckpt-1740 \
	--iterations_per_loop=1000 ---max_seq_length=256 --predict_batch_size=16 --learning_rate=1e-5 --num_train_epochs=5 --save_checkpoints_steps=20 --output_dir=../predict_output/
