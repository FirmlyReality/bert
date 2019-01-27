BERT_BASE_DIR=../chinese_L-12_H-768_A-12
MY_DATASET=../data

python3 run_classifier.py --task_name=guba --do_train=true --do_eval=true --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=256 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=3 --output_dir=../guba_output/
