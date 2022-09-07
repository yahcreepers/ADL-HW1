python train_intent.py --batch_size="${1}"
python test_intent.py --test_file "data/intent/eval.json" --ckpt_path ckpt/intent/best.pt --pred_file "pred.intent.csv" --batch_size="${1}"
