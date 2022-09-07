mkdir cache
cd cache
mkdir intent
cd intent
wget https://www.dropbox.com/s/vafuvm74y0r6ld1/embeddings.pt?dl=1 -O embeddings.pt
wget https://www.dropbox.com/s/mgrpzsrt35fff96/intent2idx.json?dl=1 -O intent2idx.json
wget https://www.dropbox.com/s/r7yz04ydnzm4fqw/vocab.pkl?dl=1 -O vocab.pkl
cd ..
mkdir slot
cd slot
wget https://www.dropbox.com/s/2g9uwekjvg6bqkh/embeddings.pt?dl=1 -O embeddings.pt
wget https://www.dropbox.com/s/c6wqsewmpb5tbjs/tag2idx.json?dl=1 -O tag2idx.json
wget https://www.dropbox.com/s/e45hxqwn0bfzr32/vocab.pkl?dl=1 -O vocab.pkl
cd ..
cd ..
mkdir ckpt
cd ckpt
mkdir intent
cd intent
wget https://www.dropbox.com/s/30ii64vnfy9qxpi/best.pt?dl=1 -O best.pt
cd ..
mkdir slot
cd slot
wget https://www.dropbox.com/s/fzwbtu92bxs31hq/best.pt?dl=1 -O best.pt
