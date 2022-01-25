# Before running, ensure you have made testdata.mat and traindata.mat input
# First, we make the input graph stored in pretrained.mat:
python pytorch/edgeConstruction.py --dataset child_poet --prep none --samples 50012

# Then we pretrain the Autoencoder, and merge the graph and autoencoder, before training DCC
python pytorch/pretraining.py --data child_poet --tensorboard --id 1 --niter 50000 --lr 10 --step 20000 --dtype mat
python pytorch/extract_feature.py --data child_poet --net checkpoint_4.pth.tar --features pretrained --dtype mat
python pytorch/copyGraph.py --data child_poet --graph pretrained.mat --features pretrained.pkl --out pretrained --dtype mat
python pytorch/DCC.py --data child_poet --net checkpoint_4.pth.tar --tensorboard --id 2 --dtype mat