# Before running the DCC parts, ensure you have made testdata.mat and traindata.mat input:
#python preprocessing/make_dataset.py

# First, we make the input graph stored in pretrained.mat:
python pytorch/edgeConstruction.py --dataset child_poet --prep none --samples 50012 --k 10

# Then we pretrain the Autoencoder, and merge the graph and autoencoder, before training DCC
python pytorch/pretraining.py --data child_poet --tensorboard --id 3 --niter 50000 --lr 10 --step 20000 --dtype mat
python pytorch/extract_feature.py --data child_poet --net checkpoint_4.pth.tar --features pretrained --dtype mat
python pytorch/copyGraph.py --data child_poet --graph pretrained.mat --features pretrained.pkl --out pretrained --dtype mat
python pytorch/DCC.py --data child_poet --net checkpoint_4.pth.tar --tensorboard --id 3 --dtype mat

# Use this line to inspect the data:
# tensorboard --logdir data/child_poet/results/runs/DCC/2/ --samples_per_plugin images=100