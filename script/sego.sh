#!/bin/bash

python treecl.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -batch_size_test 128 -num_epoch 400 -num_cluster 2 -alpha 0.0005

python treecl.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -num_epoch 20 -num_cluster 5 -alpha 0.0005

python treecl.py -exp_type oodd -DS_pair PTC_MR+MUTAG -num_epoch 400 -num_cluster 2 -alpha 0.1

python treecl.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -batch_size_test 128 -num_epoch 400 -num_cluster 20 -alpha 0.5

python treecl.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider -batch_size_test 128 -num_epoch 400 -num_cluster 5 -alpha 0.7

python treecl.py -exp_type oodd -DS_pair AIDS+DHFR -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.0002

python treecl.py -exp_type oodd -DS_pair ogbg-molbbbp+ogbg-molbace -batch_size_test 128 -num_epoch 400 -num_cluster 30 -alpha 0.5

python treecl.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 400 -num_cluster 2 -alpha 0.5

python treecl.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo -batch_size_test 128 -num_epoch 300 -num_cluster 30 -alpha 0.001

python treecl.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -num_epoch 150 -num_cluster 15 -alpha 0.5
