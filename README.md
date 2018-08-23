# mts-drowsiness

==Config:==
NVIDIA driver: 384.111
CUDA version: 8.0.61
CUDNN version: 5.1.10
TORCH7 git commit: 5beb83c46e91abd273c192a3fa782b62217072a6

== DATASET ==

* Sequence of eyelids distances: are in data/raw/eld-seq folder, with filenames "SUBJECT-TEST.t7"
* Reaction times: are in data/raw/rt folder, with filename "SUBJECT-TEST.txt"

==TRAIN eyelids distance module==
 th -i main.lua -dataset ieye2eld -max_epochs 50 -batch_size 32 -optimizer rmsprop -layers '32 64 128 256' -flip true -seed 29 -lr 0.001428 -alpha 0.9886

==EVALUATE eyelids distance module==
th -i main.lua -dataset ieye2eld -testOnly true -testModel models/ieye2eld/ieye2eld_0.20510455.net

==TRAIN drowsiness module==
th -i crossval_main.lua -dataset eld2multid -max_epochs 200 -earlystop 20 -progress false -optimizer adam -min_train_error 0.1 -augment balance -n_augment 256 -dropout 0.7 -layers '32 15 32 31 16' -weight_eval fixed -seed 106 -lr 0.0016029

==EVALUATE drowsiness module==
th -i main.lua -dataset eld2multid -testOnly true -testModel models/eld2multid/eld2multid_0.35916242.net
