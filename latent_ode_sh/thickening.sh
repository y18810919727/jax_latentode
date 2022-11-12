cd ..
CUDA_VISIBLE_DEVICES=2 python main.py --train-dir ode_save/thickener --batch-size 1024 --data thickener --sp=0.25 --evenly=False
