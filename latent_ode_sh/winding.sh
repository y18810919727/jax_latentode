cd ..
CUDA_VISIBLE_DEVICES=2 python main.py --train-dir ode_save/winding --batch-size 1024 --data winding --sp=0.25 --evenly=False
