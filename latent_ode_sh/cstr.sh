cd ..
CUDA_VISIBLE_DEVICES=3 python main.py --train-dir ode_save/cstr --batch-size 1024 --data cstr --sp 0.25 --evenly False
