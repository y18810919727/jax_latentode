cd ..
CUDA_VISIBLE_DEVICES=3 python latent_sde.py --train-dir sde_save/winding --batch-size 2048 --show-prior False --data winding --adjoint False --dt 5e-2 --adjoint False
