cd ..
CUDA_VISIBLE_DEVICES=2 python latent_sde.py --train-dir sde_save/winding --batch-size 512 --show-prior False --data winding --adjoint False --dt 5e-2 --adjoint False
