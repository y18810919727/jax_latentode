cd ..
CUDA_VISIBLE_DEVICES=2 python latent_sde.py --train-dir sde_save/cstr --batch-size 2048 --show-prior False --data cstr --adjoint False --dt 5e-2 --adjoint False
