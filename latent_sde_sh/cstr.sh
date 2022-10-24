cd ..
CUDA_VISIBLE_DEVICES=3 python latent_sde.py --train-dir sde_save/cstr --batch-size 512 --show-prior False --data cstr --adjoint False --dt 5e-2 --adjoint False
