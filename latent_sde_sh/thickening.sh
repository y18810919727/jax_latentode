cd ..
CUDA_VISIBLE_DEVICES=2 python latent_sde.py --train-dir sde_save/thickener --batch-size 512 --show-prior False --data thickener --adjoint False --dt 5e-2 --adjoint False
