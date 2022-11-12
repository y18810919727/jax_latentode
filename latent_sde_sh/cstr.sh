cd ..
<<<<<<< Updated upstream
CUDA_VISIBLE_DEVICES=2 python latent_sde.py --train-dir sde_save/cstr --batch-size 2048 --show-prior False --data cstr --adjoint False --dt 5e-2 --adjoint False
=======
CUDA_VISIBLE_DEVICES=3 python latent_sde.py --train-dir sde_save/cstr --batch-size 1024 --show-prior False --data cstr --adjoint False --dt 5e-2 --adjoint False --sp=0.25 --evenly=False
>>>>>>> Stashed changes
