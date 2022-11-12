cd ..
<<<<<<< Updated upstream
CUDA_VISIBLE_DEVICES=3 python latent_sde.py --train-dir sde_save/winding --batch-size 2048 --show-prior False --data winding --adjoint False --dt 5e-2 --adjoint False
=======
CUDA_VISIBLE_DEVICES=2 python latent_sde.py --train-dir sde_save/winding --batch-size 1024 --show-prior False --data winding --adjoint False --dt 5e-2 --adjoint False --sp=0.25 --evenly=False
>>>>>>> Stashed changes
