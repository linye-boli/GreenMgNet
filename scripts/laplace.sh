# 513
# baseline fno
python fourier_1d.py --device 0 --task laplace --model fno --seed 0

# baseline toep gl 
python green_mgnet.py --device 1 --task laplace --act relu --seed 0 --mode gl --lr_adam 1e-3 --ep_adam 5000 --ep_lbfgs 0 --sch --thr 0

# toep green mg
python green_mgnet.py --device 2 --task laplace --k 5 --m 3 --seed 0 --mode dd_mg --lr_adam 1e-3 --ep_adam 1000 --ep_lbfgs 0 --seed 1 --sch --thr 1.0 --w log

# 8193
