# 513
# baseline fno
python fourier_1d.py --device 0 --task logarithm --model fno --seed 0

# baseline toep gl 
python green_mgnet.py --device 2 --task logarithm --act relu --seed 0 --mode toep_gl --lr_adam 1e-3 --ep_adam 5000 --ep_lbfgs 0 --sch --thr 0
# 8193
python green_mgnet.py --device 2 --task logarithm --act relu --seed 0 --mode toep_gl --lr_adam 1e-3 --ep_adam 5000 --ep_lbfgs 0 --sch --thr 0 --fine

# toep green mg
python green_mgnet.py --device 1 --task logarithm --k 5 --m 3 --seed 0 --mode toep_mg --lr_adam 1e-3 --ep_adam 1000 --ep_lbfgs 0 --seed 1 --sch --thr 1.0 --w log
python green_mgnet.py --device 1 --task laplace --k 9 --m 3 --seed 0 --mode dd_mg --lr_adam 1e-3 --ep_adam 1000 --ep_lbfgs 0 --seed 1 --sch --thr 1e-5 --w one --fine

