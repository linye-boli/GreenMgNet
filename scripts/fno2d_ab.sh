for ab in ca cb cc cd ce
do
    python fno/fourier_2d_ab.py --dataset_nm 'darcy' --ab_cfg $ab --trasub 3 --testsub 3 --seed $2 --batch_size 16 --ntest 100 --device $1
done 

for ab in ma mb mc md me
do
    python fno/fourier_2d_ab.py --dataset_nm 'darcy' --ab_cfg $ab --trasub 3 --testsub 3 --seed $2 --batch_size 16 --ntest 100 --device $1
done 
