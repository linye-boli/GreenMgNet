for ab in ca cb cc cd ce
do
    python gt/gt_2d_ab.py --dataset_nm 'darcy' --ab_cfg $ab --trasub 3 --testsub 3 --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
done 

for ab in ma mb mc md me
do
    python gt/gt_2d_ab.py --dataset_nm 'darcy' --ab_cfg $ab --trasub 3 --testsub 3 --seed $2 --batch_size 2 --ntest 100 --epochs 200 --lr 0.00025 --device $1
done 
