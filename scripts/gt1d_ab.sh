for ab in ca cb cc cd ce
do
    python gt/gt_1d_ab.py --dataset_nm 'burgers' --ab_cfg $ab --trasub 4 --testsub 4 --seed $2 --batch_size 16 --ntest 200 --device $1
done 
