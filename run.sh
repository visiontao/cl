python main.py --model ser --dataset cifar10 --n_classes 10 --n_tasks 5 --buffer_size 200 --lr 0.03 --n_epochs 20 --alpha 0.2 --beta 0.2 --device_id 0

python main.py --model ser --dataset cifar100 --n_classes 100 --n_tasks 20 --buffer_size 200 --lr 0.03 --n_epochs 50 --alpha 0.5 --beta 0.5 --device_id 0

python main.py --model ser --dataset tinyimg --n_classes 200 --n_tasks 10 --buffer_size 200  --lr 0.03 --n_epochs 100 --alpha 0.2 --beta 1.0 --device_id 0

python main.py --model ser --dataset perm-mnist --n_classes 10 --n_tasks 20 --buffer_size 200 --lr 0.1 --n_epochs 1 --alpha 0.2 --beta 0.2 --batch_size 128 --device_id 0

python main.py --model ser --dataset rot-mnist --n_classes 10 --n_tasks 20 --buffer_size 200 --lr 0.1 --n_epochs 1 --alpha 0.2 --beta 0.2 --batch_size 128 --device_id 0