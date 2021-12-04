# run federated learning datat minist with CNN
python src/federated_main.py --model=cnn --dataset=mnist  \
            --local_ep=5 --num_users=100  \
            --frac=0.1 --local_bs=10  \
            --epochs=1500 \
            --iid=0\