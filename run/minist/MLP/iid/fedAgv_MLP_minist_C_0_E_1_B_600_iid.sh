# run federated learning datat minist with MLP
python src/federated_main.py --model=mlp --dataset=mnist  \
            --local_ep=1 --num_users=100  \
            --frac=0 --local_bs=600  \
            --epochs=1500 \
            --iid=1\