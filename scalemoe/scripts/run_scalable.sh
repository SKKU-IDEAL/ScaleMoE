rm /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/layer.py
rm /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/sharded_moe.py
rm /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/experts.py
cp ../moe/layer.py /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/layer.py
cp ../moe/sharded_moe.py /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/sharded_moe.py
cp ../moe/experts.py /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/experts.py
