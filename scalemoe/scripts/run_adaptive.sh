rm /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/layer.py
rm /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/sharded_moe.py
rm /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/experts.py
cp ../moe/layer_tutel.py /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/layer.py
cp ../moe/sharded_moe_adaptive.py /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/sharded_moe.py
cp ../moe/experts_tutel.py /opt/conda/envs/deepspeed_env/lib/python3.11/site-packages/deepspeed/moe/experts.py
