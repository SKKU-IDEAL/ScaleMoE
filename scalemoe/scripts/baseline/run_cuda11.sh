mv /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe_org
cp -r /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe_org /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe 
export PATH="/opt/conda/envs/deepspeed_env/bin:$PATH"
git clone https://github.com/microsoft/tutel.git
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py install
rm /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe/layer.py
rm /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe/sharded_moe.py
rm /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe/experts.py
cp ../../moe/layer_tutel.py /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe/layer.py
cp ../../moe/sharded_moe_tutel.py /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe/sharded_moe.py
cp ../../moe/experts_tutel.py /opt/conda/envs/deepspeed_env/lib/python3.10/site-packages/deepspeed/moe/experts.py
