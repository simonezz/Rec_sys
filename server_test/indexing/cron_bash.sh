#!/bin/bash --login
source /home/master/.bashrc
export PATH=/home/master/anaconda3/envs/py37/bin/python3.7:${PATH}
export PYTHONPATH=/home/master/anaconda3/envs/py37/lib/python3.7/site-packages:${PYTHONPATH}
export PYTHONPATH=/home/master/source/project/Rec_sys/utils:${PYTHONPATH}
/home/master/anaconda3/envs/py37/bin/python3.7 /home/master/source/project/Rec_sys/server_test/indexing/put_server_mathText_img.py --d1 $(date --date="yesterday" +%Y%m%d) 
