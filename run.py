import os
import subprocess
import numpy as np
import sys

def run(command):
    subprocess.call(command, shell=True)

def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    parameter, result, time_cost = '', '', ''
    for line in iter(p.stdout.readline, b''):
        line = line.rstrip().decode('utf8')
        if 'args:' in line:
            parameter = line
        if 'final best performance:' in line:
            result = line
        if 'Experiment cost:' in line:
            time_cost = line
    return parameter, result, time_cost

def per_data():

    #(4) joint restaurant
    for  para in [1]:
        path=os.path.join('out/run_joint_res', "ramdom1_task1" )
        cmd = 'python -m absa.run_joint_span ' +' '+ \
              ' --logit_threshold 9.6 ' + ' '+ \
              ' --train_file  split_rest_total_train.txt ' +' '+ \
              ' --predict_file  rest_total_test.txt'  + ' ' + \
              ' --weight_span  1e-7 '  + ' ' + \
              ' --weight_ac  0.05'  + ' ' + \
              ' --output_dir '  + str(path) + ' '+ \
              ' --shared_weight 9 ' + ' ' + \
              ' --layer_GRU 3 ' + ' ' + \
              ' --random_train 1 ' + ' '
        run(cmd)
        sys.stdout.flush()
    # # #
    # # # # #(5)joint laptop
    random_train=[0,0.1,0.2,0.99,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.3,1]
    for para in [1]:
            path = os.path.join('out/run_joint_laptop', "ramdom1_task1" )
            cmd = 'python -m absa.run_joint_span ' +' '+ \
                  ' --logit_threshold  9.8 ' + ' '+ \
                  ' --train_file  split_laptop14_train.txt ' +' '+ \
                  ' --predict_file  laptop14_test.txt ' + ' ' + \
                  ' --weight_span 1e-7 '  +' ' + \
                  ' --weight_ac  0.08 '  + ' ' + \
                  ' --output_dir ' + str(path) + ' '+ \
                  ' --shared_weight 1 ' + ' ' + \
                  ' --layer_GRU 2 ' + ' ' + \
                  ' --random_train 1 ' + ' '
            run(cmd)
            sys.stdout.flush()

    # #(6) joint twitter

    for j in [10]:
        train_path = os.path.join('split_twitter%s_train.txt' % (j))
        test_path = os.path.join('twitter%s_test.txt' % (j))
        path = os.path.join('out/ramdon_twitter', "ramdom1-j-%s" % (j))
        cmd = 'python -m absa.run_joint_span ' + ' ' + \
              ' --logit_threshold 9.6 ' + ' ' + \
              ' --train_file ' + str(train_path) + ' ' + \
              ' --predict_file ' + str(test_path) + ' ' + \
              ' --weight_ac 0.05 ' + ' ' + \
              ' --weight_span 1e-7 ' + ' ' + \
              ' --output_dir ' + str(path) + ' ' + \
              ' --shared_weight 1 ' + ' ' + \
              ' --layer_GRU 2 ' + ' ' + \
              ' --random_train 1 ' + ' '
        run(cmd)
        sys.stdout.flush()
per_data()
