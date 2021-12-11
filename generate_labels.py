import os
import argparse
import os.path as path

BASE_DIR = "raw_data/trdg"


with open(path.join(BASE_DIR,'igbo_target.txt'), mode='w') as f:
    for file in os.listdir('igbo'):
        new_file = file.replace(' ', '_')
        print(new_file)
        os.rename(path.join('igbo', file), path.join('igbo', new_file))
        f.write(new_file + " " + file.replace('.jpg','').replace.split('_')[0]+"\n")
