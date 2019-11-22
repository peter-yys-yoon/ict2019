import os
import platform
import argparse
import time
PUMA = 'puma'
OBAMA = 'obama'

PATH_DATASET = '/home/peter/dataset'
if platform.node() == PUMA:  #
    PATH_DATASET = '/home/peter/dataset'
    PATH_CONDA = '/home/peter/.conda/envs'

else:
    PATH_DATASET = '/home/peter/extra/dataset'
    PATH_CONDA = '/home/peter/anaconda3/envs'


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--db', help='dataset directory name', action='store', required=True)
    parser.add_argument('--tag', help='model name', action='store', required=True)
    parser.add_argument('--env', help='conda env', action='store', required=True)
    parser.add_argument('--py', help='python file', action='store', required=True)

    return parser.parse_args()


if __name__ == "__main__":

    # opt = get_parser()

    ###################
    DB_NAME = 'gist'
    DB_SUBDIR='org'
    TAG = 'alpha'
    ENV = 'alpha.torch'
    PY = '/home/peter/workspace/AlphaPose/video_demo.py'
    ## other arguments here !!!

    ##################

    VIDEOPATH = f'{PATH_DATASET}/{DB_NAME}/{DB_SUBDIR}'
    PATH_OUTPUT_ROOT = f'{PATH_DATASET}/{DB_NAME}/{DB_SUBDIR}_{TAG}'
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'

    OUTPATH = os.path.join(VIDEOPATH, 'output')

    DATA = f'{DB_SUBDIR}/'  # delimeter

    if not os.path.exists(PATH_OUTPUT_ROOT):
        os.mkdir(PATH_OUTPUT_ROOT)

    for r, d, f in os.walk(VIDEOPATH):
        # print(r,d,f)
        for dd in d:
            subdir = r.split(DATA)
            if len(subdir) == 1:
                tmp_path = os.path.join(PATH_OUTPUT_ROOT, dd)
            else:
                tmp_path = os.path.join(PATH_OUTPUT_ROOT, r.split(DATA)[1], dd)


            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)

        for ff in f:
            if not 'fight2' in ff:
                continue
            in_video = os.path.join(r, ff)
            out_dir = os.path.join(PATH_OUTPUT_ROOT, r.split(DATA)[1])

            if ff.split('.')[1] in ['avi', 'mp4', 'asf', 'webm', 'swp']:
                print(f'{in_video} >> {out_dir}')
                ckpt = time.time()
                COMMAND = f'{EXE} {PY} --video {in_video} --outdir {out_dir} --save_video'
                os.system(COMMAND)
                print(f'Process took {time.time()-ckpt} sec')

                #print(COMMAND)
