import os
import platform
import argparse

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
    TAG = ' alpha'
    ENV = 'alpha'
    PY = '/home/peter/extra/Workspace/codes/tiah_module/standalone/video_resize.py'

    ## other arguments here !!!

    ##################

    VIDEOPATH = f'{PATH_DATASET}/{DB_NAME}'
    PATH_OUTPUT_ROOT = f'{PATH_DATASET}/{TAG}_{DB_NAME}'
    EXE = f'{PATH_CONDA}/{ENV}/bin/python'

    OUTPATH = os.path.join(VIDEOPATH, 'output')

    DATA = f'{DB_NAME}/'  # delimeter

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
            in_video = os.path.join(r, ff)
            out_dir = os.path.join(PATH_OUTPUT_ROOT, r.split(DATA)[1])

            if ff.split('.')[1] in ['avi', 'mp4', 'asf', 'webm', 'swp']:
                print(f'{in_video} >> {out_dir}')
                COMMAND = f'{EXE} {PY} --video {in_video} --size vga --outdir {out_dir} --fixed '
                # os.system(COMMAND)
