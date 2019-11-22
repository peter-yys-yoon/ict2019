import os

OBAMA = 'obama'
EXE = '/home/peter/anaconda3/envs/py36/bin/python'  
PY = 'video_demo.py'
OUTDIR = '/home/peter/extra/dataset/gist/demo2019/ver1'
datapath ='/home/peter/extra/dataset/gist/demo2019/trim'
for ff in os.listdir(datapath):
    # if ff in ['trim_fight_jh_s1_student4.mp4', 'trim_fight_jh_s2_student4.mp4']:
    #     continue
    # if os.path.exists(os.path.join(OUTDIR, f'AlphaPose_{ff.split('.')[0]}_results.json')):
    #     continue

    if ff != 'trim_fight_jh_s1_student4.mp4':
        continue
    invideo = os.path.join(datapath, ff)
    print(invideo, os.path.exists(invideo))
    COMMAND = f'{EXE} {PY} --video {invideo} --outdir {OUTDIR} --save_video --sp'
    os.system(COMMAND)

