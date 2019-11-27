from opt import opt
args = opt
# assert args.fight or args.gta or args.park , "select flag"

# if args.fight:
#     from WriterFight import DataWriter
# elif args.gta:
#     from WriterGTA import DataWriter
# else:
#     from WriterPark import DataWriter

import ntpath
import os
import sys

import cv2
from tqdm import tqdm

from SPPE.src.main_fast_inference import *
from dataloader import VideoLoader, DetectionLoader, DetectionProcessor, Mscoco, DataWriter
from fn import getTime

from pPose_nms import write_json


args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    videofile = args.video


    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if not len(videofile):
        raise IOError('Error: must contain --video')

    if videofile == '1':
        videofile = '/home/peter/dataset/gist/org/mid2019/gta_jh_trial_1/trim_student1.mp4'
    elif videofile == '2':
        videofile = '/home/peter/dataset/gist/org/mid2019/gta_jh_trial_2/trim_ohryong1.mp4'
    elif videofile == '3':
        videofile = '/home/peter/dataset/gist/org/mid2019/roaming_kdh_trial_1/trim_student1.mp4'
    elif videofile == '4':
        videofile ='/home/peter/dataset/gist/org/mid2018/nexpa_fight2.mp4'
    elif videofile == '5':
        videofile ='/home/peter/dataset/gist/org/mid2018/nexpa_vehicle_accident.mp4'
    elif videofile == '6': #PARKING
        videofile ='/home/peter/extra/dataset/gist/demo2019/trim/trim_park_kdh_s2_student1.mp4'
    elif videofile == '7': # FIGHT1 
        videofile ='/home/peter/extra/dataset/gist/demo2019/trim/trim_fight_jh_s1_student4.mp4'
    elif videofile == '8': # FIGHT2 
        videofile ='/home/peter/extra/dataset/gist/demo2019/trim/trim_fight_jh_s2_student4.mp4'
    elif videofile == '9': # GTA1
        videofile ='/home/peter/extra/dataset/gist/demo2019/trim/trim_gta_dh_s1_ohryong1.mp4'
    elif videofile == '10': # GTA2
        videofile ='/home/peter/extra/dataset/gist/demo2019/trim/trim_gta_dy_s1_student1.mp4'
    elif videofile == '11': # GTA3
        videofile ='/home/peter/extra/dataset/gist/demo2019/trim/trim_gta_jh_s1_ohryong1.mp4'
    elif videofile == '12': # GTA4
        videofile ='/home/peter/extra/dataset/gist/demo2019/trim/trim_gta_jh_s2_student1.mp4'
        
    print('Processing ' , videofile)
        

    # Load input video
    data_loader = VideoLoader(videofile, batchSize=args.detbatch).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)

    print('InferNet')
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_' + ntpath.basename(videofile).split('.')[0] + '.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    im_names_desc = tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2, CAR) = det_processor.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name, CAR)
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name, CAR)

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            im_names_desc.set_description(
                'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']),
                    pn=np.mean(runtime_profile['pn']))
            )

    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print(
            '===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while (writer.running()):
        pass
    writer.stop()
    writer.write_enery()
    final_result = writer.results()
    write_json(videofile, final_result, args.outputpath)
