

import os
import subprocess
path = '/home/lunar/DBA-Fusion/datasets/tumvi/512_16_euroc/dataset-room1_512_16'
for i in ['room1']:
    p = subprocess.Popen("python demo_vio_tumvi.py" +\
    " --imagedir=%s/mav0/cam0/data"%path +\
    " --imagestamp=%s/mav0/cam0/data.csv"%path +\
    " --imupath=%s/mav0/imu0/data.csv"%path +\
    " --gtpath=%s/dso/gt_imu.csv"%path +\
    " --enable_h5" +\
    " --h5path=/home/lunar/DBA-Fusion/datasets/tumvi/512_16_euroc/%s.h5" % i +\
    " --resultpath=results/result_%s.txt"%i +\
    " --calib=calib/tumvi.txt" +\
    " --stride=4" +\
    " --max_factors=48" +\
    " --active_window=12" +\
    " --frontend_window=5" +\
    " --frontend_radius=2" +\
    " --frontend_nms=1" +\
    " --far_threshold=0.02" +\
    " --inac_range=3" +\
    " --visual_only=0" +\
    " --translation_threshold=0.2" +\
    " --mask_threshold=-1.0" +\
    " --skip_edge=[-4,-5,-6]" +\
    " --save_pkl" +\
    " --pklpath=results/%s.pkl"%i +\
    " --show_plot",shell=True)
    p.wait()
    
    print("Finished processing dataset: %s" % i)
