import numpy as np
import numpy.linalg as la
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation as R
import argparse

def main(args):
    #Define dataset patbaseTflange
    dir = Path(args.dir)
    df = pd.read_csv(str(dir/'poses'/'linkFlangePosition.csv'))
    df1 = pd.read_csv(str(dir/'poses'/'linkcamera_poses.csv'))
    flangeTcam = la.inv(np.array([[-0.999934889413633,   0.011124389433672,  -0.002542615400269,0.008864516956889],
                            [-0.002854035919604,  -0.028068672932281,   0.999601922806669,-0.061164000497153],
                            [0.011048593227886,   0.999544094855024 ,  0.028098594771555,-0.017462100676280],
                            [0,0,0,1]]))
    # camTflange = (np.array([[-0.999906388491091,  0.003430762112551,  -0.013245532304538,  0.008395972230855],
    #                     [0.013335504190397,  0.027684412985082,   -0.999527757296244,   0.015366334011880],
    #                     [-0.003062447173649,  -0.999610825846238,   -0.027727572347938,   0.061589322918292],
    #                     [0,                   0,                   0,   1.000000000000000]]))
    # flangeTcam = la.inv(camTflange)
    # flangeTcam = (np.array([[-0.999906388491091,  0.013335504190397, -0.003062447173649,  0.008378882507517],
    #                         [0.003430762112551, 0.027684412985082,  -0.999610825846238, -0.061111141425385],
    #                         [-0.013245532304538,  -0.999527757296244, - 0.027727572347938, -0.017178008901248],
    #                         [0,             0,                  0,  1.000000000000000]]))
    eye = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
    
    for idx in range(df.shape[0]):
        r = R.from_quat(np.array(df.loc[idx, 'Q_x':'Q_w'])) 
        t = np.array(df.loc[idx, 'X':'Z'])
        baseTflange = np.eye(4)
        baseTflange[:3,:3] = r.as_matrix()
        baseTflange[:3,-1]= t
        # camTbase = camTflange@la.inv(baseTflange)
        baseTcam = baseTflange@flangeTcam
        quat = R.from_matrix(baseTcam[:3,:3]).as_quat()
        df.loc[idx,'Q_x':'Q_w'] = quat
        df.loc[idx,'X':'Z'] = baseTcam[:3,-1]
        pass
    df.to_csv(str(dir/'poses'/'HE_poses.csv'))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        nargs='?',
                        const='/home/alex/data/dataset12/r200/',
                        type=str,
                        default='/home/alex/data/dataset12/r200/')
    SystemExit(main(parser.parse_args()))

