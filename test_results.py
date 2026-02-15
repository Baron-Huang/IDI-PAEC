import numpy as np
import os

if __name__ == '__main__':

    import joblib
    read_feat_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Results/Features/DHMC_Kidney/IGI_PAEC'

    feat_1d = joblib.load(os.path.join(read_feat_path, 'feats_1d.joblib'))
    feat_2d = joblib.load(os.path.join(read_feat_path, 'feats_2d.joblib'))
    feat_3d = joblib.load(os.path.join(read_feat_path, 'feats_3d.joblib'))
    feat_full = joblib.load(os.path.join(read_feat_path, 'feats_full.joblib'))

    print(len(feat_1d), feat_1d[0].shape)
    print(len(feat_2d), feat_2d[0].shape)
    print(len(feat_3d), feat_3d[0].shape)
    print(len(feat_full), feat_full[0].shape)
    print('-------------------------------------------------')

    layers = np.load('/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Results/Layers/DHMC_Kidney/IGI_PAEC/layers.npy')
    print(layers.shape)
    print('-------------------------------------------------')

    read_relat_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Results/Relations/DHMC_Kidney/IGI_PAEC'

    relations = joblib.load(os.path.join(read_relat_path, 'relations.joblib'))
    labels = joblib.load(os.path.join(read_relat_path, 'labels.joblib'))
    distances = joblib.load(os.path.join(read_relat_path, 'distances.joblib'))

    print(len(relations), relations[0])
    print(len(labels), labels[0])
    print(len(distances), distances[0])