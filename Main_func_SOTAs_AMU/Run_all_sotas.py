import subprocess
import os

# 列表中包含你想要顺序执行的脚本名
read_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Codes/Main_func_SOTAs_AMU'
model_names = ['AB_MIL_Gated', 'AB_MIL_Linear', 'CLAM_MB_B', 'CLAM_MB_S',
               'CLAM_SB_B', 'CLAM_SB_S', 'DGR_MIL', 'DTFD_MIL',
               'FRMIL', 'HAG_MIL', 'ILRA_MIL', 'RRTMIL',
               'S4MIL', 'TransMIL']



scripts = []
#scripts.append(r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Codes/IGI_PAEC_main_DHMC_Kidney.py')
for name_i in model_names:
    scripts.append(os.path.join(read_path, name_i + '.py'))



# 遍历列表，依次运行每个脚本
for script in scripts:
    subprocess.run(['python', script])
