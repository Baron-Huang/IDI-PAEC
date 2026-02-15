import subprocess
import os

# 列表中包含你想要顺序执行的脚本名
read_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Codes/Main_func_SOTAs'
model_names = ['ILRA_MIL', 'DTFD_MIL']

scripts = []
#scripts.append(r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Codes/IGI_PAEC_main_DHMC_Kidney.py')
for name_i in model_names:
    scripts.append(os.path.join(read_path, name_i + '.py'))



# 遍历列表，依次运行每个脚本
for script in scripts:
    subprocess.run(['python', script])
