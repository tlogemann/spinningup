import os
import torch
from pathlib import Path

model = None

# Source: https://stackoverflow.com/a/50098973
file_dir = str(Path().absolute())

pyt_save_path = os.path.join(file_dir, '../../../data/ddpg/ddpg_s0/pyt_save')

model_path = os.path.join(pyt_save_path, 'model.pt')

if os.path.isfile(model_path):
    ac = torch.load(model_path)
    model = ac.pi.pi
    extracted_model_path = os.path.join(pyt_save_path, 'model_ac_pi_pi.pt')
    torch.save(model, extracted_model_path)


