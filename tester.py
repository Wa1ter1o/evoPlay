import torch
print(torch.cuda.is_available())   # Debe dar True
print(torch.cuda.get_device_name(0))  # Nombre de tu GPU