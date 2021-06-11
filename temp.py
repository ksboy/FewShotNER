import torch
path = "./output/conll->wnut/structshot_support_fixed.pth.tar"
model = torch.load(path, map_location=torch.device('cpu'))
keys = list(model['state_dict'].keys())
for key in keys:
    if not key.startswith('word_encoder.module.'):
        print("fail")
    else:
        print('true')



# model["hyper_parameters"]["labels"]="./data/conll-2003/labels.txt"
# torch.save(model, "./output/conll-2003/checkpointepoch=2.ckpt")