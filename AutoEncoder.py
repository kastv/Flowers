import pygame
import threading
import torch
import numpy as np
import os
import tqdm
from Tools.scripts.pathfix import keep_flags
from torch import nn
import torch_directml as dml
import random
import time
import math
#device = dml.device()
device='cpu'
print(device)

flowers:dict[str,list[pygame.Surface]]={}
flowers_arr:dict[str,list[torch.Tensor]]={}
for flower in os.listdir('flowers'):
    flowers[flower]=[]
    flowers_arr[flower]=[]
    for i in os.listdir(f'flowers/{flower}'):
        surf=pygame.image.load(f'flowers/{flower}/{i}')
        arr=np.zeros((32,32,3),dtype=np.uint8)
        pygame.surfarray.surface_to_array(arr,surf)
        flowers[flower].append(surf)
        flowers_arr[flower].append(torch.Tensor(arr)[None,...])
flowers_conc:dict[str,torch.Tensor]={}
for i in flowers_arr:
    flowers_conc[i]=torch.concatenate(flowers_arr[i],dim=0)

embeddings=torch.tensor(np.random.normal(0,1,(len(flowers_conc),16)),dtype=torch.float32,requires_grad=True,device=device)
x=torch.concatenate([flowers_conc[i] for i in flowers_conc],dim=0)

class Reshape(nn.Module):
    def __init__(self,*shape):
        super().__init__()
        self.shape=shape
    def forward(self,x):
        return x.reshape(self.shape)
class MyRELU(nn.Module):
    def forward(self,x):
        return 0.2 * x + 0.8 * torch.clamp(x, 0.0, 1.0)
class MyDropout(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.p=p
    def forward(self,x:torch.tensor):
        if self.training:
            m=torch.bernoulli(x,p=self.p).detach()
            u=torch.normal(float(torch.mean(x)),float(torch.std(x)),tuple(x.shape)).to(x.device)
            return (1-m)*x+m*u
        else:
            return x
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3, 8, (7, 7), 2, 3),
            MyRELU(),
            nn.InstanceNorm2d(8),
            MyDropout(0),
            Reshape(-1, 16 * 16 * 8),
            nn.Linear(16 * 16 * 8, 16 * 16 * 8),
            MyRELU(),
            nn.LayerNorm(16 * 16 * 8),
            MyDropout(0),
            Reshape(-1, 8, 16, 16),
            nn.Conv2d(8, 16, (5, 5), 2, 2),
            MyRELU(),
            nn.InstanceNorm2d(16),
            MyDropout(0),
            Reshape(-1, 8 * 8 * 16),
            nn.Linear(8 * 8 * 16, 8 * 8 * 16),
            MyRELU(),
            nn.LayerNorm(8 * 8 * 16),
            MyDropout(0),
            Reshape(-1, 16, 8, 8),
            nn.Conv2d(16, 16, (3, 3), 2, 1),
            MyRELU(),
            nn.InstanceNorm2d(16),
            MyDropout(0),
            Reshape(-1, 4 * 4 * 16),
            nn.Linear(4 * 4 * 16, 4 * 4 * 16),
            MyRELU(),
            MyDropout(0),
            nn.LayerNorm(4 * 4 * 16),
        )
        self.decoder=nn.Sequential(
            Reshape(-1,16+1,4,4),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(17,8,(7,7),padding=3),
            MyDropout(0),
            nn.InstanceNorm2d(8),
            Reshape(-1,16*16*8),
            nn.Linear(16*16*8,16*16*8),
            MyRELU(),
            MyDropout(0),
            Reshape(-1,8,16,16),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(8,4,(5,5),padding=2),
            nn.Sigmoid()
        )
    def decode(self,x,noice):
        inp=torch.concatenate((x,torch.ones((x.shape[0],16),device=x.device)*noice),dim=1)
        encoded=self.decoder(inp)
        color,mask=encoded[:,0:3,:,:],encoded[:,3,:,:][:,None,:,:]
        t_mask=torch.where(mask>0.25,mask,0)
        out=color * (mask + (t_mask - mask).detach())
        return out,mask,color
    def encode(self,x):
        return self.encoder(x)
    def Forward(self,x):
        encoded=self.encode(x)
        return self.decode(encoded)
x = torch.swapdims(x, 1, 3)
x = torch.swapdims(x, 2, 3)/255
x = x.type(torch.float32)
x=x.to(device=device)
class dataset(torch.utils.data.Dataset):
    def __init__(self,x):
        self.x=x
        self.indexes=torch.range(0,20)//4
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx],self.indexes[idx]
draw=False
mask=torch.zeros((5,1,32,32))
color=torch.zeros((5,3,32,32))
model = Model().to(device)

disc_true_out=torch.concatenate((torch.ones((x.shape[0],1)),torch.zeros((x.shape[0],1))),dim=0).to(device)
def learn_thread():
    global out,draw,embeddings,model
    gen_optim=torch.optim.Adam(list(model.parameters()),lr=1e-4,betas=(0.5,0.999))
    model.compile()
    epoch=0
    while u_t.is_alive():
        w1=1-0.0**epoch
        w2=0.8*0.0**epoch+0.2
        model.zero_grad()
        gen_optim.zero_grad()
        encoded=model.encode(x)
        noice=((math.sin(math.pi*epoch/60)+1)/2)**3*(0)
        encoded+=torch.normal(0,1,encoded.shape,device=device)*noice
        fake, mask, _ = model.decode(encoded,noice)
        out=fake
        x_mask=torch.where(x>0,1,0)

        draw = True
        gen_score_error=torch.mean((fake-x)**2)**0.5*w1+w2*torch.mean((mask-x_mask)**2)**0.5
        gen_score_error.backward()
        gen_optim.step()

        print(f'e:{epoch} gen_err: {gen_score_error:.4} n:{noice:.4}')
        epoch += 1
        while draw:pass
        if epoch%100==0:
            torch.save(model.state_dict(),f'models/torch_e{epoch}')
def ui_thread():
    global out,draw,model
    #while not draw:pass
    sc=pygame.display.set_mode(((320,320)))
    st=time.time()
    while True:
        if draw:
            if time.time()-st>0:
                sc.fill((0,0,0))
                rm=random.randint(-2,2)
                idx=0#random.randint(0,x_batch.shape[0]-1)
                predicted=out.cpu().detach().numpy()
                predicted=np.moveaxis(predicted*255,(0,2,3,1),(0,1,2,3)).astype(np.uint8)

                surf=pygame.Surface((32,32))
                pygame.surfarray.array_to_surface(surf,predicted[2+rm])
                t_suft=pygame.transform.scale(surf,(128,128))
                sc.blit(t_suft,t_suft.get_rect(center=(94,94)))

                pygame.surfarray.array_to_surface(surf, predicted[7+rm])
                t_suft = pygame.transform.scale(surf, (128, 128))
                sc.blit(t_suft, t_suft.get_rect(center=(320-94, 94)))

                pygame.surfarray.array_to_surface(surf, predicted[12+rm])
                t_suft = pygame.transform.scale(surf, (128, 128))
                sc.blit(t_suft, t_suft.get_rect(center=(94, 320-94)))

                pygame.surfarray.array_to_surface(surf, predicted[17+rm])
                t_suft = pygame.transform.scale(surf, (128, 128))
                sc.blit(t_suft, t_suft.get_rect(center=(320-94, 320-94)))
                st=time.time()
            draw=False
        for i in pygame.event.get():
            if i.type==pygame.QUIT:
                quit()
            if i.type==pygame.KEYDOWN:
                if i.key==pygame.K_l:
                    model.load_state_dict(torch.load(f'models/torch_e{300}'))
        pygame.display.flip()
if __name__=='__main__':
    l_t=threading.Thread(target=learn_thread)
    u_t = threading.Thread(target=ui_thread)
    u_t.start()
    l_t.start()
    l_t.join()