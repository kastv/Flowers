from AutoEncoder import model, torch, np, x, pygame, device
from Generator import Generator

model.load_state_dict(torch.load('models/torch_e300'))
gen=Generator(16,256)
gen.load_state_dict(torch.load(f'models/gen_e500'))
gen=gen.to(device)
embeddings=torch.load(f'models/emb_e500')['emb'].to(device)

sc=pygame.display.set_mode((320,320))
clock=pygame.time.Clock()
encoded=model.encode(x)

mask=torch.rand((20,16),device=device)
emb=embeddings[torch.randint(0,4,(20,))][None,:]*mask+embeddings[torch.randint(0,4,(20,))][None,:]*(1-mask)
generated=gen.genetate(emb[0],5)
encoded=generated
#encoded=torch.normal(0,1,(20,4*4*16),device=device)

idx=0
a=0
while True:
    start = encoded[(a)%20]
    end = encoded[(a+1)%20]
    arr=pygame.Surface((32,32))
    t=max(0,min(1,idx/100))
    if idx==120:
        a+=1
        idx=0
    cur=start+(end-start)*t
    model.eval()
    decoded,_,_=model.decode(cur[None,...],0)
    decoded=decoded[0]
    decoded=torch.moveaxis(decoded,(1,2,0),(0,1,2)).cpu().detach().numpy()
    pygame.surfarray.array_to_surface(arr,np.uint8(decoded*255))
    arr=pygame.transform.scale(arr,(320,320))
    sc.blit(arr,arr.get_rect(center=(160,160)))
    for i in pygame.event.get():
        if i.type==pygame.QUIT:
            quit()
    pygame.display.flip()
    clock.tick(60)
    idx+=1
