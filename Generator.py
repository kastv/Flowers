from AutoEncoder import Model, torch, np, pygame, device, x, nn, MyRELU
class generator_block(nn.Module):
    def __init__(self,emb_size,inp_size):
        super().__init__()
        self.generator_block = nn.Sequential(
            nn.Linear(emb_size + inp_size + 1, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, int(256 * 1.5)),
            nn.GELU(),
            nn.LayerNorm(int(256 * 1.5)),
            nn.Linear(int(256 * 1.5), 256),
        )
    def forward(self,x):
        return self.generator_block.forward(x)
class Generator(nn.Module):
    def __init__(self,emb_size=16,inp_size=256):
        super().__init__()
        self.emb_size=emb_size
        self.inp_size=inp_size
        self.generator_block1=generator_block(emb_size,inp_size)
        self.generator_block2=generator_block(emb_size,inp_size)
        self.generator_block3=generator_block(emb_size,inp_size)
        self.weights=torch.nn.Parameter(torch.zeros((3,),requires_grad=True))
    def forward(self,emb,inp_vector,t):
        t=torch.ones((20,1),device=device)*t
        inp=torch.concatenate((emb,inp_vector,t),dim=1)
        inp_vector=inp_vector+self.generator_block1.forward(inp)*self.weights[0]
        inp = torch.concatenate((emb, inp_vector,t), dim=1)
        inp_vector = inp_vector + self.generator_block2.forward(inp)*self.weights[1]
        inp = torch.concatenate((emb, inp_vector,t), dim=1)
        inp_vector = inp_vector + self.generator_block3.forward(inp)*self.weights[2]
        return inp_vector
    def genetate(self,emb,num_steps):
        out=torch.normal(0,1,(emb.shape[0],self.inp_size),device=emb.device)
        for i in range(num_steps,1,-1):
            t=i/num_steps
            t_prev=(i-1)/num_steps
            pred=self.forward(emb,out,t)
            noice_pred=(out-(1-t**0.5)*pred)/(t**0.5+1e-8)
            out=t_prev**0.5*noice_pred+(1-t_prev**0.5)*pred
        out=self.forward(emb,out,0)
        return out


if __name__=='__main__':
    embendings = torch.normal(0, 1, (4, 16), device=device, requires_grad=True)
    model = Model()
    model.load_state_dict(torch.load('models/torch_e300'))
    model.to(device)
    true_outs = model.encode(x).detach()
    gen=Generator(16,256).to(device)
    optim=torch.optim.Adam([*gen.parameters(),embendings],lr=0.001)
    e=1
    while True:
        inp_emb=embendings[torch.arange(20)//5]
        out_emb=true_outs
        error=0
        num_steps=30
        noice=torch.normal(0,1,(20,256),device=device)
        for i in range(1,num_steps+1):
            pemb=out_emb
            inp=(i/num_steps)**0.5*noice+(1-(i/num_steps)**0.5)*true_outs
            model_out=gen.forward(inp_emb,inp,i/num_steps)
            local_error=torch.mean((true_outs-model_out)**2)
            error+=local_error
            t=i/num_steps
            print(f'e:{e} t={t:.2f}  err={local_error:.4f}, initial_error:{torch.mean((inp-true_outs)**2)**0.5}')
        if e%500==0:
            torch.save(gen.state_dict(),f'models/gen_e{e}')
            torch.save({'emb':embendings.detach()},f'models/emb_e{e}')
        error=(error/num_steps)**0.5
        optim.zero_grad()
        error.backward()
        optim.step()
        print()
        e+=1