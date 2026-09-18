# Flowers
В Первую очередь нужно запустить файл `AutoEncoder.py` и подождать пока результат не начнёт вас удовлетворять(предполагаемая эпоха 200-300)
Затем в файле `Generator.py` в 49 строчке кода нужно прописать нужный файл сохранения эпохи AutoEncoder

`model.load_state_dict(torch.load('models/torch_e{ваш номер эпохи}'))`

подождать сходимости(сохранение каждые 500 эпох)
затем в файле `test.py` с 4 по 8 строчку нужно прописать все файлы

`model.load_state_dict(torch.load('models/torch_e{эпоха автоэнкодера}'))`  
`gen=Generator(16,256)`  
`gen.load_state_dict(torch.load(f'models/gen_e{эпоха генератора}'))`  
`gen=gen.to(device)`  
`embeddings=torch.load(f'models/emb_e{эпоха генератора}')['emb'].to(device)`
