import torch
import matplotlib.pyplot as plt
from train import Gen, Disc, gen_opt, disc_opt,test_dataloader,show_tensor_images
import numpy as np

pth_file = f'pix2pix_50.pth'

checkpoint = torch.load(pth_file)



generator_losses = checkpoint['gen_losses']
discriminator_losses = checkpoint['disc_losses']

plt.plot(discriminator_losses, label = 'Disc')
plt.plot(generator_losses, label = 'Gen')
plt.show()


Gen.load_state_dict(checkpoint['gen'])
Gen.eval()

MSE = 0

for (batch,test_data) in enumerate(test_dataloader):
    truePVE = test_data[:,0,:,:].unsqueeze(1)
    truePVfree = test_data[:,1,:,:].unsqueeze(1)

    fakePVfree = Gen(truePVE.float())


    print(torch.mean((fakePVfree-truePVfree)**2, axis = (0,1,2,3)))

    # show_tensor_images(torch.cat((truePVE, truePVfree, fakePVfree), 1))



