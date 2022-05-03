import itk
import torch
from torchvision import transforms
import glob


dataset_path = '../PVE_data/Analytical_data/dataset'
batchsize = 3
prct_train = 0.8




transform = transforms.Compose([transforms.ToTensor()])

dataset = torch.empty((1,2,128,128))


for filename_ in glob.glob(f'{dataset_path}/?????.mhd'):
    filename_PVE = f'{filename_[:-4]}_PVE.mhd'
    img_PVE = itk.array_from_image(itk.imread(filename_PVE))
    tensor_PVE = torch.from_numpy(img_PVE)[None,:]

    filename_PVf = f'{filename_[:-4]}_PVfree.mhd'
    img_PVf = itk.array_from_image(itk.imread(filename_PVf))
    tensor_PVf = torch.from_numpy(img_PVf)[None,:]


    cat_PVf_PVE = torch.cat((tensor_PVf,tensor_PVE), 1)

    dataset = torch.cat((dataset,cat_PVf_PVE),0)


# Division of dataset into train_dataset and test_dataset
train_size = int(prct_train* len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)



# for step, X in enumerate(test_dataloader):
#     print(step)
#     print(X.shape)