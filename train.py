from data.dataset import load_data
from functions.losses import Pix2PixLosses
from models.Pix2PixModel import  PVEPix2PixModel


data_params = {'dataset_path': '../PVE_data/Analytical_data/dataset','training_batchsize':5, 'test_batchsize':5, 'training_prct':0.3}
training_params = {'n_epochs':2,'learning_rate':0.0002, 'input_channels':1, 'hidden_channels_gen':64, 'hidden_channels_disc':9, 'display_step':30,'optimizer':'Adam','training_device':'cpu'}
losses_params = {'adv_loss':'BCE', 'recon_loss':'L1','lambda_recon':200 }

train_dataloader,test_dataloader = load_data(dataset_path=data_params['dataset_path'],
                                             training_batchsize=data_params['training_batchsize'], testing_batchsize=data_params['test_batchsize'],prct_train=data_params['training_prct'])

pix2pixlosses = Pix2PixLosses(losses_params)

DeepPVEModel = PVEPix2PixModel(training_params=training_params, losses = pix2pixlosses)


def train():

    for epoch in range(DeepPVEModel.n_epochs):
        print(f'Epoch {epoch}/{DeepPVEModel.n_epochs}')
        for step,batch in enumerate(train_dataloader):
            print(f'step {step}/{len(train_dataloader)-1}.........................')

            DeepPVEModel.input_data(batch)

            DeepPVEModel.optimize_parameters()

            if (step % DeepPVEModel.display_step == 0):
                DeepPVEModel.display()

        DeepPVEModel.update_epoch()

    DeepPVEModel.save_model()
    DeepPVEModel.plot_losses()






if __name__ == '__main__':
    train()


