from data.dataset import load_data
from models.Pix2PixModel import PVEPix2PixModel
import time
import json
import click



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('json_filename', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--output', '-o', help='Output filename, default = automatic name', default='auto')
@click.option('--output_folder', '-f', help='Output folder (ignored if output is not "auto")', default='.')
def train(json_filename, output, output_folder):
    params_file = open(json_filename).read()
    params = json.loads(params_file)
    data_params = params['data_params']
    training_params = params['training_params']
    losses_params = params['losses_params']


    train_dataloader, test_dataloader = load_data(dataset_path=data_params['dataset_path'],
                                                  training_batchsize=data_params['training_batchsize'],
                                                  testing_batchsize=data_params['test_batchsize'],
                                                  prct_train=data_params['training_prct'])

    DeepPVEModel = PVEPix2PixModel(training_params=training_params, losses_params=losses_params)

    DeepPVEModel.show_infos()

    # ajouter la date aux params
    # "training_date": time.asctime()


    # t0 = time.time()
    # for epoch in range(DeepPVEModel.n_epochs):
    #     print(f'Epoch {epoch}/{DeepPVEModel.n_epochs}')
    #     for step,batch in enumerate(train_dataloader):
    #         print(f'step {step}/{len(train_dataloader)-1}.........................')
    #
    #         DeepPVEModel.input_data(batch)
    #
    #         DeepPVEModel.optimize_parameters()
    #
    #         if (step % DeepPVEModel.display_step == 0):
    #             DeepPVEModel.display()
    #
    #     DeepPVEModel.update_epoch()
    #
    # DeepPVEModel.save_model()
    # tf = time.time()
    # total_time = round(tf-t0)
    # print(f'Total training time : {total_time} s')
    # DeepPVEModel.plot_losses()






if __name__ == '__main__':
    train()


