import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.optim as optim
import time
import argparse
import datetime
import model as md
import wandb




class CustomDataset(Dataset):
    def __init__(self,json_path):
        self.root_dir = json_path
        self.samples = self._load_samples()


    def _load_samples(self):
        samples=[]
        data_dict=torch.load(self.root_dir)
        for key in data_dict.keys():
            samples.append([data_dict[key]['image_features'][0],data_dict[key]['description_embeddings'][0],data_dict[key]['target_index']])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_features, description_embeddings, target_index= self.samples[idx]
        return image_features, description_embeddings, target_index


# Define your DataLoader
def get_dataloader(root_dir, batch_size):
    dataset = CustomDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)









if __name__ == "__main__":

    # Crear un objeto ArgumentParser
    parser = argparse.ArgumentParser(description='Descripción del programa')

    # Agregar los argumentos
    parser.add_argument('--ruta_features_train', type=str,
                        default="Features_LLaVA_CLIP_train_16.pt",
                        help='Training path')
    parser.add_argument('--ruta_features_val', type=str,
                        default="Features_LLaVA_CLIP_val_16.pt",
                        help='Validation path')
    parser.add_argument('--ruta_features_test', type=str,
                        default="Features_LLaVA_CLIP_test_16.pt",
                        help='Test path')
    #parser.add_argument('--weight_Clip', type=float, default=0.5, help='Descripción del peso de Clip')
    #parser.add_argument('--num_epochs', type=int, default=30, help='Número de épocas')
    #parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--pretrained', type=int, default=0, help='pretrained ')
    #parser.add_argument('--num_layers', type=int, default=1, help='num_layers ')
    #parser.add_argument('--dropout', type=float, default=0.5, help='dropout ')
    #parser.add_argument('--hidden_dim', type=int, default=512 * 4, help='hidden_dim ')
    #parser.add_argument('--lr', type=float, default=0.01, help='learning rate ')
    #parser.add_argument('--momentum', type=float, default=0.9, help='momentum ')
    #parser.add_argument('--patience', type=int, default=10, help='patience ')

    # Parsear los argumentos
    args = parser.parse_args()

    # Acceder a los valores de los argumentos
    ruta_features_train = args.ruta_features_train
    ruta_features_val = args.ruta_features_val
    ruta_features_test = args.ruta_features_test
    #weight_Clip = args.weight_Clip
    #num_epochs = args.num_epochs
    #batch_size = args.batch_size
    pretrained = args.pretrained
    #num_layers = args.num_layers
    #dropout = args.dropout
    #hidden_dim = args.hidden_dim
    #lr = args.lr
    #momentum = args.momentum
    #patience = args.patience

    # ruta_img_train='D:/Udea/Maestria/Bases_de_datos/eccv_18_all_images_sm/Organizado/categorias/train'
    # ruta_img_val = 'D:/Udea/Maestria/Bases_de_datos/eccv_18_all_images_sm/Organizado/categorias/cis_val'

    wandb.login(key="282780c770de0083eddfa3c56402f555ee60e108")

    # Configurar las settings de W&B
    sweep_config = {
        'method': 'random',
         'name': 'Serengeti_Train_modelV1_loss_supervised_contrastive_16',
    }
    metric = {
        'name': 'loss_val',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'opt': {
            'values': ['sgd']
        },
        't': {
            'values': [1,0.1, 0.01, 0.001]
        },
    }

    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'momentum': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 4,
            'max': 256,
        },
        'weight_Clip': {
            'distribution': 'uniform',
            'min': 0.4,
            'max': 0.8,
        },
        'hidden_dim': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 1024,
            'max': 2048,
        },
        'dropout': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5,
        },
        'num_layers': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 1,
            'max': 10,
        },
        'num_epochs': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 1,
            'max': 200,
        }
    })
    parameters_dict.update({
        'patience': {
            'value': 20}
    })
    sweep_id = wandb.sweep(sweep_config, project="LlaVA_Clip")


    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    # Generate a unique identifier based on the current date and time
    def train(config=None):
        set_seed(42)
        with wandb.init(config=config):
            config = wandb.config
            config.name = f"run_num_layers_{config.num_layers}_lr_{config.lr}_weight_Clip_{config.weight_Clip}_dropout_{config.dropout}"
            #wandb.init(
            #    # set the wandb project where this run will be logged
            #    project="LlaVA_Clip",
            #    config=config)
            hidden_dim=config.hidden_dim
            num_layers=config.num_layers
            dropout=config.dropout
            batch_size=config.batch_size
            opt=config.opt
            lr=config.lr
            momentum=config.momentum
            num_epochs=config.num_epochs
            weight_Clip=config.weight_Clip
            patience=config.patience
            t=config.t

            unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "cpu:":
                # Get the number of available GPUs
                num_gpus = torch.cuda.device_count()
                # Create a list of devices
                devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
            else:
                # If CUDA is not available, just use CPU
                devices = [torch.device("cpu")]
            print('num_gpus: ', num_gpus, devices)

            text_features=torch.load('Text_features_16.pt')
            text_features = text_features.to(device)



            projection_model = md.LLaVA_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
                                          pretrained=pretrained)  # MLP(input_dim=768, hidden_dim=hidden_dim, output_dim=512, num_layers=num_layers, dropout=dropout, return_embeds=True)

            projection_model = projection_model.to(device)

            # Get your DataLoader
            dataloader = get_dataloader(ruta_features_train, batch_size)
            dataloader_val = get_dataloader(ruta_features_val, batch_size)
            dataloader_test = get_dataloader(ruta_features_test, batch_size)




            def build_optimizer( optimizer, learning_rate, momentum):
                # Definir los parámetros y configuraciones de optimización para cada conjunto de parámetros
                params1 = {"params": projection_model.description_encoder.parameters(), "lr": learning_rate,
                           "momentum": momentum}
                params2 = {"params": projection_model.logit_scale_CLIP, "lr": learning_rate, "momentum": momentum}
                params3 = {"params": projection_model.logit_scale_LLaVA, "lr": learning_rate, "momentum": momentum}

                # Inicializar el optimizador con todos los conjuntos de parámetros
                if optimizer == "sgd":
                    optimizer = optim.SGD([params1, params2, params3], lr=learning_rate, momentum=momentum)
                elif optimizer == "adam":
                    optimizer = optim.Adam([params1, params2, params3], lr=learning_rate)

                return optimizer

            # Optimizer
            #params1 = {"params": projection_model.description_encoder.parameters(), "lr": lr, "momentum": momentum}
            #params2 = {"params": projection_model.logit_scale_CLIP, "lr": lr, "momentum": momentum}
            #params3 = {"params": projection_model.logit_scale_LLaVA, "lr": lr, "momentum": momentum}

            optimizer = build_optimizer(opt,lr,momentum)#optim.SGD([params1, params2,params3])


            # Function to measure GPU memory usage
            def get_gpu_memory_usage():
                return torch.cuda.memory_allocated() / 1024 ** 2  # Convert bytes to megabytes


            # Now you can use this dataloader in your training loop
            acc_best = 0
            counter = 0
            for epoch in range(num_epochs):
                print(epoch)
                # Training
                projection_model.train()
                time_in = time.time()
                running_loss = 0.0
                running_corrects = 0.0
                size=0
                for batch in dataloader:
                    image_features, description_embeddings, target_index = batch
                    size+=len(image_features)
                    image_features=image_features.to(device)
                    description_embeddings = description_embeddings.to(device)


                    #batch_text = text_features.t()[target_index]

                    loss, acc = projection_model(description_embeddings, image_features, text_features, weight_Clip,target_index,t)

                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # Update the parameters
                    optimizer.step()
                    # Zero the gradients
                    optimizer.zero_grad()

                    # statistics Train
                    running_loss += loss.item()
                    running_corrects += float(acc)

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = (running_corrects / size) * 100

                # validation
                projection_model.eval()

                running_loss_val = 0
                running_corrects_val = 0.0
                size_val = 0
                with torch.no_grad():

                    for batch_val in dataloader_val:
                        image_features_val, description_embeddings_val, target_index_val = batch_val
                        size_val += len(image_features_val)
                        image_features_val = image_features_val.to(device)
                        description_embeddings_val = description_embeddings_val.to(device)

                        # batch_text_val = text_features.t()[target_index_val]

                        loss_val, acc_val = projection_model(description_embeddings_val,
                                                                     image_features_val, text_features,
                                                                     weight_Clip, target_index_val,t)

                        running_loss_val += loss_val.item()
                        running_corrects_val += float(acc_val)


                epoch_loss_val = running_loss_val / len(dataloader_val)
                epoch_acc_val = (running_corrects_val / size_val) * 100

                # log metrics to wandb #
                wandb.log({"acc_train": epoch_acc, "loss_train": epoch_loss, "acc_val": epoch_acc_val,
                           "loss_val": epoch_loss_val})
                time_end = time.time()
                tiempo_transcurrido = time_end - time_in
                # Print the loss at every nth epoch
                print_every = 1
                if epoch % print_every == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}]")
                    print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
                    print('Val loss: {:.4f}, Val acc: {:.4f}'.format(epoch_loss_val, epoch_acc_val))
                    print(f"Time for epoch [{tiempo_transcurrido}]")
                    if epoch_acc_val > acc_best:
                        print('Save model')
                        acc_best = epoch_acc_val
                        counter = 0

                        # Create a directory for each training session based on the unique identifier
                        os.makedirs(f'Super_Contrative_16_V1/training_{unique_id}', exist_ok=True)
                        # Save the model parameters within the corresponding directory
                        model_params_path = f'Super_Contrative_16_V1/training_{unique_id}/best_model_params_{num_layers}_{hidden_dim}.pth'
                        torch.save(projection_model.state_dict(), model_params_path)

                    else:
                        counter = counter + 1
                        print("The acc don't increase")
                if epoch==(num_epochs-1) or counter >= patience:
                    projection_model = md.LLaVA_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
                                                     pretrained=0, pretrained_path="")
                    projection_model.load_state_dict(torch.load(model_params_path))
                    projection_model = projection_model.to(device)
                    projection_model.eval()

                    running_loss_test = 0
                    running_corrects_test = 0.0
                    size_test = 0
                    with torch.no_grad():

                        for batch_test in dataloader_test:
                            image_features_test, description_embeddings_test, target_index_test = batch_test
                            size_test += len(image_features_test)
                            image_features_test = image_features_test.to(device)
                            description_embeddings_test = description_embeddings_test.to(device)

                            # batch_text_val = text_features.t()[target_index_val]

                            loss_test, acc_test = projection_model(description_embeddings_test,
                                                                         image_features_test, text_features,
                                                                         weight_Clip, target_index_test, t)

                            running_loss_test += loss_test.item()
                            running_corrects_test += float(acc_test)



                    epoch_loss_test = running_loss_test / len(dataloader_test)
                    epoch_acc_test = (running_corrects_test / size_test) * 100

                    print('Test loss: {:.4f},Test acc: {:.4f}'.format(epoch_loss_test, epoch_acc_test))
                    wandb.log({"acc_test": epoch_acc_test,"loss_test": epoch_loss_test})

                # Check early stopping condition
                if counter >= patience:
                    print(f'Validation acc has not improved for {patience} epochs. Stopping training.')
                    break


    wandb.agent(sweep_id, train, count=100)
    wandb.finish()