import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from data.gp_dataloader import GPDataGenerator
from data.mask_generator import GetBatchMask
from utils import plot_np_results


# Set the random seed for reproducibility
# torch.manual_seed(1)
# np.random.seed(1)

plot_freq= 10000


def train_1d(
    model,
    model_name,
    epochs=10000,
    train_gen=GPDataGenerator(),
    test_gen=GPDataGenerator(testing=True, batch_size=1),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        context_x, context_y, target_x, target_y = prepare_data(train_gen, device)

        optimizer.zero_grad()
        pred_y, std, loss = model(context_x, context_y, target_x, target_y)
        loss.backward()
        optimizer.step()

        if epoch % plot_freq == 0 or (epoch+1)==epochs:
            model.eval()
            with torch.no_grad():
              context_x, context_y, target_x, target_y = prepare_data(test_gen, device)
              pred_y, std, loss= model(context_x, context_y, target_x, target_y)
            
              plot_np_results(
                  *prepare_plot([target_x, target_y, context_x, context_y, pred_y, std]),
                  title=f"Epoch: {epoch} Loss: {loss.item():.4f}",
              )
            

            os.makedirs("/content/drive/MyDrive/neural-processes-main/models/convcnp/1d_saved", exist_ok=True)
              
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "/content/drive/MyDrive/neural-processes-main/models/convcnp/1d_saved/"+"convcnp_model"+"_"+model_name+"_"+str(epoch)+".pt",
            )


def prepare_data(generator, device):
    context_x, context_y, target_x, target_y = generator.generate_batch().get_all()
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    target_x = target_x.to(device)
    target_y = target_y.to(device)
    return context_x, context_y, target_x, target_y


def prepare_plot(objects):
    if not isinstance(objects, list):
        objects = [objects]
    return [obj.detach().cpu().numpy()[0] for obj in objects]



def train_2d(
    model,
    model_name,
    train_loader,
    val_loader,
    a,
    b,
    epochs=50,
    report_interval=400,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    masker = GetBatchMask(a=a, b=b)
    def train_one_epoch(epoch):
        running_loss = 0.0
        last_loss = 0.0
        
        for idx, (data,_) in enumerate(train_loader):
            data = data.to(device)
            mask = masker(data.shape[0], (data.shape[2], data.shape[3]))
            mask = mask.to(device)
            optimizer.zero_grad()
            pred_y, std, loss = model(data, mask)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if idx % report_interval + 1 == report_interval:
                last_loss = running_loss / report_interval
                print("  batch {} loss: {}".format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.0
        return last_loss
            
    best_val_loss = 1e20
    for epoch in range(epochs):
        
        print("EPOCH {}:".format(epoch + 1))
        model.train(True)
        avg_training_loss = train_one_epoch(epoch)
    
        model.train(False)
        running_val_loss = 0.0
        with torch.no_grad():
            for idx, (data, _) in enumerate(val_loader):
                
                data=data.to(device)
                mask = masker(data.shape[0], (data.shape[2], data.shape[3]))
                mask = mask.to(device)
                pred_y_val, std_val, val_loss = model(data, mask)
                running_val_loss += val_loss
            
                if idx==0:
                    if data.shape[1]==1:
                        query = random.randint(0, data.shape[0])
                        img = data[query].squeeze(0)
                        img = img.detach().cpu()
                        single_mask = mask[query].squeeze(0)
                        single_mask = single_mask.detach().cpu()
                        masked_img = img.clone()
                        masked_img[single_mask==0.0] = np.NaN
                        predicted_img = pred_y_val[query].squeeze(-1)
                        predicted_img = predicted_img.detach().cpu()
                        plt.figure()

                        f, axe = plt.subplots(1,3) 
                        axe[0].imshow(img,cmap="gray")
                        axe[0].xaxis.set_tick_params(labelbottom=False)
                        axe[0].yaxis.set_tick_params(labelleft=False)
                        axe[0].set_xticks([])
                        axe[0].set_yticks([])
                        axe[1].imshow(masked_img,cmap="gray")
                        axe[1].xaxis.set_tick_params(labelbottom=False)
                        axe[1].yaxis.set_tick_params(labelleft=False)
                        axe[1].set_xticks([])
                        axe[1].set_yticks([])
                        axe[1].patch.set_facecolor('blue')
                        axe[2].imshow(predicted_img,cmap="gray")
                        axe[2].xaxis.set_tick_params(labelbottom=False)
                        axe[2].yaxis.set_tick_params(labelleft=False)
                        axe[2].set_xticks([])
                        axe[2].set_yticks([])
                    
                    else:
                        query = random.randint(0, data.shape[0])
                        img = data[query].permute(1,2,0)
                        img = img.detach().cpu()
                        single_mask = mask[query].permute(1,2,0)
                        single_mask = single_mask.detach().cpu()
                        masked_img = img.clone()
                        masked_img[(single_mask.expand_as(img))==0.0] = np.NaN
                        predicted_img = pred_y_val[query]
                        predicted_img = predicted_img.detach().cpu()
                        plt.figure()

                        f, axe = plt.subplots(1,3) 
                        axe[0].imshow(img)
                        axe[0].xaxis.set_tick_params(labelbottom=False)
                        axe[0].yaxis.set_tick_params(labelleft=False)
                        axe[0].set_xticks([])
                        axe[0].set_yticks([])
                        axe[1].imshow(masked_img)
                        axe[1].xaxis.set_tick_params(labelbottom=False)
                        axe[1].yaxis.set_tick_params(labelleft=False)
                        axe[1].set_xticks([])
                        axe[1].set_yticks([])
                        axe[1].patch.set_facecolor('black')
                        axe[2].imshow(predicted_img)
                        axe[2].xaxis.set_tick_params(labelbottom=False)
                        axe[2].yaxis.set_tick_params(labelleft=False)
                        axe[2].set_xticks([])
                        axe[2].set_yticks([])
                    
            
        
            avg_val_loss = running_val_loss/len(val_loader)
        
        print(
            "LOSS train {:.5f} valid {:.5f}".format(
                avg_training_loss, avg_val_loss
            )
        )
        
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
            os.makedirs("/content/drive/MyDrive/neural-processes-main/models/convcnp/2d_saved", exist_ok=True)
              
            torch.save(
               {
                   "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
               },
               "/content/drive/MyDrive/neural-processes-main/models/convcnp/2d_saved/"+"gridconvcnp_model"+"_"+model_name+"_"+str(epoch+1)+".pt",
           )
            model_path =  "/content/drive/MyDrive/neural-processes-main/models/convcnp/2d_saved/"+"gridconvcnp_model"+"_"+model_name+"_"+str(epoch+1)+".pt"
    
    return model_path
