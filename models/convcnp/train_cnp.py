import torch
from pathlib import Path
from datetime import datetime
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
    epochs=10000,
    train_gen=GPDataGenerator(),
    test_gen=GPDataGenerator(testing=True, batch_size=1),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    NLL = []
    Path("checkpoint/{}".format(timestamp)).mkdir(parents=True, exist_ok=True)   

    for epoch in range(epochs):
        model.train()
        context_x, context_y, target_x, target_y = prepare_data(train_gen, device)

        optimizer.zero_grad()
        pred_y, std, loss = model(context_x, context_y, target_x, target_y)
        loss.backward()
        optimizer.step()

        if (epoch+1) % plot_freq == 0 or (epoch+1)==epochs:
            model.eval()
            with torch.no_grad():
                context_x, context_y, target_x, target_y = prepare_data(test_gen, device)
                pred_y, std, loss= model(context_x, context_y, target_x, target_y)
            
                plot_np_results(
                  *prepare_plot([target_x, target_y, context_x, context_y, pred_y, std]),
                   title=f"Epoch: {epoch} Loss: {loss.item():.4f}",
              )
                model_path = "checkpoint/{}/model_{}".format(timestamp, epoch + 1) 
                torch.save(
                 model.state_dict(),
                 model_path
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
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    NLL = []
    Path("checkpoint/{}".format(timestamp)).mkdir(parents=True, exist_ok=True)      
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
        
            avg_val_loss = running_val_loss/len(val_loader)
        
        NLL.append(avg_training_loss)
        print(
            "LOSS train {:.5f} valid {:.5f}".format(
                avg_training_loss, avg_val_loss
            )
        )
        
        
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        
        # os.makedirs("/content/drive/MyDrive/neural-processes-main/models/convcnp/2d_saved", exist_ok=True)
        model_path = "checkpoint/{}/model_{}".format(timestamp, epoch + 1) 
        torch.save(
             model.state_dict(),
             model_path
        )
        
    plt.plot(NLL)
    plt.xlabel("epoch")
    plt.ylabel("NLL")
    plt.savefig("plots/GridConvCNP_Celeba_NLL")
    
    return model_path

def test_2d(model, a, b,  test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.train(False)
    running_test_loss = 0.0
    masker = GetBatchMask(a=a, b=b)
    with torch.no_grad():
        for idx, (data, _) in enumerate(test_loader):
                
            data=data.to(device)
            mask = masker(data.shape[0], (data.shape[2], data.shape[3]), is_same_mask=True)
            mask = mask.to(device)
            pred_y_test, std_test, test_loss = model(data, mask)
            running_test_loss += test_loss
    
    avg_test_loss = running_test_loss/len(test_loader)
    
    print(
            "LOSS test {:.5f}".format(
                avg_test_loss
            )
        )
    
    return avg_test_loss
