import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from data.gp_dataloader import GPDataGenerator
from utils import plot_np_results


# Set the random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

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
