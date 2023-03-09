import torch
from data.gp_dataloader import GPDataGenerator
from models.cnp.cnp_model import CNPModel
from models.anp import prepare_plot
from utils import alt_plot_np_results

if __name__ == "__main__":

    model = CNPModel()
    test_gen = GPDataGenerator(batch_size=1, max_n_context=10, testing=True)

    root = "/home/lb953/rds/hpc-work/MLMI4/"
    model_path = "cnp_1d_128hidden/model_200000.pt"

    checkpoint = torch.load(root+model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cpu")
    model.eval()
    context_x, context_y, target_x, target_y = test_gen.generate_batch().get_all()
    pred_y, std, loss, _, _ = model(context_x, context_y, target_x, target_y)

    alt_plot_np_results(*prepare_plot([target_x, target_y, context_x, context_y, pred_y, std]), title=f"Model: {model_path}",
                    filename=f"{model_path}_test.png"
                )
