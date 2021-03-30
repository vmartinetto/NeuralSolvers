import sys
import numpy as np
from argparse import ArgumentParser
from datasets import TI_KS_INV_1D
import torch
from torch.utils.data import Dataset
sys.path.append('../../')
import PINNFramework as pf

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--idnetifier", type=str, default="KS_Inv_DeepHPM")
    parser.add_argument("--path_data", type=str, default="./data/")
    parser.add_argument("--epochs", type=int, default=40000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_hidden", type=int, default=8)
    parser.add_argument("--hidden_size_lap", type=int, default=500)
    parser.add_argument("--num_hidden_lap", type=int, default=8)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--use_horovod", type=int, default=0)
    args = parser.parse_args()

    pde_dataset = np.loadtxt('harmonic_oscillator_grnd.txt')
    pde_dataset = torch.from_numpy(pde_dataset[:,:-1])
    low_bound = pde_dataset[0,1].float()
    up_bound = pde_dataset[-1,1].float()

    # PINN model
    # same as the PINN model in HZDR pinn-dft repo train_PINN_Inv
    # takes the grid points of x and predicts the Kohn-Sham potential at each point
    model = pf.models.MLP(input_size=1,
                          output_size=1,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=low_bound,
                          ub=up_bound)

    # eigenfunctions second derivative model
    # takes the gridpoint of x and predicts the second derivative of the eigenfunction
    lap_net = pf.models.MLP(input_size=1,
                            output_size=1,
                            hidden_size=args.hidden_size_lap,
                            num_hidden=args.num_hidden_lap,
                            lb=low_bound,
                            ub=up_bound)

    lap_net.cuda()
    hpm_model = pf.models.KSInverseHPM(lap_net)
    hpm_loss = pf.HPMLoss.INV_HPMLoss(pde_dataset, TI_KS_INV_1D, hpm_model)
    pinn = pf.PINN(
        model,
        input_dimension=1, # x
        output_dimension=1, # vks
        pde_loss=hpm_loss,
        initial_condition=None,
        boundary_condition=None,
        use_gpu=args.use_gpu,
        use_horovod=False,
        use_wandb=True,
        project_name='KS_inversion')