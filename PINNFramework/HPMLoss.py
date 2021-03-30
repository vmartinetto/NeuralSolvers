from .PDELoss import PDELoss

class HPMLoss(PDELoss):
    def __init__(self, dataset, hpm_input, hpm_model, norm='L2', weight=1.):
        """
        Constructor of the HPM loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            hpm_input(function): function that calculates the needed input for the HPM model. The hpm_input function
            should return a list of tensors, where the last entry is the time_derivative
            hpm_model (torch.nn.Module): model for the HPM, represents the underlying PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(HPMLoss, self).__init__(dataset, None, norm, weight)
        self.hpm_input = hpm_input
        self.hpm_model = hpm_model

    def __call__(self, x, model, **kwargs):
        """
        Calculation of the HPM Loss

        applies MSE Loss between the time derivative and the HPM model output


        Args:
            x(torch.Tensor): residual points
            model(torch.nn.module): model representing the solution
        """
        x.requires_grad = True
        prediction_u = model(x)
        hpm_input = self.hpm_input(x, prediction_u)
        time_derivative = hpm_input[:, -1]
        input = hpm_input[:, :-1]
        hpm_output = self.hpm_model(input)
        return self.weight * self.norm(time_derivative, hpm_output)

class INV_HPMLoss(PDELoss):
    def __init__(self, dataset, hpm_input, hpm_model, norm='L2', weight=1.):
        """
        Constructor of the HPM loss

        Args:
            dataset (torch.utils.Dataset): dataset that provides the residual points
            hpm_input(function): function that calculates the needed input for the HPM model. The hpm_input function
            should return a list of tensors, where the last entry is the time_derivative
            hpm_model (torch.nn.Module): model for the HPM, represents the underlying PDE
            norm: Norm used for calculation PDE loss
            weight: Weighting for the loss term
        """
        super(INV_HPMLoss, self).__init__(dataset, None, norm, weight)
        self.hpm_input = hpm_input
        self.hpm_model = hpm_model

    def __call__(self, data, model, **kwargs):
        """
        Calculation of the HPM Loss

        applies MSE Loss between the time derivative and the HPM model output


        Args:
            data(torch.tensor): contains residual points, eigenvalue, and groundstate eigenfunction
            x(torch.Tensor): residual points
            model(torch.nn.module): model representing the solution
        """

        x = data[:, 1]
        Ev = data[0,0]
        ef = data[:,2]
        x.requires_grad = True
        prediction_u = model(x)
        hpm_input = self.hpm_input(x, prediction_u)
        time_derivative = hpm_input[:, -1]
        input = hpm_input[:, :-1]
        hpm_output = self.hpm_model(input, Ev, ef)
        return self.weight * self.norm(time_derivative, hpm_output)