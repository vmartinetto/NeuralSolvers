from torch import tanh
from torch.nn import Module

class KSInverseHPM(Module):

    def __init__(self, ef_derivative: Module):
        """
        Constuctor of the single model HPM
        """
        super(KSInverseHPM,self).__init__() # same as super().__init__()
        self.ef_derivative = ef_derivative

    def forward(self, X, Ev, ef):
        """
        represents the second derviative of the wavefunction as a neural net

        input
        =====
        X: [nx2], torch.tensor, [x,ef]
            the points of the grid values and predicted KS potential at each point

        output
        ======

        """
        lap_input = X[:, 0]
        lap_output = self.ef_derivative(lap_input)
        lap_output = lap_output.view(-1)

        f = Ev*ef + .5*lap_output - X[:,1]*ef
        return f

    def cuda(self):
        super(MLP, self).cuda()
        self.lb = self.lb.cuda()
        self.ub = self.ub.cuda()

    def cpu(self):
        super(MLP, self).cpu()
        self.lb = self.lb.cpu()
        self.ub = self.ub.cpu()