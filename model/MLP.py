from torch import nn

from model.layers.MLP_layer import MLPLayer


class MLPBlock(nn.Module):

    def __init__(self, n_view, d_vec, out, hidden, drop_prob=0.1):
        # d_vec: the max length of views
        # [batch_size, n_view, d_vec] -> [batch_size, n_view, d_model]
        super().__init__()
        self.n_view = n_view
        self.MLPs = MLPLayer(n_view * d_vec, out, hidden, drop_prob=drop_prob)

    def forward(self, x):
        # input x: [batch_size, n_view, d_vec]
        # output x: [batch_size, n_view, d_model]
        batch_size, n_view, d_vec = x.size()
        x = x.contiguous().view(batch_size, n_view * d_vec)  # [batch_size, n_view * d_vec]
        x = self.MLPs(x)
        x = self.split(x)
        return x

    def split(self, tensor):
        """
        split tensor by number of view

        :param tensor: [batch_size, n_view * d_vec]
        :return: [batch_size, n_view, d_model]
        """
        batch_size, length = tensor.size()

        d_model = length // self.n_view

        assert d_model * self.n_view == length
        tensor = tensor.view(batch_size, self.n_view, d_model)

        return tensor
