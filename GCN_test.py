import torch
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)
        self.conv3 = GCNConv(in_channels, out_channels)
        self.conv4 = torch.nn.Linear(10, 10)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        y = self.conv4(x[:, :10])
        print(y.shape)
        x[:, :10] = x[:, :10] + y
        x_feats = self.conv2(x, edge_index)
        x_probas = self.conv3(x_feats, edge_index)
        return x_feats, x_probas

if __name__ == '__main__':

    model = GCN(768, 768, 3).cuda()
    model = model
    x_train = torch.randn((961, 768)).cuda()
    x_test = torch.randn((961, 768)).cuda()
    for i in range(1):
        d = torch.sqrt(torch.mean((x_train - x_test) ** 2, dim=1))
    x_A = torch.abs(torch.randn((40, 40)))
    print(x_A.shape)

    x_edges = [[i, j] for i in range(x_A.shape[0]) for j in range(x_A.shape[1]) if x_A[i, j] > 2]
    x_edges = torch.tensor(x_edges).permute(1, 0)
    x_edges = x_edges.cuda()
    #print(torch.max(x_A), torch.min(x_A))
    #print(x_edges)
    x_B = torch.rand((50, 10))
    x_mean = torch.sum(x_B, dim=0, keepdim=True)
    print(x_mean.shape)

    y_pre = model(x_train, x_edges)
    #print(y_pre[0].shape, y_pre[1].shape)
    #print(d.shape)





