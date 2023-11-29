import torch


class Classifier(torch.nn.Module):

    def __init__(self, n_inputs, n_classes, n_hidden=384, dropout=0.0):
        super(Classifier, self).__init__()
        
        self.classifier = torch.nn.Sequential(
            #torch.nn.BatchNorm1d(n_inputs),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(n_inputs, n_hidden), torch.nn.BatchNorm1d(n_hidden), torch.nn.ReLU(), torch.nn.Dropout(p=dropout),
            #torch.nn.Linear(n_hidden, n_hidden), torch.nn.ReLU(), torch.nn.Dropout(p=dropout),
            torch.nn.Linear(n_hidden, n_classes),
            torch.nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.classifier(x)
        return x