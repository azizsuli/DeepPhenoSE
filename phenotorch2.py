import os
import click as ck
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import math
import time
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from utils import Ontology
from aminoacids import MAXLEN, to_onehot

logging.basicConfig(level=logging.DEBUG)
print("GPU Available: ", torch.cuda.is_available())


class HPOLayer(nn.Module):
    def __init__(self, nb_classes):
        super(HPOLayer, self).__init__()
        self.nb_classes = nb_classes
        self.hpo_matrix = nn.Parameter(
            torch.zeros((nb_classes, nb_classes), dtype=torch.float32), requires_grad=False)

    def set_hpo_matrix(self, hpo_matrix):
        self.hpo_matrix.data = torch.tensor(hpo_matrix, dtype=torch.float32)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.nb_classes, 1)
        return x * self.hpo_matrix


class FlatModel(nn.Module):
    def __init__(self, input_shape, exp_shape, nb_classes):
        super(FlatModel, self).__init__()
        self.dense_1 = nn.Linear(input_shape, 1500)
        self.dense_out = nn.Linear(1500 + exp_shape, nb_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, gos, exp_x):
        x = F.gelu(self.dense_1(gos))
        x = self.dropout(x)
        x = torch.cat((x, exp_x), dim=1)
        out = torch.sigmoid(self.dense_out(x))
        return out


class DeepPhenoModel(nn.Module):
    def __init__(self, input_shape, exp_shape, nb_classes, hpo_matrix):
        super(DeepPhenoModel, self).__init__()
        self.flat_model = FlatModel(input_shape, exp_shape, nb_classes)
        self.hpo_layer = HPOLayer(nb_classes)
        self.hpo_layer.set_hpo_matrix(hpo_matrix)
        self.pool = nn.MaxPool1d(kernel_size=nb_classes)
        self.flatten = nn.Flatten()

    def forward(self, gos, exp_x):
        flat_out = self.flat_model(gos, exp_x)
        hpo_out = self.hpo_layer(flat_out)
        hpo_out = hpo_out.permute(0, 2, 1)
        pooled_out = self.pool(hpo_out)
        output = self.flatten(pooled_out).view(-1, self.hpo_layer.nb_classes)
        return output


class GeneDataset(Dataset):
    def __init__(self, df, gos_dict, terms_dict, expression_df):
        self.df = df
        self.gos_dict = gos_dict
        self.terms_dict = terms_dict
        self.expression_df = expression_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data_gos = np.zeros(len(self.gos_dict), dtype=np.float32)

        data_seq = to_onehot(row['sequences'])
        gene_id = row['genes']
        matching_expression = self.expression_df[self.expression_df['Gene ID'] == gene_id]

        if not matching_expression.empty:
            data_exp = matching_expression.iloc[:, 2:].values.flatten()
        else:
            data_exp = np.zeros(53, dtype=np.float32)

        labels = np.zeros(len(self.terms_dict), dtype=np.int32)

        for item in row['deepgo_annotations']:
            t_id, score = item.split('|')
            if t_id in self.gos_dict:
                data_gos[self.gos_dict[t_id]] = float(score)

        for t_id in row['hp_annotations']:
            if t_id in self.terms_dict:
                labels[self.terms_dict[t_id]] = 1
        
        return torch.FloatTensor(data_gos), torch.FloatTensor(data_exp), torch.FloatTensor(labels)


def get_data_loaders(train_df, valid_df, test_df, gos_dict, terms_dict, expression_df, batch_size):
    train_dataset = GeneDataset(train_df, gos_dict, terms_dict, expression_df)
    valid_dataset = GeneDataset(valid_df, gos_dict, terms_dict, expression_df)
    test_dataset = GeneDataset(test_df, gos_dict, terms_dict, expression_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def train(model, train_loader, valid_loader, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCELoss()
    model = model.to(device)

    best_loss = 1000000
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for gos, exp, labels in train_loader:
            gos, exp, labels = gos.to(device), exp.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(gos, exp)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        preds, loss = evaluate(model, valid_loader, device)
        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), 'data/model.th')

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0
    loss_fn = nn.BCELoss()
    with torch.no_grad():
        for gos, exp, labels in data_loader:
            gos, exp, labels = gos.to(device), exp.to(device), labels.to(device)
            outputs = model(gos, exp)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    ##print(f"Number of predictions: {all_preds.shape}")#####
    ##print(f"Number of labels: {all_labels.shape}")#####
    
    try:
        roc_auc = roc_auc_score(all_labels.flatten(), all_preds.flatten())
    except ValueError as e:
        print(f"Skipping ROC AUC: {str(e)}")
        roc_auc = None

    mcc = matthews_corrcoef(all_labels.flatten(), all_preds.flatten() > 0.5)
    print(f'ROC AUC: {roc_auc if roc_auc is not None else "N/A"}, MCC: {mcc:.3f}')
    return all_preds, running_loss

def get_hpo_matrix(hpo, terms_dict):
    nb_classes = len(terms_dict)
    res = np.zeros((nb_classes, nb_classes), dtype=np.float32)
    for hp_id, i in terms_dict.items():
        subs = hpo.get_term_set(hp_id)
        res[i, i] = 1
        for h_id in subs:
            if h_id in terms_dict:
                res[i, terms_dict[h_id]] = 1
    return res


@ck.command()
@ck.option('--hp-file', '-hf', default='data/hp.obo', help='Human Phenotype Ontology file in OBO Format')
@ck.option('--data-file', '-df', default='data/human.pkl', help='Data file with sequences and annotations')
@ck.option('--terms-file', '-tf', default='data/terms.pkl', help='Terms file')
@ck.option('--gos-file', '-gf', default='data/gos.pkl', help='GO classes file')
@ck.option('--model-file', '-mf', default='data/model.th', help='Model file')
@ck.option('--out-file', '-o', default='data/predictions.pkl', help='Output predictions file')
@ck.option('--fold', '-f', default=1, help='Fold index')
@ck.option('--batch-size', '-bs', default=32, help='Batch size')
@ck.option('--epochs', '-e', default=100, help='Training epochs')
@ck.option('--load', '-ld', is_flag=True, help='Load pre-trained model?')
@ck.option('--device', '-d', default='cpu', help='Device to use (e.g., "cpu" or "cuda:0")')
def main(hp_file, data_file, terms_file, gos_file, model_file, out_file, fold, batch_size, epochs, load, device):
    gos_df = pd.read_pickle(gos_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}

    hpo = Ontology(hp_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    expression_df = pd.read_csv('data/E-MTAB-5214-query-results.tpms.tsv', sep='\t', comment='#')

    train_df, valid_df, test_df = load_data(data_file, terms_dict, fold)
    hpo_matrix = get_hpo_matrix(hpo, terms_dict)

    train_loader, valid_loader, test_loader = get_data_loaders(
        train_df, valid_df, test_df, gos_dict, terms_dict, expression_df, batch_size)

    model = DeepPhenoModel(len(gos), 53, len(terms), hpo_matrix)
    device = torch.device(device)

    if not load:## from here 
        train(model, train_loader, valid_loader, epochs, device)
    model.load_state_dict(torch.load(model_file))
    preds, _ = evaluate(model, test_loader, device)############
    test_df['preds'] = list(preds)#######
    print('Saving predictions')
    test_df.to_pickle(out_file)
    torch.save(model.state_dict(), model_file)##to here 




def load_data(data_file, terms_dict, fold=1):
    df = pd.read_pickle(data_file)
    n = len(df)
    index = np.arange(n)
    np.random.seed(10)
    np.random.shuffle(index)
    fn = n // 5

    train_index = []
    test_index = []

    for i in range(1, 6):
        start, end = (i - 1) * fn, i * fn
        if i == fold:
            test_index.extend(index[start:end])
        else:
            train_index.extend(index[start:end])

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    valid_df = train_df.iloc[int(len(train_df) * 0.9):]
    train_df = train_df.iloc[:int(len(train_df) * 0.9)]

    # **Debug: Print class distributions in splits**
    #print(f"Train class distribution: {train_df['hp_annotations'].explode().value_counts().to_dict()}")
    #print(f"Validation class distribution: {valid_df['hp_annotations'].explode().value_counts().to_dict()}")
    #print(f"Test class distribution: {test_df['hp_annotations'].explode().value_counts().to_dict()}")

    # **Ensure at least two classes in each split**
    assert len(np.unique(train_df['hp_annotations'].explode())) >= 2, "Train split has only one class!"
    assert len(np.unique(valid_df['hp_annotations'].explode())) >= 2, "Validation split has only one class!"
    assert len(np.unique(test_df['hp_annotations'].explode())) >= 2, "Test split has only one class!"

    return train_df, valid_df, test_df



if __name__ == '__main__':
    main()
