import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from prop_pred.models import MLP, ResNet, WRN
from data.get_data_loaders import get_data_loaders

def get_model(model_type, input_dim, depth, output_dim, p=0.1):
    if model_type == 'mlp':
        return MLP(input_dim, depth, output_dim, p)
    elif model_type == 'resnet':
        return ResNet(input_dim, depth, output_dim, p)
    elif model_type == 'wrn':
        return WRN(input_dim, depth, output_dim, p)
    else:
        raise ValueError('Model type not recognized')
    
def train_model(model, train_loader, val_loader, num_epochs, prop_id, device='cuda'):
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs)/3, gamma=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y[:, prop_id].to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = F.mse_loss(y_pred.squeeze(-1), y.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_loss = eval_model(model, val_loader, device)
        print(f'Epoch {epoch}, Train Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss}')
        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)
    return train_losses, val_losses

def eval_model(model, data_loader, prop_id, device, return_correlation=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        if return_correlation:
            y_true = []
            y_pred = []

        for batch in data_loader:
            x, y = batch
            x, y = x.to(device), y[:, prop_id].to(device)
            y_pred = model(x)
            loss = F.mse_loss(y_pred.squeeze(-1), y.squeeze(-1))
            total_loss += loss.item()
            if return_correlation:
                y_true.append(y.cpu().numpy())
                y_pred.append(y_pred.cpu().numpy())
        if return_correlation:
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            return total_loss / len(data_loader), np.corrcoef(y_true, y_pred, rowvar=False)[0, 1]
    return total_loss / len(data_loader)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of model to use')
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument('--depth', type=int, help='Depth of the model')
    parser.add_argument('--p', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on')
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to data')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--log_file', type=str, default='log.txt', help='File to save logs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--prop_id', type=int, default=0, help='Index of property to train on')
    
    args = parser.parse_args()
    
    print('Get model')
    model = get_model(args.model_type, args.input_dim, args.depth, 1, args.p)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('Load data')
    train_loader, val_loader, test_loader = get_data_loaders(args.data_path, args.batch_size, shuffle_train=True, validation_ratio=0.2)

    print('Train model')
    train_losses, val_losses = train_model(model, train_loader, args.num_epochs, args.log_dir, args.device)
    torch.save(model, os.path.join(args.save_path, 'model.pt'))
    np.save(os.path.join(args.save_path, 'train_losses.list'), train_losses)
    np.save(os.path.join(args.save_path, 'val_losses.list'), val_losses)

    print('Test model')
    train_loss, train_corr = eval_model(model, train_loader, args.device, return_correlation=True)
    val_loss, val_corr = eval_model(model, val_loader, args.device, return_correlation=True)
    test_loss, test_corr = eval_model(model, test_loader, args.device, return_correlation=True)

    with open(args.log_file, 'a') as f:
        model_id = f'{args.model_type}_{args.input_dim}_{args.depth}_{1}_{args.p}'
        f.write(f'{model_id}, {train_loss}, {val_loss}, {test_loss}, {train_corr}, {val_corr}, {test_corr}\n')