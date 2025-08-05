import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from Unet_model import ConditionalUNet, create_data_loaders, get_model_info

def train_model(model, train_loader, val_loader, config):
    wandb.init(project="polygon-coloring-unet", config=config, name="polygon_coloring_exp1")
    wandb.log(get_model_info(model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for inputs, colors, targets in train_loader:
            inputs, colors, targets = inputs.to(device), colors.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, colors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, colors, targets in val_loader:
                inputs, colors, targets = inputs.to(device), colors.to(device), targets.to(device)
                outputs = model(inputs, colors)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')

    torch.save(model.state_dict(), 'final_model.pth')
    wandb.save('final_model.pth')
    wandb.finish()
    return model

def visualize_sample_predictions(model, val_loader, color_to_idx, device):
    import matplotlib.pyplot as plt
    import wandb  

    model.eval()
    sample_inputs, sample_colors, sample_targets = next(iter(val_loader))
    sample_inputs, sample_colors = sample_inputs.to(device), sample_colors.to(device)

    with torch.no_grad():
        outputs = model(sample_inputs, sample_colors)

    # Visualize a few samples
    num_samples = min(4, sample_inputs.size(0))
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    for i in range(num_samples):
        axs[i, 0].imshow(sample_inputs[i].squeeze().cpu(), cmap='gray')
        axs[i, 0].set_title("Input")
        axs[i, 1].imshow(sample_targets[i].permute(1, 2, 0).cpu())
        axs[i, 1].set_title("Target")
        axs[i, 2].imshow(outputs[i].permute(1, 2, 0).cpu())
        axs[i, 2].set_title("Prediction")
        for j in range(3):
            axs[i, j].axis('off')

    plt.tight_layout()
    
    
    if wandb.run is not None:
        wandb.log({"sample_predictions": wandb.Image(fig)})
    else:
        print(" wandb.run is None, skipping wandb.log().")

def main():
    config = {'epochs': 30, 'batch_size': 16, 'lr': 1e-3, 'img_size': 256, 'num_workers': 2}
    data_path = '/content/Project/Ayna_ML/dataset_Ayna/dataset'
    train_loader, val_loader, color_to_idx, num_colors = create_data_loaders(
        data_path, config['batch_size'], config['img_size'], config['num_workers'])

    model = ConditionalUNet(n_channels=1, n_classes=3, num_colors=num_colors)
    config.update(get_model_info(model))
    trained_model = train_model(model, train_loader, val_loader, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_sample_predictions(trained_model, val_loader, color_to_idx, device)
    print("Training complete.")

if __name__ == "__main__":
    main()
