import torch
from train import Places365Classifier

def load_trained_model(model_path='beach_mountain_classifier.pth', device='cuda'):
    # Initialize model
    model = Places365Classifier(num_classes=2)
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    # Move model to the appropriate device
    model = model.to(device)
    # Set model to evaluation mode
    model.eval()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_trained_model(device=device)
