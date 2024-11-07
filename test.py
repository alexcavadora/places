import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from train import Places365Classifier
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def evaluate_model(model_path, test_dir, device):
    # Load model
    model = Places365Classifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    true_labels = []
    predicted_labels = []
    confidences = []
    
    # Process beach images
    beach_dir = os.path.join(test_dir, 'beach')
    for img_name in os.listdir(beach_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(beach_dir, img_name)
            pred_class, confidence = predict_image(model, img_path, device)
            true_labels.append(0)
            predicted_labels.append(pred_class)
            confidences.append(confidence)
    
    # Process mountain images
    mountain_dir = os.path.join(test_dir, 'mountain')
    for img_name in os.listdir(mountain_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(mountain_dir, img_name)
            pred_class, confidence = predict_image(model, img_path, device)
            true_labels.append(1)
            predicted_labels.append(pred_class)
            confidences.append(confidence)
    
    return true_labels, predicted_labels, confidences

def plot_evaluation_results(true_labels, predicted_labels, confidences):
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, 
                              target_names=['Beach', 'Mountain']))
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.savefig('confidence_distribution.png')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'beach_mountain_classifier.pth'
    test_dir = 'img'
    
    true_labels, predicted_labels, confidences = evaluate_model(model_path, test_dir, device)
    plot_evaluation_results(true_labels, predicted_labels, confidences)

if __name__ == '__main__':
    main()