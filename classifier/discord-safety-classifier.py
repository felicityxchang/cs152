import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DiscordDataset(Dataset):
    """Dataset class for Discord messages"""
    def __init__(self, messages, labels, tokenizer, max_length=128):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, idx):
        message = str(self.messages[idx])
        label = self.labels[idx]
        
        # Tokenize the message
        encoding = self.tokenizer(
            message,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    """
    Load data from a CSV or JSON file
    Expected format: columns 'message' and 'label' (0 or 1)
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("File must be CSV or JSON")
    
    messages = df['message'].tolist()
    labels = df['label'].tolist()
    
    return messages, labels

def create_synthetic_data():
    """
    Create synthetic training data for demonstration
    In real use, replace with actual labeled data
    """
    # This is synthetic data for demonstration only
    safe_messages = [
        "Hey, want to play some games tonight?",
        "Just finished my homework, feeling good!",
        "Anyone up for voice chat?",
        "Check out this cool meme I found",
        "GG everyone, that was a fun match",
        "Happy birthday! Hope you have a great day",
        "Thanks for the help with that quest",
        "Streaming later if anyone wants to watch",
        "Pizza or burgers for dinner?",
        "New episode just dropped, no spoilers!",
        "Study group at 7pm, who's in?",
        "Just got a new highscore!",
        "Weekend plans anyone?",
        "Movie recommendations?",
        "Thanks for being such a good friend"
    ]
    
    # NOTE: These are synthetic examples for academic purposes only
    concerning_messages = [
        "Everything feels hopeless right now",
        "I don't want to be here anymore",
        "Nobody would care if I disappeared",
        "I've been thinking about ending it all",
        "Life isn't worth living",
        "I can't take this pain anymore",
        "Planning to hurt myself tonight",
        "Goodbye everyone, this is my last message",
        "I don't see a point in continuing",
        "The world would be better without me",
        "I've made up my mind about ending things",
        "No one understands how much this hurts",
        "I'm done trying to fight this",
        "This is my final decision",
        "I won't be around much longer"
    ]
    
    # Create balanced dataset
    messages = safe_messages + concerning_messages
    labels = [0] * len(safe_messages) + [1] * len(concerning_messages)
    
    return messages, labels

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def main():
    # Configuration
    config = {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_steps': 100,
        'output_dir': './discord_safety_model'
    }
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    model = DistilBertForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=2
    )
    model.to(device)
    
    # Load or create data
    print("Loading data...")
    try:
        # Try to load from file
        messages, labels = load_data('discord_messages.csv')
        print(f"Loaded {len(messages)} messages from file")
    except:
        # Use synthetic data for demonstration
        print("Creating synthetic data for demonstration...")
        messages, labels = create_synthetic_data()
        print(f"Created {len(messages)} synthetic examples")
    
    # Split data
    train_messages, val_messages, train_labels, val_labels = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = DiscordDataset(train_messages, train_labels, tokenizer, config['max_length'])
    val_dataset = DiscordDataset(val_messages, val_labels, tokenizer, config['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    best_f1 = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        val_results = evaluate(model, val_loader, device)
        print(f"Validation metrics:")
        print(f"  Accuracy: {val_results['accuracy']:.4f}")
        print(f"  Precision: {val_results['precision']:.4f}")
        print(f"  Recall: {val_results['recall']:.4f}")
        print(f"  F1: {val_results['f1']:.4f}")
        
        # Save best model
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            if not os.path.exists(config['output_dir']):
                os.makedirs(config['output_dir'])
            model.save_pretrained(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
            print(f"Saved best model with F1: {best_f1:.4f}")
    
    print("\nTraining complete!")
    
    # Print confusion matrix
    cm = confusion_matrix(val_results['true_labels'], val_results['predictions'])
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

class DiscordSafetyClassifier:
    """Easy-to-use classifier class"""
    def __init__(self, model_path='./discord_safety_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, message):
        """
        Predict if a message is concerning (1) or safe (0)
        Returns: (prediction, confidence)
        """
        encoding = self.tokenizer(
            message,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return prediction, confidence

def demo_inference():
    """Demo of using the trained model"""
    print("\n" + "="*50)
    print("INFERENCE DEMO")
    print("="*50)
    
    # Load the trained model
    classifier = DiscordSafetyClassifier('./discord_safety_model')
    
    # Test messages
    test_messages = [
        "Want to play Minecraft later?",
        "I'm feeling really down and hopeless",
        "Pizza party this weekend!",
        "Thanks for being a great friend",
        "I don't think I can go on anymore"
    ]
    
    print("\nTesting classifier on sample messages:")
    for msg in test_messages:
        pred, conf = classifier.predict(msg)
        status = "CONCERNING" if pred == 1 else "SAFE"
        print(f"\nMessage: '{msg}'")
        print(f"Prediction: {status} (confidence: {conf:.3f})")

if __name__ == "__main__":
    # Train the model
    main()
    
    # Demo inference
    try:
        demo_inference()
    except:
        print("\nModel not found for demo. Train the model first.")
    
    print("\n" + "="*50)
    print("IMPORTANT NOTES:")
    print("="*50)
    print("1. This is for educational purposes only")
    print("2. Real deployment requires professional oversight")
    print("3. Always include human review for concerning content")
    print("4. Provide mental health resources in any implementation")
    print("5. Regular bias auditing is essential"
