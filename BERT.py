# log the model and metrics to MLflow
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("BERT")

# import libraries
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt


# HYPERPARAMETERS
batch_size = 16
num_epochs = 2
learning_rate = 2e-5
dropout_rate = 0.1
max_length = 512
test_size=0.1
weight_decay = 0.002

# register hyperparameters
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("num_epochs", num_epochs)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("dropout_rate", dropout_rate)
mlflow.log_param("max_length", max_length)
mlflow.log_param("test_size", test_size)
mlflow.log_param("weight_decay", weight_decay)

# Set up parameters
bert_model_name = 'bert-base-uncased'
num_classes = 2

# load data
df = pd.read_csv("data/imdb.csv")
texts = df['review'].tolist()
labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]

class TextClassificationDataset(Dataset):
    """Create a PyTorch dataset for text classification."""

    def __init__(self, texts, labels, tokenizer, max_length):
        """Constructor."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    
class BERTClassifier(nn.Module):
    """BERT text classifier."""

    def __init__(self, bert_model_name, num_classes):
        """Constructor."""
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    
def train(model, data_loader, optimizer, scheduler, device):
    """Train the model."""

    # Put the model into training mode.
    model.train()
    
    # counters
    losses = []
    predictions = []
    actual_labels = []
    
    # For each batch of training data...
    for batch in tqdm(data_loader, total=len(data_loader)):

        # Zero the gradients
        optimizer.zero_grad()

        # Load the batch to the GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        actual_labels.extend(labels.cpu().tolist())
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # get and store the predictions
        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().tolist())
        
        # calculate the loss
        loss = nn.CrossEntropyLoss()(outputs, labels)
        losses.append(loss.item())

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        optimizer.step()
        scheduler.step()

    # return the loss
    return (
        accuracy_score(actual_labels, predictions), 
        f1_score(actual_labels, predictions),
        precision_score(actual_labels, predictions),
        recall_score(actual_labels, predictions),
        sum(losses) / len(losses)
    )


def evaluate(model, data_loader, device):
    """Evaluate the model."""
    
    # Put the model into evaluation mode
    model.eval()

    # counters
    losses = []
    predictions = []
    actual_labels = []

    # For each batch of validation data...
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):

            # Load the batch to the GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            actual_labels.extend(labels.cpu().tolist())

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # get and store the predictions
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())

            # calculate the loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            losses.append(loss.item())
            

    return (
        accuracy_score(actual_labels, predictions), 
        f1_score(actual_labels, predictions),
        precision_score(actual_labels, predictions),
        recall_score(actual_labels, predictions),
        sum(losses) / len(losses)
    )

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    """Predict sentiment of text."""
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "positive" if preds.item() == 1 else "negative"

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=test_size, random_state=42)

# Create data loaders
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Create the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

# Create optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Train the model and store losses for plotting
metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': [],
    'train_f1': [],
    'val_f1': [],
    'train_precision': [],
    'val_precision': [],
    'train_recall': [],
    'val_recall': []
}
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train the model
    accuracy, f1, precision, recall, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
    
    # store the metrics
    metrics['train_loss'].append(train_loss)
    metrics['train_accuracy'].append(accuracy)
    metrics['train_f1'].append(f1)
    metrics['train_precision'].append(precision)
    metrics['train_recall'].append(recall)
    
    # evaluate the model
    accuracy, f1, precision, recall, val_loss = evaluate(model, val_dataloader, device)
    
    # store the metrics
    metrics['val_loss'].append(val_loss)
    metrics['val_accuracy'].append(accuracy)
    metrics['val_f1'].append(f1)
    metrics['val_precision'].append(precision)
    metrics['val_recall'].append(recall)

    # prints
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1: {f1:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")

# register metrics
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1", f1)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)

# log the model
mlflow.pytorch.log_model(model, "model")

# plot the losses
plt.plot(metrics['train_loss'], label='train_loss')
plt.plot(metrics['val_loss'], label='val_loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BERT")
plt.legend()
plt.savefig("data/BERT_loss.png")
plt.close()

# plot other metrics
plt.plot(metrics['train_accuracy'], label='train_accuracy')
plt.plot(metrics['val_accuracy'], label='val_accuracy')
plt.plot(metrics['train_f1'], label='train_f1')
plt.plot(metrics['val_f1'], label='val_f1')
plt.plot(metrics['train_precision'], label='train_precision')
plt.plot(metrics['val_precision'], label='val_precision')
plt.plot(metrics['train_recall'], label='train_recall')
plt.plot(metrics['val_recall'], label='val_recall')
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("BERT")
plt.legend()
plt.savefig("data/BERT_metrics.png")
plt.close()