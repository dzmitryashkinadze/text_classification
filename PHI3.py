import numpy as np
import pandas as pd
from transformers import BitsAndBytesConfig

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

import os
import time
import zipfile
import urllib.request
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import lightning as L

tqdm.pandas()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
# extracted_path = "sms_spam_collection"
# data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


# def download_and_unzip(url, zip_path, extracted_path, data_file_path):
#     if data_file_path.exists():
#         print(f"{data_file_path} already exists. Skipping download and extraction.")
#         return

#     # Downloading the file
#     with urllib.request.urlopen(url) as response:
#         with open(zip_path, "wb") as out_file:
#             out_file.write(response.read())

#     # Unzipping the file
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extracted_path)

#     # Add .tsv file extension
#     original_file_path = Path(extracted_path) / "SMSSpamCollection"
#     os.rename(original_file_path, data_file_path)
#     print(f"File downloaded and saved as {data_file_path}")


# download_and_unzip(url, zip_path, extracted_path, data_file_path)

# df = pd.read_csv(data_file_path, sep="\t", header=None, names=["target", "text"])


# def create_balanced_dataset(df):

#     # Count the instances of "spam"
#     num_spam = df[df["target"] == "spam"].shape[0]

#     # Randomly sample "ham" instances to match the number of "spam" instances
#     ham_subset = df[df["target"] == "ham"].sample(num_spam, random_state=123)

#     # Combine ham "subset" with "spam"
#     balanced_df = pd.concat([ham_subset, df[df["target"] == "spam"]])

#     return balanced_df


# balanced_df = create_balanced_dataset(df)
# print(balanced_df["target"].value_counts())

# balanced_df["target"] = df.target.map({"spam": 1, "ham": 0})


# def random_split(df, train_frac, validation_frac):
#     # Shuffle the entire DataFrame
#     df = df.sample(frac=1, random_state=123).reset_index(drop=True)

#     # Calculate split indices
#     train_end = int(len(df) * train_frac)
#     validation_end = train_end + int(len(df) * validation_frac)

#     # Split the DataFrame
#     train_df = df[:train_end]
#     validation_df = df[train_end:validation_end]
#     test_df = df[validation_end:]

#     return train_df, validation_df, test_df


# train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
# # Test size is implied to be 0.2 as the remainder

# train_df.to_csv("train.csv", index=None)
# validation_df.to_csv("validation.csv", index=None)
# test_df.to_csv("test.csv", index=None)

# Initialize tokenizer with model ID and authentication token
model_id = "microsoft/Phi-3-mini-4k-instruct"
hf_token = "hf_"  # Replace your token here on huggingface

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

# Set padding token to end-of-sequence token and configure padding side
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

train = pd.read_csv("data/train_500.csv")
# test = pd.read_csv("test.csv")
# val = pd.read_csv("validation.csv")

# follow the phi 3 on huggingface:
# "Some applications/frameworks might not include a BOS token (<s>) at the start of the conversation.
# Please ensure that it is included since it provides more reliable results."

train["text"] = tokenizer.bos_token + train["text"]
# test["text"] = tokenizer.bos_token + test["text"]
# val["text"] = tokenizer.bos_token + val["text"]

# sample = tokenizer(train.text[0], add_special_tokens=False).input_ids
# tokenizer.decode(sample)


class CustomDataset(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        return text, target

    def __len__(self):
        return len(self.targets)


# Set seed for reproducibility
L.seed_everything(seed=252)

# Create train dataset and dataloader
train_dataset = CustomDataset(
    texts=train["text"].values.tolist(), targets=train["target"].values.tolist()
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=4, shuffle=True, drop_last=True
)

# # Create test dataset and dataloader
# test_dataset = CustomDataset(
#     texts=test["text"].values.tolist(), targets=test["target"].values.tolist()
# )
# test_dataloader = DataLoader(
#     dataset=test_dataset, batch_size=16, shuffle=False, drop_last=False
# )

# # Create validation dataset and dataloader
# val_dataset = CustomDataset(
#     texts=val["text"].values.tolist(), targets=val["target"].values.tolist()
# )
# val_dataloader = DataLoader(
#     dataset=val_dataset, batch_size=16, shuffle=False, drop_last=False
# )


def tokenize_text(text):
    """
    Tokenize the text and return PyTorch tensors with dynamic padding
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",  # Dynamically pad each batch to the length of the longest sequence
        add_special_tokens=False,
    )

    return encodings


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


# Define the configuration for BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# Define a neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Load configuration from a pre-trained model
        config = AutoConfig.from_pretrained(model_id)

        # Load pre-trained language model with specific configurations
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=bnb_config,
            token=hf_token,
            # attn_implementation="flash_attention_2",
        )

        # Replace language model head with an identity function
        self.llm.lm_head = nn.Identity()

        # Freeze all parameters of the language model backbone
        for name, param in self.llm.named_parameters():
            param.requires_grad = False

        # Define classification head
        self.cls_head = nn.Sequential(
            nn.Linear(config.hidden_size, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, 2),
        )

    # Define the forward pass
    def forward(self, input_ids, attention_mask):
        x = self.llm(input_ids, attention_mask).logits  # get last hidden state
        logits = self.cls_head(x)[
            :, -1, :
        ]  # Apply classification head to the last token's output
        return logits


def get_optimizer(model, learning_rate=0.0001, diff_lr=0.00001, weight_decay=0.01):
    """
    Get optimizer with different learning rates for specified layers.

    Args:
        model (torch.nn.Module): The neural network model.
        learning_rate (float): Learning rate for non-differential layers.
        diff_lr (float): Learning rate for differential layers.
        weight_decay (float): Weight decay (decoupled from L2 penalty) for optimizer.

    Returns:
        torch.optim.AdamW: Optimizer for the model.
    """

    # Define parameters with different learning rates and weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = ["llm"]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": diff_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": diff_lr,
                "weight_decay": 0,
            },
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer


num_epochs = 1
learning_rate = 0.0002
diff_lr = 0.00001  # not being used because I freeze the llm backbone
warmup_steps = 0
seed = 252
weight_decay = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set seed for reproducibility
L.seed_everything(seed=seed)

# Instantiate the neural network model
model = Net()
model.to(device)  # Move model to the device

# Display the names of trainable parameters
print("Here are the trainable parameters:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print(n)

# Get the optimizer
optimizer = get_optimizer(
    model, learning_rate=learning_rate, diff_lr=diff_lr, weight_decay=weight_decay
)

# Set up the scheduler for learning rate adjustment
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_epochs * len(train_dataloader),
)

scaler = GradScaler()

start_time = time.time()
for epoch in range(num_epochs):

    for batch_idx, batch in enumerate(train_dataloader):

        model.train()

        prompt, targets = batch

        encodings = tokenize_text(prompt)

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        targets = targets.to(device)

        # Perform forward pass with autocast for mixed precision training
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, targets)

        # Backward pass, optimization step, and learning rate adjustment
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Logging training progress
        print(
            f"Epoch: {epoch+1} / {num_epochs}"
            f"| Batch: {batch_idx+1}/{len(train_dataloader)}"
            f"| Loss: {loss.item():.4f}"
        )

end_time = time.time()
training_time = (end_time - start_time) / 60
print(f"Total training time: {training_time:.2f} min")


# def calc_accuracy(dataloader, type):
#     with torch.no_grad():
#         model.eval()
#         pred_scores = []
#         actual_scores = []
#         for batch in tqdm(
#             dataloader, total=len(dataloader), desc=f"Calc {type} accuracy"
#         ):
#             prompt, targets = batch
#             encodings = tokenize_text(prompt)

#             input_ids = encodings["input_ids"].to(device)
#             attention_mask = encodings["attention_mask"].to(device)

#             with autocast():
#                 logits = model(input_ids, attention_mask)
#                 pred_score = (
#                     F.softmax(logits, dim=-1)
#                     .argmax(dim=-1)
#                     .cpu()
#                     .detach()
#                     .numpy()
#                     .tolist()
#                 )
#                 pred_scores.extend(pred_score)
#                 actual_scores.extend(targets.numpy().tolist())

#         pred_scores = np.array(pred_scores)
#         accuracy = accuracy_score(actual_scores, pred_scores)

#         return accuracy


# train_acc = calc_accuracy(train_dataloader, type="train")
# test_acc = calc_accuracy(test_dataloader, type="test")
# val_acc = calc_accuracy(val_dataloader, type="val")

# print("Train accuracy:", train_acc)
# print("Test accuracy:", test_acc)
# print("Val accuracy:", val_acc)
