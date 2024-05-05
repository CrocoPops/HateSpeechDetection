import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./logs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, epochs, lr, batch_size, train_ds, test_ds, logger_suffix):
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)
    print(f"Starting training: epochs - {epochs}, lr - {lr}, batch size - {batch_size}")
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_correct_train = 0
        total_samples_train = 0
        for batch in train_dl:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            writer.add_scalar(f"Loss/train_{logger_suffix}", loss, epoch)

            total_loss += loss.item()

            _, predicted_train = torch.max(outputs.logits, 1)
            total_samples_train += labels.size(0)
            total_correct_train += (predicted_train == labels).sum().item()

            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}: Average Loss: {total_loss / len(train_dl)}')
        all_loss['train_loss'].append(total_loss / len(train_dl))
        all_acc['train_acc'].append(total_correct_train / total_samples_train)

        # TESTING PHASE
        model.eval()
        total_val_loss = 0
        total_correct_val = 0
        total_samples_val = 0
        with torch.no_grad():
            for batch in test_dl:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss

                total_val_loss += val_loss.item()

                _, predicted_val = torch.max(outputs.logits, 1)
                total_samples_val += labels.size(0)
                total_correct_val += (predicted_val == labels).sum().item()
        
        all_loss['val_loss'].append(total_val_loss / len(test_dl))
        all_acc['val_acc'].append(total_correct_val / total_samples_val)

        # Write to TensorBoard
        writer.add_scalar(f"Loss/val_{logger_suffix}", total_val_loss / len(test_dl), epoch)
        writer.flush()

    print(f"Training complete: epochs - {epochs}, lr - {lr}, batch size - {batch_size}")
    return model, all_loss, all_acc

def test(model, batch_size, test_dl):
    model.eval()
    predicted_labels = []
    true_labels = []

    test_dl = DataLoader(test_dl, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_dl:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_labels.extend(torch.argmax(logits, axis=1).tolist()) #Extract the max logits value and convert from pytorch to a pyList
            true_labels.extend(labels.tolist()) 
    return true_labels, predicted_labels

def predict_nn(model, tokenizer, sentence):
    input_ids_gpt2 = tokenizer.encode(sentence, return_tensors='pt').to(DEVICE)
    attention_mask_gpt2 = torch.ones_like(input_ids_gpt2).to(DEVICE)

    # Generate output for GPT-2
    with torch.no_grad():
        outputs = model(input_ids_gpt2, attention_mask=attention_mask_gpt2)
        predicted_probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return { "predicted_class" : predicted_class, "prediction_probabilities" : predicted_probs.tolist()[0]}

def predict_lr(model, vectorized_sentence):
    
    predicted_class = model.predict(vectorized_sentence)[0]
    predicted_probs = model.predict_proba(vectorized_sentence)

    return { "predicted_class" : predicted_class, "prediction_probabilities" : predicted_probs[0] }
    