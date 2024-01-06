import torch
from seqeval.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
import numpy as np
import config
from data.dataset import train_dataloader, valid_dataloader, tag_values
import Model

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    Model.optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
model = Model.model.to(config.device)

# storing loss values
loss_values, validation_loss_values = [], []

# TRAINING AND VALIDATION

for _ in trange(epochs, desc="Epoch"):
    # /|\==>TRAINLOOP(ONEPASS)<==\|/
    model.train()
    total_loss = 0  # so it resets each epoch

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(config.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch  # also the order in train_data/val_data
        b_input_ids = b_input_ids.type(torch.long)
        b_labels = b_labels.type(torch.long)
        model.zero_grad()  # clearing previous gradients for each epoch

        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)  # forward pass

        loss = outputs[0]
        loss.backward()  # getting the loss and performing backward pass

        total_loss += loss.item()  # tracking loss

        torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_norm)
        # ^^^ preventing exploding grads

        Model.optimizer.step()  # updates parameters

        scheduler.step()  # update learning_rate

    avg_train_loss = total_loss / len(train_dataloader)
    print('Average train loss : {}'.format(avg_train_loss))

    loss_values.append(avg_train_loss)  # storing loss values if you choose to plot learning curve

    # /|\==>VALIDATION(ONEPASS)<==\|/
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    for batch in valid_dataloader:
        batch = tuple(t.to(config.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.type(torch.long)
        b_labels = b_labels.type(torch.long)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print('Validation loss: {}'.format(eval_loss))

    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                 for p_i, l_i, in zip(p, l) if tag_values[l_i] != 'PAD']

    valid_tags = [tag_values[l_i] for l in true_labels
                  for l_i in l if tag_values[l_i] != 'PAD']

    print('Validation Accuracy: {}'.format(accuracy_score(pred_tags, valid_tags)))
    print('Validation F-1 Score:{}'.format(f1_score([pred_tags], [valid_tags])))


