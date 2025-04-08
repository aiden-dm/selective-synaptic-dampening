# Original implementations of https://github.com/vikram2000b/Fast-Machine-Unlearning and https://github.com/vikram2000b/bad-teaching-unlearning

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .datasets import UnLearningData
import numpy as np
from .utils import *

# Adding the local files to the system path
sys.path.append('/content/Unlearning-MIA-Eval/Final_Structure')

from Final_Structure.evaluate import train_validation

def UnlearnerLoss(
    output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature
):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(
    model,
    unlearning_teacher,
    full_trained_teacher,
    unlearn_data_loader,
    optimizer,
    device,
    KL_temperature,
):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def fit_one_unlearning_cycle(epochs, model, train_loader, val_loader, lr, device):
    history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            loss.backward()
            train_losses.append(loss.detach().cpu())

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history


def blindspot_unlearner(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    loaders,
    epochs=10,
    optimizer="adam",
    lr=0.01,
    batch_size=256,
    device="cuda",
    KL_temperature=1,
):
    # creating the unlearning dataset.
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    # Extract loaders used for validation
    train_retain_loader = loaders[0]
    train_forget_loader = loaders[1]
    val_retain_loader = loaders[2]
    val_forget_loader = loaders[3]

    # Create a history variable to store information
    history = []
    tf_accs = []
    tr_accs = []
    vf_accs = []
    vr_accs = []
    losses = []
    epoch_list = []

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer  # (model.parameters())

    for epoch in range(epochs):
        loss = unlearning_step(
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
        )
        
        losses.append(loss)
        epoch_list.append(epoch)
        acc_dict = train_validation(model, 
                                    train_retain_loader, 
                                    train_forget_loader, 
                                    val_retain_loader, 
                                    val_forget_loader)
        tr_accs.append(acc_dict['tr_acc'])
        tf_accs.append(acc_dict['tf_acc'])
        vr_accs.append(acc_dict['vr_acc'])
        vf_accs.append(acc_dict['vf_acc'])
        history.append({
            'losses': losses,
            'epoch_list': epoch_list,
            'tr_accs': tr_accs,
            'tf_accs': tf_accs,
            'vr_accs': vr_accs,
            'vf_accs': vf_accs
        })

        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))

    return model, history

class UNSIR_noise(torch.nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


def UNSIR_noise_train(
    noise, model, forget_class_label, num_epochs, noise_batch_size, device="cuda"
):
    opt = torch.optim.Adam(noise.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = []
        inputs = noise()
        labels = torch.zeros(noise_batch_size).to(device) + forget_class_label
        outputs = model(inputs)
        loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(
            torch.sum(inputs**2, [1, 2, 3])
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss.append(loss.cpu().detach().numpy())
        if epoch % 5 == 0:
            print("Loss: {}".format(np.mean(total_loss)))

    return noise


def UNSIR_create_noisy_loader(
    noise,
    forget_class_label,
    retain_samples,
    batch_size,
    num_noise_batches=80,
    device="cuda",
):
    noisy_data = []
    for i in range(num_noise_batches):
        batch = noise()
        for i in range(batch[0].size(0)):
            noisy_data.append(
                (
                    batch[i].detach().cpu(),
                    torch.tensor(forget_class_label),
                    torch.tensor(forget_class_label),
                )
            )
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][2]),
                torch.tensor(retain_samples[i][2]),
            )
        )
    noisy_data += other_samples
    noisy_loader = DataLoader(noisy_data, batch_size=batch_size, shuffle=True)

    return noisy_loader
