import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix
import torchvision.models as models

# Data Transformation

transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),

    transforms.RandomHorizontalFlip(), # No parameters
    transforms.RandomRotation(degrees=(-20,20)),
    transforms.ColorJitter(brightness=(0.5,1.5)),

    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


# Data Declaration / Preparation

train_dataset = ImageFolder('./chest_xray_clean/train',transform=transform)

test_dataset = ImageFolder('./chest_xray_clean/test',transform=test_transform)

val_dataset = ImageFolder('./chest_xray_clean/val' , transform=test_transform)


train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=True)



# ------------------------------- Model Development --------------------

# Hyper-Parameters
learning_rate = 0.001
batch_size = 32
num_epochs = 5

model = models.resnet18(weights="IMAGENET1K_V1")


# Freezing
for params in model.parameters():
    params.requires_grad = False


# Fetch the number of features resnet18 currently has (512 for ResNet18)
num_ftrs = model.fc.in_features

# Only the parameters of this layer will be trained
model.fc = nn.Linear(num_ftrs,2)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# Only new layer will be trained, not the entire model, the entire model is "freezed"
optimizer = torch.optim.SGD(model.fc.parameters(),lr=learning_rate)

#Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)

PATIENCE = 5
min_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = None


n_total_steps = len(train_loader)

# Running the epochs but only for the FC new layer, not the entire model

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        # Convert them into proper for the current device
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # Loss calculation
        loss = criterion(output,labels)

        #Backward Propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tracking the losss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Batch {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')


    avg_train_loss = epoch_train_loss_sum / n_total_steps

    # ------ Validation on unseen data -----------

    model.eval()
    val_loss = 0.0
    n_correct  = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images,labels = images.to(device) , labels.to(device)

            output = model(images)

            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader) * 100

        # -- Reporting and Saving
        print(f'Epoch :- {epoch + 1}  , Train Average loss :- {avg_train_loss:.4f} , Val average loss :- {avg_val_loss:.4f} ,Val accuracy :- {val_accuracy:.3f}' )

        #Scheduelr Setup
        scheduler.step()
        print(f"Current LR : {optimizer.param_groups[0]['lr']:.6f}")

        # Model Check pointing
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_weights = model.state_dict()
            print(f"New best model saved , Val loss :- {min_val_loss:.3f}")
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve. Patience : {epochs_no_improve} / {PATIENCE}')

        # Early Stopping
        if epochs_no_improve == PATIENCE:
            print(f'Early stopping triggered after {epoch + 1} epochs of no improvement.')
            break


min_val_loss = float('inf')
epochs_no_improve = 0

# With this setup, we have the entire backbone frozen, so now we wanna experiment with unfreezing
# last residual block

# --------- Phase : 2  --- Unfreezing last residual block -------


for param in model.layer4.parameters():
    param.requires_grad = True

# Training both the final FC layer and last block of ResNet

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.0005,
    momentum=0.9
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)

num_epochs = 3

# Load the best model weights from phase 1 before starting phase 2 training
if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded best model weights from phase 1 training.')

min_val_loss = float('inf')
epochs_no_improve = 0

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        # Convert them into proper for the current device
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # Loss calculation
        loss = criterion(output,labels)

        #Backward Propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tracking the losss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Batch {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')

    avg_train_loss = epoch_train_loss_sum / n_total_steps

    # ------ Validation on unseen data -----------

    model.eval()
    val_loss = 0.0
    n_correct  = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images,labels = images.to(device) , labels.to(device)

            output = model(images)

            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader) * 100

        # -- Reporting and Saving
        print(f'Epoch :- {epoch + 1}  , Train Average loss :- {avg_train_loss:.4f} , Val average loss :- {avg_val_loss:.4f} ,Val accuracy :- {val_accuracy:.3f}' )




# Phase : 3 ------------ Unfreeze the entire network ---------------


# Making sure to load the weights of previous model of they are any better
if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded the Phase 2 best weights in phase 3 model training')


for param in model.parameters():
    param.requires_grad = True


optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.0001,
    momentum=0.9
)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)


num_epochs = 3


# Reset Checkpointing variables for Phase 3
min_val_loss = float('inf')
epochs_no_improve = 0

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        # Convert them into proper for the current device
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # Loss calculation
        loss = criterion(output,labels)

        #Backward Propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tracking the losss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Batch {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')

    avg_train_loss = epoch_train_loss_sum / n_total_steps


    # ------ Validation on unseen data -----------

    model.eval()
    val_loss = 0.0
    n_correct  = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images,labels = images.to(device) , labels.to(device)

            output = model(images)

            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader) * 100

        # -- Reporting and Saving
        print(f'Epoch :- {epoch + 1}  , Train Average loss :- {avg_train_loss:.4f} , Val average loss :- {avg_val_loss:.4f} ,Val accuracy :- {val_accuracy:.3f}' )

        # Scheduler
        scheduler.step()

        print(f"Current LR : {optimizer.param_groups[0]['lr']:.8f}") # Use 8 decimal places for tiny LR

        # ⭐ 2. Model Check pointing
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Save the current best weights for use in final testing
            best_model_weights = model.state_dict()
            print(f"New best model saved, Val loss :- {min_val_loss:.3f}")
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve. Patience : {epochs_no_improve} / {PATIENCE}')

        # ⭐ 3. Early Stopping
        if epochs_no_improve == PATIENCE:
            print(f'Early stopping triggered after {epoch + 1} epochs of no improvement.')
            break





# ---------------- Final Testing ------------------

print("\n--- FINAL EVALUATION ON TEST DATA ---")


if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded absolute best model weights for final testing.')



model.eval()
test_loss = 0.0
n_correct_test = 0
total_samples_test = 0


# For evaluating purposes
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(output, 1)
        n_correct_test += (predicted == labels).sum().item()
        total_samples_test += images.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(output.cpu().numpy())


avg_test_loss = test_loss / total_samples_test
test_accuracy = n_correct_test / total_samples_test * 100

print(f'Test Average Loss: {avg_test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.3f}%')


# Evaluating using metrics

y_true = np.array(y_true)
y_pred_logits = np.array(y_pred)

# Convert logits to probabilites (Softmax)
y_pred_probs = F.softmax(torch.from_numpy(y_pred_logits),dim=1).numpy()

y_pred_positive_class_prob = y_pred_probs[:,1]

#Convert logits to hard predictions
y_pred_hard = np.argmax(y_pred_probs,axis=1)

# F1-Score
f1_resnet18 = f1_score(y_true,y_pred_hard,average='binary')

# Roc Auc Score
roc_auc_resnet = roc_auc_score(y_true,y_pred_positive_class_prob)

print(f'\nF1 Score (Binary): {f1_resnet18:.4f}')
print(f'ROC AUC Score: {roc_auc_resnet:.4f}')


# Confusion Matrix
cm = confusion_matrix(y_true,y_pred_hard)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Saving the model
torch.save(model.state_dict(),'X_ray_classifier.pth')


# save the class indexing (required from deployment)
class_to_idx = train_dataset.class_to_idx
torch.save(class_to_idx,'class_to_index.pth')


print('Everything done')






