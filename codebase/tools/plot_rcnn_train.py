# Adjusting the parsing to handle two data points per epoch and plotting mid points
import numpy as np
import matplotlib.pyplot as plt
import json
# Reinitializing the lists for the new data processing
epochs = []
mid_losses = []
mid_accuracies = []
mid_learning_rates = []

# Variables to store intermediate values
losses_temp = []
accuracies_temp = []
learning_rates_temp = []
current_epoch = None
new_file_path = '/Users/david/Documents/thesis/Thesis_code/hr_thesis_dataset_evaluator_config_7c_ex/config_test_time/20240110_162812/vis_data/50epoch_train.json'
# Processing the file
with open(new_file_path, 'r') as file:
    for line in file:
        line_num = 0
        
        try:
            entry = json.loads(line)
            epoch = entry.get('epoch', 0)
            line_num += 1
            # Check if we've moved to a new epoch
            if current_epoch is not None and epoch != current_epoch:
                # Calculate and store the mid point values for the previous epoch
                mid_losses.append(np.mean(losses_temp))
                mid_accuracies.append(np.mean(accuracies_temp))
                mid_learning_rates.append(np.mean(learning_rates_temp))
                
                # Reset temporary lists
                losses_temp = []
                accuracies_temp = []
                learning_rates_temp = []

            # Update the current epoch and temporary lists
            current_epoch = epoch
            epochs.append(epoch)
            losses_temp.append(entry.get('loss_mask', 0))
            accuracies_temp.append(entry.get('acc', 0))
            learning_rates_temp.append(entry.get('lr', 0))
        except json.JSONDecodeError:
            continue

# Add the last set of mid points
if losses_temp and accuracies_temp and learning_rates_temp:
    mid_losses.append(np.mean(losses_temp))
    mid_accuracies.append(np.mean(accuracies_temp))
    mid_learning_rates.append(np.mean(learning_rates_temp))

# Creating the plots with the mid point dataset
plt.figure(figsize=(15, 5))
# pitch one for each five epochs
epochs = epochs[::6]
# choose range of visualization
max_epochs = 30
epochs = epochs[:max_epochs]
mid_losses = mid_losses[:max_epochs]
mid_accuracies = mid_accuracies[:max_epochs]
mid_learning_rates = mid_learning_rates[:max_epochs]
# Loss Plot
plt.subplot(1, 3, 1)
plt.plot(epochs[:len(mid_losses)], mid_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss_Mask')
plt.title('Mask Loss over Epochs')

# Accuracy Plot
plt.subplot(1, 3, 2)
plt.plot(epochs[:len(mid_accuracies)], mid_accuracies, marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')

# Learning Rate Plot
plt.subplot(1, 3, 3)
plt.plot(epochs[:len(mid_learning_rates)], mid_learning_rates, marker='o', color='red')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Epochs')

plt.tight_layout()
plt.show()
