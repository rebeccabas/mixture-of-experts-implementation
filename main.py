import torch
import torch.nn as nn
import torch.optim as optim
from expert import Expert
from MOE import MOE

num_samples = 5000
input_dim = 4
hidden_dim = 32

# generate equal numbers of labels 0, 1, and 2
y_data = torch.cat([
    torch.zeros(num_samples // 3),
    torch.ones(num_samples // 3),
    torch.full((num_samples - 2 * (num_samples // 3),), 2)  # filling the remaining to ensure exact num_samples
]).long()

# Biasing the data based on the labels
x_data = torch.randn(num_samples, input_dim)

for i in range(num_samples):
    if y_data[i] == 0:
        x_data[i, 0] += 1 
    elif y_data[i] == 1:
        x_data[i, 1] -= 1 
    elif y_data[i] == 2:
        x_data[i, 0] -= 1 

# shuffle to randomize the order
indices = torch.randperm(num_samples)
x_data = x_data[indices]
y_data = y_data[indices]

# verify the label distribution
y_data.bincount()


shuffled_indices = torch.randperm(num_samples)
x_data = x_data[shuffled_indices]
y_data = y_data[shuffled_indices]


x_train_experts = x_data[:int(num_samples/2)]
y_train_experts = y_data[:int(num_samples/2)]

mask_expert1 = (y_train_experts == 0) | (y_train_experts == 1)
mask_expert2 = (y_train_experts == 1) | (y_train_experts == 2)
mask_expert3 = (y_train_experts == 0) | (y_train_experts == 2)
mask_expert4 = (y_train_experts == 1) | (y_train_experts == 2)

num_samples_per_expert = \
min(mask_expert1.sum(), mask_expert2.sum(), mask_expert3.sum())

x_expert1 = x_train_experts[mask_expert1][:num_samples_per_expert]
y_expert1 = y_train_experts[mask_expert1][:num_samples_per_expert]

x_expert2 = x_train_experts[mask_expert2][:num_samples_per_expert]
y_expert2 = y_train_experts[mask_expert2][:num_samples_per_expert]

x_expert3 = x_train_experts[mask_expert3][:num_samples_per_expert]
y_expert3 = y_train_experts[mask_expert3][:num_samples_per_expert]

x_expert4 = x_train_experts[mask_expert4][:num_samples_per_expert]
y_expert4 = y_train_experts[mask_expert4][:num_samples_per_expert]


x_remaining = x_data[int(num_samples/2)+1:]
y_remaining = y_data[int(num_samples/2)+1:]

split = int(0.8 * len(x_remaining))
x_train_moe = x_remaining[:split]
y_train_moe = y_remaining[:split]

x_test = x_remaining[split:]
y_test = y_remaining[split:]

print(x_train_moe.shape,"\n", x_test.shape,"\n",
      x_expert1.shape,"\n",
      x_expert2.shape,"\n", x_expert3.shape, "\n", x_expert4.shape)


output_dim = 3
hidden_dim = 32

epochs = 1000
learning_rate = 0.001

expert1 = Expert(input_dim, hidden_dim, output_dim)
expert2 = Expert(input_dim, hidden_dim, output_dim)
expert3 = Expert(input_dim, hidden_dim, output_dim)
expert4 = Expert(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()#loss calculation


#optimzation
optimizer_expert1 = optim.Adam(expert1.parameters(), lr=learning_rate)
optimizer_expert2 = optim.Adam(expert2.parameters(), lr=learning_rate)
optimizer_expert3 = optim.Adam(expert3.parameters(), lr=learning_rate)
optimizer_expert4 = optim.Adam(expert4.parameters(), lr=learning_rate)


#training loops
for epoch in range(epochs):
    optimizer_expert1.zero_grad()
    outputs_expert1 = expert1(x_expert1)
    loss_expert1 = criterion(outputs_expert1, y_expert1)
    loss_expert1.backward()
    optimizer_expert1.step()

for epoch in range(epochs):
    optimizer_expert2.zero_grad()
    outputs_expert2 = expert2(x_expert2)
    loss_expert2 = criterion(outputs_expert2, y_expert2)
    loss_expert2.backward()
    optimizer_expert2.step()

for epoch in range(epochs):
    optimizer_expert3.zero_grad()
    outputs_expert3 = expert3(x_expert3)
    loss_expert3 = criterion(outputs_expert3, y_expert3)
    loss_expert3.backward()

for epoch in range(epochs):
    optimizer_expert4.zero_grad()
    outputs_expert4 = expert4(x_expert4)
    loss_expert4 = criterion(outputs_expert4, y_expert4)
    loss_expert4.backward()


#initialize MOE
moe_model = MOE([expert1, expert2, expert3, expert4])

#train
optimizer_moe = optim.Adam(moe_model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    optimizer_moe.zero_grad()
    outputs_moe = moe_model(x_train_moe)
    loss_moe = criterion(outputs_moe, y_train_moe)
    loss_moe.backward()
    optimizer_moe.step()

def evaluate(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
    return accuracy

accuracy_expert1 = evaluate(expert1, x_test, y_test)
accuracy_expert2 = evaluate(expert2, x_test, y_test)
accuracy_expert3 = evaluate(expert3, x_test, y_test)
accuracy_expert4 = evaluate(expert4, x_test, y_test)
accuracy_moe = evaluate(moe_model, x_test, y_test)

print("Expert 1 Accuracy:", accuracy_expert1)
print("Expert 2 Accuracy:", accuracy_expert2)
print("Expert 3 Accuracy:", accuracy_expert3)
print("Expert 4 Accuracy:", accuracy_expert4)
print("Mixture of Experts Accuracy:", accuracy_moe)