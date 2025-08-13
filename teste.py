import torch
import torchvision.models as models
import time
import timeit

num_epochs = 5
total_samples = 20000
batch_size = 128
input_shape = (3, 256, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = models.resnet50().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
def get_dummy_batch():
    return (
        torch.randn(batch_size, *input_shape).to(device),
        torch.randint(0, 1000, (batch_size,)).to(device)
    )

# Benchmark
num_batches = total_samples // batch_size
start_time = time.time()

for epoch in range(num_epochs):
    a = timeit.default_timer()
    model.train()
    for _ in range(num_batches):
        inputs, labels = get_dummy_batch()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    b = timeit.default_timer()
    print(f'{b-a} seconds')

total_time = time.time() - start_time
print(f"Total training time: {total_time} seconds")
