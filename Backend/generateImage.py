from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

img, label = test_dataset[0]

plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.savefig("digit.png")
print("Saved digit.png with label:", label)
