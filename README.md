    # ERA-S11

In this project, we explore the concept of skip connections and delve into the working of the ResNet (Residual Network) architecture, a groundbreaking advancement in deep learning for image classification.

## Introduction to ResNet :
Residual Networks, commonly referred to as ResNets, were introduced by Kaiming He et al. in the paper "Deep Residual Learning for Image Recognition" (2015). ResNets tackle the problem of vanishing gradients in very deep networks by introducing a new architectural element called skip connections, or shortcuts. These skip connections allow the network to learn residual functions instead of the entire transformation, enabling the training of extremely deep networks.

## Skip Connections :
Skip connections, also known as shortcut connections, provide a way to bypass one or more layers in a neural network. This is achieved by adding the input of a layer to the output of a later layer. In the context of ResNets, the skip connections involve adding the original input of a layer to the output of a later layer. Mathematically, this can be expressed as:

```
F(x) = H(x) + x
```

Here, F(x) is the desired output, H(x) is the residual function that the layer is expected to learn, and x is the original input to the layer. By introducing such skip connections, ResNets make it easier for the network to learn identity mappings, and the gradients can flow through these shortcuts, mitigating the vanishing gradient problem.

## How Skip Connections are Implemented in ResNet

Skip connections, also known as shortcut connections, are implemented in the ResNet architecture by adding the original input of a layer to the output of a later layer. This allows the network to learn residual functions instead of complete transformations, which helps in training very deep networks. In the code, these skip connections are implemented as follows:

The following model implementation can be found in [`ResNet.py`](https://github.com/shashankg69/core/blob/main/models/resnet.py)

```python
class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != self.expansion * output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, self.expansion * output_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * output_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```
`ConvBlock` Class:

This class defines a basic convolutional block used in ResNet. Each block consists of two convolutional layers followed by batch normalization and ReLU activation functions. The residual connection is added via the shortcut connection, which bypasses the convolutional layers. The block allows for controlling the stride for downsampling.

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.input_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, output_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_channels, output_channels, stride))
            self.input_channels = output_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```

`ResNet` Class:
This class defines the overall ResNet architecture. It consists of multiple convolutional layers organized into different layers (layer1, layer2, etc.). The layer1 corresponds to the initial convolutional layer. The other layers are constructed using the _make_layer function which replicates the specified number of ConvBlock blocks. These layers have increasing numbers of filters to capture different levels of features. The forward function processes the input data through the layers and computes the final class scores.

`_make_layer` Function:
This function is used within the ResNet class to create a sequence of ConvBlock layers. The stride value is used to control downsampling, and the self.input_channels variable keeps track of the input channel count for each layer.

### Skip Connections:

The core concept of skip connections is implemented in the ConvBlock class. The line `out += self.shortcut(x)` adds the original input x to the output of the second convolutional layer. This bypasses the convolutional layers and directly combines the input with the output, forming a "skip connection." The skip connection ensures that gradient information flows directly through the network layers, mitigating the vanishing gradient problem.

## GradCAM
GradCAM (Gradient-weighted Class Activation Mapping) is a technique used to visualize the areas of an image that contributed the most to a neural network's prediction. It helps in understanding where the network is focusing its attention within an image.

### Why GradCAM?

`Interpretable Insights`: GradCAM provides interpretable visualizations, allowing us to see which parts of an image influenced a certain classification decision.

`Model Understanding`: It helps in understanding whether the model is focusing on the correct features or learning unintended patterns.

`Model Validation`: GradCAM can be used to validate whether a network is making decisions based on relevant image regions.

# Results
- No of Params: `11,173,962`
- Best Training Accuracy : `85.81`
- Best Test Accuracy : `90.25`


![image](https://github.com/shashankg69/ERA-S11/assets/59787210/e0406969-3687-4c0e-9f2e-1e486817b477)
![image](https://github.com/shashankg69/ERA-S11/assets/59787210/311bfc08-84e3-43fa-805a-139c5ebe8bd6)






