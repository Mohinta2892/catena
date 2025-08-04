# File: /rsunet.py
import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvMod(nn.Module):
    """
    Convolution module for isotropic 3D data.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvMod, self).__init__()

        # Isotropic convolutions - changed from (1,3,3) to (3,3,3)
        self.conv1 = nn.Conv3d(in_channels,  out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # BatchNorm
        self.bn1 = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.01)
        self.bn2 = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.01)
        self.bn3 = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.01)

        # Activation function
        self.activation = nn.ELU()

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        skip = x

        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Conv 3 with residual connection
        x = self.conv3(x)
        x = x + skip
        x = self.bn3(x)
        return self.activation(x)


class RSUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(RSUNet, self).__init__()

        # Initial convolution - changed from (1,5,5) to (5,5,5) for isotropic
        self.conv0 = nn.Conv3d(in_ch, 28, kernel_size=5, padding=2)
        self.elu0 = nn.ELU()

        # Encoder path
        self.conv1 = ConvMod(28, 36)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Changed from (1,2,2) to 2

        self.conv2 = ConvMod(36, 48)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ConvMod(48, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = ConvMod(64, 80)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv_bottleneck = ConvMod(80, 96)

        # Decoder path with proper channel matching
        self.up5 = nn.ConvTranspose3d(96, 80, kernel_size=2, stride=2)
        self.conv5 = ConvMod(80 + 64, 64)  # 80 from up5 + 64 from conv3

        self.up6 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.conv6 = ConvMod(48 + 48, 48)  # 48 from up6 + 48 from conv2

        self.up7 = nn.ConvTranspose3d(48, 36, kernel_size=2, stride=2)
        self.conv7 = ConvMod(36 + 36, 36)  # 36 from up7 + 36 from conv1

        self.up8 = nn.ConvTranspose3d(36, 28, kernel_size=2, stride=2)
        self.conv8 = ConvMod(28 + 28, 28)  # 28 from up8 + 28 from conv0

        # Final convolution - changed from (1,5,5) to (5,5,5)
        self.conv9 = nn.Conv3d(28, out_ch, kernel_size=5, padding=2)

        # Optional sigmoid activation
        self.sig0 = nn.Sigmoid()

    def _crop_and_concat(self, upsampled, skip):
        """
        Crop skip connection to match upsampled tensor size and concatenate.
        """
        # Get the target size from upsampled tensor
        target_size = upsampled.shape[2:]
        skip_size = skip.shape[2:]

        # Calculate cropping for each dimension
        crops = []
        for i in range(3):  # 3D
            diff = skip_size[i] - target_size[i]
            if diff > 0:
                start = diff // 2
                end = start + target_size[i]
                crops.append(slice(start, end))
            else:
                crops.append(slice(None))

        # Crop the skip connection
        skip_cropped = skip[:, :, crops[0], crops[1], crops[2]]

        # Concatenate along channel dimension
        return torch.cat([upsampled, skip_cropped], dim=1)

    def forward(self, x):
        # Encoder
        c0 = self.conv0(x)
        c0 = self.elu0(c0)

        c1 = self.conv1(c0)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bottleneck = self.conv_bottleneck(p4)

        # Decoder with proper size matching
        up_5 = self.up5(bottleneck)
        c5 = self.conv5(self._crop_and_concat(up_5, c3))

        up_6 = self.up6(c5)
        c6 = self.conv6(self._crop_and_concat(up_6, c2))

        up_7 = self.up7(c6)
        c7 = self.conv7(self._crop_and_concat(up_7, c1))

        up_8 = self.up8(c7)
        c8 = self.conv8(self._crop_and_concat(up_8, c0))

        # Final output
        c9 = self.conv9(c8)

        # Return logits for BCEWithLogitsLoss
        return c9


# # Test function to verify the network works with 128^3 input
def test_network():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RSUNet(in_ch=1, out_ch=1).to(device)

    # Test with batch size 1, 1 channel, 128x128x128
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)

    # Print parameter count
    print_model_parameters(model, "RSUNet")

    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Network test passed!")

    return model

if __name__ == "__main__":
    model = test_network()