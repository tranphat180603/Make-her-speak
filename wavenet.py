import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Train WaveNet Model")
    
    # Dataset parameters
    parser.add_argument('--files_dir', type=str, required=True, help='Path to the directory containing audio files.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sampling rate for audio.')
    parser.add_argument('--mu', type=int, default=255, help='Mu value for mu-law quantization.')
    
    # Model hyperparameters
    parser.add_argument('--layers', type=int, default=20, help='Number of layers in the model.')
    parser.add_argument('--stacks', type=int, default=2, help='Number of stacks in the model.')
    parser.add_argument('--in_channels', type=int, default=256, help='Number of input channels.')
    parser.add_argument('--res_channels', type=int, default=128, help='Number of residual channels.')
    parser.add_argument('--gate_channels', type=int, default=128, help='Number of gate channels.')
    parser.add_argument('--skip_channels', type=int, default=256, help='Number of skip channels.')
    parser.add_argument('--out_channels', type=int, default=256, help='Number of output channels.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolutions.')
    parser.add_argument('--local_cond_channels', type=int, default=80, help='Number of local condition channels.')
    parser.add_argument('--upsample_scales', type=int, nargs='+', default=[4, 4], help='Upsample scales for upsampling network.')
    parser.add_argument('--num_classes', type=int, default=256, help='Number of classes for embedding.')
    
    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer.')
    
    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    parser.add_argument('--save_model', type=str, default='wavenet_model.pth', help='Path to save the trained model.')
    
    args = parser.parse_args()
    return args

def get_audio_paths(files_dir):
    """
    Assemble all audio file paths in the specified directory.
    
    Args:
        files_dir (str): Path to the directory containing audio files.
        
    Returns:
        list: Sorted list of full paths to audio files.
    """
    audio_names = sorted(os.listdir(files_dir))
    audio_paths = [os.path.join(files_dir, name) for name in audio_names]
    return audio_paths

def mu_law_quantize(waveform, mu=255):
    """
    Apply mu-law quantization to the waveform.
    
    Args:
        waveform (torch.Tensor): Audio waveform tensor.
        mu (int, optional): Number of quantization levels. Defaults to 255.
        
    Returns:
        torch.Tensor: Quantized waveform.
    """
    waveform = waveform / waveform.abs().max()  # Normalize to range [-1, 1]
    mu_tensor = torch.tensor(mu, dtype=waveform.dtype, device=waveform.device)  
    waveform_mu = torch.sign(waveform) * (torch.log1p(mu * torch.abs(waveform)) / torch.log1p(mu_tensor))
    
    # Scale the values to [0, mu] and round to integers
    quantized = ((waveform_mu + 1) / 2 * mu_tensor).long()
    return quantized

def window_fn(window_length):
    """
    Create a Hann window function.
    
    Args:
        window_length (int): Length of the window.
        
    Returns:
        torch.Tensor: Hann window tensor.
    """
    return torch.hann_window(window_length, device='cpu')

def compute_mel_spectrogram(waveform, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=216, f_min=125.0, f_max=7600.0):
    """
    Compute the log-mel spectrogram of the waveform.
    
    Args:
        waveform (torch.Tensor): Audio waveform tensor.
        sample_rate (int, optional): Sampling rate. Defaults to 22050.
        n_mels (int, optional): Number of mel bands. Defaults to 80.
        n_fft (int, optional): FFT size. Defaults to 1024.
        hop_length (int, optional): Hop length for STFT. Defaults to 216.
        f_min (float, optional): Minimum frequency. Defaults to 125.0.
        f_max (float, optional): Maximum frequency. Defaults to 7600.0.
        
    Returns:
        torch.Tensor: Log-mel spectrogram.
    """
    # Define STFT parameters
    win_length = n_fft  # Hann window function with the same length as n_fft
    
    # Apply STFT
    stft_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=window_fn,
        power=None  # We want complex values for further processing
    )
    
    spectrogram = stft_transform(waveform)  # Remain on CPU
    
    # Compute magnitude
    spectrogram_magnitude = torch.abs(spectrogram)
    
    # Apply Mel filterbank
    mel_transform = T.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_stft=spectrogram_magnitude.size(1)
    )
    mel_spectrogram = mel_transform(spectrogram_magnitude)
    
    # Clip and log
    mel_spectrogram = torch.clamp(mel_spectrogram, min=0.01)
    log_mel_spectrogram = torch.log(mel_spectrogram)
    
    return log_mel_spectrogram

class AudioDataset(Dataset):
    """
    Custom Dataset for loading and processing audio files.
    """
    def __init__(self, audio_dir, sample_rate=22050, max_length_seconds=3, mu=255):
        """
        Initialize the dataset.
        
        Args:
            audio_dir (list): List of audio file paths.
            sample_rate (int, optional): Sampling rate. Defaults to 22050.
            max_length_seconds (int, optional): Maximum length of audio in seconds. Defaults to 3.
            mu (int, optional): Number of quantization levels. Defaults to 255.
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_length = sample_rate * max_length_seconds
        self.mu = mu

    def __len__(self):
        return len(self.audio_dir)

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.audio_dir[idx]
        waveform, sr = torchaudio.load(audio_path)

        # Ensure sample rate is 22050 Hz
        if sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch. Expected {self.sample_rate}, but got {sr}.")

        # Trim or pad the waveform to the desired length
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.size(1) < self.max_length:
            pad_length = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Apply mu-law quantization
        quantized = mu_law_quantize(waveform, mu=self.mu).squeeze(0)  # Shape: [length]

        # Prepare input and target by shifting
        input_quantized = torch.zeros_like(quantized)
        input_quantized[1:] = quantized[:-1]
        input_quantized[0] = self.mu // 2  # Start token (e.g., middle of the quantization range)

        target = quantized  # The actual quantized waveform

        # Compute mel spectrogram
        mel_spectrogram = compute_mel_spectrogram(waveform).squeeze(0)  # Shape: [80, time]

        return input_quantized, mel_spectrogram, target

class Conv1d(nn.Module):
    """
    1D Convolutional layer with dropout and Kaiming initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0, **kwargs):
        super(Conv1d, self).__init__()
        self.m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        nn.init.kaiming_normal_(self.m.weight, nonlinearity="relu")
        
        if self.m.bias is not None:
            nn.init.constant_(self.m.bias, 0)
        
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x):
        x = self.dropout(x)
        x = self.m(x)
        return x 

class UpsamplingNetwork(nn.Module):
    """
    Upsampling network using transposed convolutions for local conditioning.
    """
    def __init__(self, c_in, c_out, upsample_scales):
        super(UpsamplingNetwork, self).__init__()
        self.conv_layers = nn.ModuleList()
        for scale in upsample_scales:
            conv = nn.ConvTranspose1d(c_in, c_out, scale * 2, scale, scale // 2)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
            self.conv_layers.append(conv)
            c_in = c_out

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x), 0.4)
        return x

class Conv1d_1x1(nn.Module):
    """
    1x1 Convolutional layer.
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d_1x1, self).__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        return self.conv(x)

class DilatedCausalConv(nn.Module):
    """
    Dilated causal convolution to ensure causality in the network.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCausalConv, self).__init__()
        self.padding = dilation * (kernel_size - 1)
        self.dilated_causal_conv = Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding
        )
        
    def forward(self, x):
        return self.dilated_causal_conv(x)[:, :, :-self.padding]

class ResidualBlock(nn.Module):
    """
    Residual block consisting of dilated causal convolution, gating mechanism, and skip connections.
    """
    def __init__(self, res_channels, gate_channels, skip_channels, kernel_size, dilation, local_cond_channels):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = DilatedCausalConv(res_channels, gate_channels, kernel_size, dilation)
        self.local_cond_proj = Conv1d_1x1(local_cond_channels, gate_channels)
        self.res_conv = Conv1d_1x1(gate_channels // 2, res_channels)
        self.skip_conv = Conv1d_1x1(gate_channels // 2, skip_channels)

    def forward(self, x, local_cond):
        dilated_out = self.dilated_conv(x)
        local_cond = self.local_cond_proj(local_cond)
        dilated_out = dilated_out + local_cond
        
        tanh_out, sigmoid_out = torch.chunk(dilated_out, 2, dim=1)
        gated_out = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
        
        res_out = self.res_conv(gated_out)
        skip_out = self.skip_conv(gated_out)
        
        return (x + res_out) * 0.707, skip_out  # Scale residual to preserve variance

class WaveNetModel(nn.Module):
    """
    WaveNet model for audio generation with conditional inputs.
    """
    def __init__(
        self, layers, stacks, in_channels, res_channels, gate_channels, skip_channels, out_channels, 
        kernel_size, local_cond_channels, upsample_scales, num_classes=256
    ):
        super(WaveNetModel, self).__init__()
        
        # Embedding layer to convert integer inputs to dense vectors
        self.embedding = nn.Embedding(num_classes, in_channels)  # [num_classes, in_channels]
        
        self.start_conv = Conv1d_1x1(in_channels, res_channels)
        self.local_cond_upsample = UpsamplingNetwork(local_cond_channels, res_channels, upsample_scales)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                res_channels, gate_channels, skip_channels, kernel_size, 
                2**(layer % (layers // stacks)), res_channels
            )
            for stack in range(stacks) 
            for layer in range(layers // stacks)
        ])
        
        self.end_conv1 = Conv1d_1x1(skip_channels, out_channels)
        self.end_conv2 = Conv1d_1x1(out_channels, out_channels)
    
    def forward(self, audio_input, local_cond):
        # Embed the input tensor
        x = self.embedding(audio_input)  # Shape: [B, T, C]
        x = x.permute(0, 2, 1)  # Shape: [B, C, T]
        x = self.start_conv(x)
        local_cond = self.local_cond_upsample(local_cond)
        
        # Ensure local_cond has the same length as x
        if local_cond.size(2) > x.size(2):
            local_cond = local_cond[:, :, :x.size(2)]
        elif local_cond.size(2) < x.size(2):
            local_cond = F.pad(local_cond, (0, x.size(2) - local_cond.size(2)))
            
        skip_connections = []
        for res_block in self.res_blocks:
            x, skip = res_block(x, local_cond)
            skip_connections.append(skip)
        
        out = torch.sum(torch.stack(skip_connections), dim=0) / (len(self.res_blocks) ** 0.5)
        out = F.relu(out)
        out = self.end_conv1(out)
        out = F.relu(out)
        out = self.end_conv2(out)
        
        return out

def initialize_model(args):
    """
    Initialize the WaveNet model, loss function, and optimizer.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        tuple: (model, criterion, optimizer, scaler)
    """
    model = WaveNetModel(
        layers=args.layers, 
        stacks=args.stacks, 
        in_channels=args.in_channels,  
        res_channels=args.res_channels,  
        gate_channels=args.gate_channels,
        skip_channels=args.skip_channels,  
        out_channels=args.out_channels, 
        kernel_size=args.kernel_size, 
        local_cond_channels=args.local_cond_channels, 
        upsample_scales=args.upsample_scales, 
        num_classes=args.num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    return model, criterion, optimizer, scaler

def train(model, dataloader, criterion, optimizer, scaler, num_epochs=10):
    """
    Train the WaveNet model.
    
    Args:
        model (nn.Module): The WaveNet model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scaler (torch.cuda.amp.GradScaler): Scaler for mixed precision.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs1, inputs2, targets) in enumerate(dataloader):
            inputs1 = inputs1.to(DEVICE, dtype=torch.long)
            inputs2 = inputs2.to(DEVICE, dtype=torch.float32)
            targets = targets.to(DEVICE, dtype=torch.long)

            optimizer.zero_grad()

            # Proper context management
            if scaler:
                # Mixed Precision Training
                with torch.cuda.amp.autocast():
                    outputs = model(inputs1, inputs2)
                    loss = criterion(outputs, targets)
                
                # Debugging statements (optional)
                # print(f"Loss requires grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard Training
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, targets)
                
                # Debugging statements (optional)
                # print(f"Loss requires grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")

                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

def main():
    """
    Main function to execute the training pipeline.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Confirm device
    print(f"Using: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Get audio file paths
    audio_dir = get_audio_paths(args.files_dir)
    print(f"Found {len(audio_dir)} audio files.")
    
    # Initialize dataset and dataloader
    dataset = AudioDataset(audio_dir=audio_dir, sample_rate=args.sample_rate, mu=args.mu)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    # Initialize model, loss, optimizer, and scaler
    model, criterion, optimizer, scaler = initialize_model(args)
    
    # Start training
    train(model, dataloader, criterion, optimizer, scaler, num_epochs=args.num_epochs)
    
    # Optionally, save the trained model
    torch.save(model.state_dict(), args.save_model)
    print(f"Training complete and model saved to {args.save_model}.")

if __name__ == "__main__":
    main()
