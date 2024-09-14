import torch
from torchviz import make_dot
from utils import ConvLSTMEEGAutoencoder
import os
import glob

def select_model():
    models = glob.glob('models/*.pth')
    if not models:
        print("❌ No models found in the 'models' directory.")
        return None
    
    print("📚 Available models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. 🤖 {os.path.basename(model)}")
    
    while True:
        try:
            choice = int(input("\n🔢 Enter the number of the model you want to visualize: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

def create_block_visualization(model, input_shape):
    try:
        x = torch.randn(input_shape)
        y = model(x)
        dot = make_dot(y[0].mean(), params=dict(model.named_parameters()))
        dot.render("model_architecture", format="png", cleanup=True, view=False)
        print("✅ Block visualization saved as 'model_architecture.png'")
    except Exception as e:
        print(f"❌ Error creating block visualization: {str(e)}")
        print("Skipping block visualization...")

def create_mermaid_graph(model):
    mermaid_code = "graph TD\n"
    mermaid_code += "  Input[Input]\n"

    def add_module(module, name, parent=None):
        nonlocal mermaid_code
        if parent:
            mermaid_code += f"  {parent} --> {name}\n"
        if isinstance(module, torch.nn.Conv1d):
            mermaid_code += f"  {name}[{name}<br>Conv1D<br>{module.in_channels}->{module.out_channels}]\n"
        elif isinstance(module, torch.nn.ConvTranspose1d):
            mermaid_code += f"  {name}[{name}<br>ConvTranspose1D<br>{module.in_channels}->{module.out_channels}]\n"
        elif isinstance(module, torch.nn.LSTM):
            mermaid_code += f"  {name}[{name}<br>LSTM<br>{module.input_size}->{module.hidden_size}]\n"
        elif isinstance(module, torch.nn.Linear):
            mermaid_code += f"  {name}[{name}<br>Linear<br>{module.in_features}->{module.out_features}]\n"
        elif isinstance(module, torch.nn.ReLU):
            mermaid_code += f"  {name}[{name}<br>ReLU]\n"
        elif isinstance(module, torch.nn.Tanh):
            mermaid_code += f"  {name}[{name}<br>Tanh]\n"

    mermaid_code += "  Input --> Encoder\n"
    for name, module in model.encoder.named_children():
        add_module(module, f"Encoder_{name}", "Encoder")

    mermaid_code += "  Encoder --> Decoder\n"
    for name, module in model.decoder.named_children():
        add_module(module, f"Decoder_{name}", "Decoder")

    mermaid_code += "  Decoder --> Output[Output]\n"

    with open("model_architecture.mmd", "w") as f:
        f.write(mermaid_code)
    print("✅ Mermaid graph saved as 'model_architecture.mmd'")

def main():
    print("\n🚀 Welcome to the Model Architecture Visualizer! 👩‍💻👨‍💻\n")

    model_path = select_model()
    if not model_path:
        return

    try:
        print("📂 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        print("🔍 Extracting model parameters...")
        model_state_dict = checkpoint['model_state_dict']
        n_channels = model_state_dict['encoder.conv1.weight'].shape[1]
        hidden_size = model_state_dict['encoder.lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
        complexity = 0  # Default to 0, adjust if needed

        print("🏗️ Initializing model...")
        model = ConvLSTMEEGAutoencoder(n_channels=n_channels, hidden_size=hidden_size, complexity=complexity)
        
        print("🔄 Loading state dict...")
        model.load_state_dict(model_state_dict)
        model.eval()

        print(f"✅ Loaded model: {os.path.basename(model_path)}")
        print(f"📊 Model parameters: n_channels={n_channels}, hidden_size={hidden_size}, complexity={complexity}")

        # print("\n🖼️ Creating block visualization...")
        # input_shape = (1, n_channels, 200)  # Adjust the last dimension based on your input size
        # create_block_visualization(model, input_shape)

        print("\n📊 Creating Mermaid graph...")
        create_mermaid_graph(model)

        print("\n🎉 Model architecture visualization complete! 🎉")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("Please check your model file and try again.")

if __name__ == "__main__":
    main()