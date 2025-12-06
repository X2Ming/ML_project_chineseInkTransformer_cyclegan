import matplotlib.pyplot as plt
import re


LOG_FILE_PATH = 'loss_log.txt'  


def parse_log(filepath):
    losses = {
        'G_A': [], 'G_B': [], 
        'cycle_A': [], 'cycle_B': [], 
        'idt_A': [], 'idt_B': [],
        'clip_style_AtoB': [], 
        'edge_AtoB': []
    }
    
    iters = []
    
    # Regular expression to match
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "epoch:" in line and "G_A:" in line:
            iters.append(i) 
            
            for key in losses.keys():
                pattern = re.compile(rf"{key}: ([\d\.]+)")
                match = pattern.search(line)
                if match:
                    losses[key].append(float(match.group(1)))
                else:
                    if len(losses[key]) > 0:
                        losses[key].append(losses[key][-1])
                    else:
                        losses[key].append(0)
                        
    return iters, losses

def plot_losses(iters, losses):
    plt.figure(figsize=(15, 10))
    
    #GAN Loss & Cycle Loss
    plt.subplot(2, 1, 1)
    plt.plot(losses['G_A'], label='G_A (Photo->Ink)', alpha=0.7)
    plt.plot(losses['G_B'], label='G_B (Ink->Photo)', alpha=0.7)
    plt.plot(losses['cycle_A'], label='Cycle A', linestyle='--', alpha=0.5)
    plt.title('Basic GAN & Cycle Losses')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Our CLIP & Edge Loss
    plt.subplot(2, 1, 2)
    if any(losses['clip_style_AtoB']):
        plt.plot(losses['clip_style_AtoB'], label='CLIP Style Loss', color='purple', linewidth=2)
    if any(losses['edge_AtoB']):
        plt.plot(losses['edge_AtoB'], label='Edge Loss', color='orange', linewidth=2)
        
    plt.title('Our Proposed Losses (Style & Edge)')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("Successfully saved training_curves.png")
    plt.show()

try:
    x, y = parse_log(LOG_FILE_PATH)
    if len(x) > 0:
        plot_losses(x, y)
    else:
        print("Cannot find any loss data")
except FileNotFoundError:
    print(f"Error: Cannot find {LOG_FILE_PATH}.")