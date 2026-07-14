import torch
from VARS_model.model import MVNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MVNetwork(net_name="mvit_v2_s")
model = model.to(device)

ckpt = torch.load("14_model.pth.tar", map_location=device)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

print("Model loaded successfully!")

dummy = torch.randn(1, 1, 3, 16, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy)

print("Length:", len(output))
for i, out in enumerate(output):
    print(f"Output[{i}] type:", type(out))
    try:
        print(f"Output[{i}] shape:", out.shape)
    except:
        print(f"Output[{i}] value:", out)