import argparse
import helper
import torch
import json

# Parsing arguments
parser = argparse.ArgumentParser(description='Predict flower type from image')
parser.add_argument('image', help='path to the image')
parser.add_argument('model', help='path to the model')
parser.add_argument('--top_k', default=1, help='return top KKK most likely classes')
parser.add_argument('--category_names', default=False, help='path to a mapping of categories to real names')
parser.add_argument('--gpu', action='store_const', const=True, default=False, help='use GPU for inference')
args = parser.parse_args()

# Loading model
model = helper.load_trained_model(args.model)
device = 'cuda' if args.gpu and torch.cuda.is_available else 'cpu'
model.to(device)

# Inference
probs, classes = helper.predict(args.image, model, topk=int(args.top_k), device=device)

# Output
print(f"Class{' '*22}Probability")
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        classes = [cat_to_name[cl] for cl in classes]
for p, cl in zip(probs, classes):
    print(f"{cl:27s}{p:.5f}")
