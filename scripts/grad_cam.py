import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_gradient)
        self.target_layer.register_full_backward_hook(self.save_gradient_backprop)

    def save_gradient(self, module, input, output):
        self.gradients = output

    def save_gradient_backprop(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, inputs):
        outputs = self.model(inputs)
        self.model.zero_grad()
        target = torch.ones(outputs.size()).to(inputs.device)
        outputs.backward(gradient=target)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.gradients.detach()

        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU activation
        heatmap = heatmap / np.max(heatmap)  # Normalisation
        return heatmap
