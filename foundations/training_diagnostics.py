import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals
        stats = []
        with torch.no_grad():
            for module in model.children():
                #forward pass through through each layer
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean_val = x.mean().item()
                    std_val = x.std().item()
                    dead_frac = (x<=0).all(dim=0).float().mean().item()

                    # Step-by-step evaluation:
                # 1. (x <= 0)
                #    [[True, True, True, False],
                #     [True, True, True, True],
                #     [True, True, False, True]]
                #
                # 2. .all(dim=0) -> [True, True, False, False]
                # 3. .float()    -> [1.0,  1.0,  0.0,   0.0]
                # 4. .mean()     -> 2 / 4 = 0.5
                # 5. .item()     -> 0.5

                    stats.append({'mean':round(mean_val, 4), 'std':round(std_val, 4), 'dead_fraction':round(dead_frac, 4)})

        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        stats = []

        output = model(x) #forward pass
        loss = nn.MSELoss()(output, y) #calculating loss
        loss.backward() #backward pass - calculates all the gradients

        for module in model.children():
            #iterating through each layer
            if isinstance(module, nn.Linear):
                grad = module.weight.grad #list of gradients of each weight in the layer
                mean_val = grad.mean().item()
                std_val = grad.std().item()
                l2_norm_val = torch.norm(grad).item()

                stats.append({'mean':round(mean_val, 4), 'std':round(std_val, 4), 'norm':round(l2_norm_val, 4)})

        return stats 



    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        #if any layer has dead_fracion > 0.5 = dead neurons
        for s in activation_stats:
            if s['dead_fraction'] > 0.5:
                return 'dead_neurons'
            
        #if any layer's gradient norm > 1000 = exploding gradients
        for s in gradient_stats:
            if s['norm'] > 1000:
                return 'exploding_gradients'
        

        #if last layer's gradient norm < 1e-5 = vanishing gradients
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'


        #if activation std of any layer < 0.1 = vanishing gradient, > 10 = exploding gradients
        for s in activation_stats:
            if s['std'] < 0.1:
                return 'vanishing_gradients'
            if s['std'] > 10:
                return 'exploding_gradients'

        return 'healthy'

