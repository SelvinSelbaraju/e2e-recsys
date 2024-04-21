from typing import Dict
import torch


class FeatureImputer:
    def __init__(self, imputations: Dict):
        self.imputations = imputations

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}
        for feature, feature_values in x.values():
            # Check if an imputation exists
            if feature in self.imputations:
                impute_value = self.imputations[feature]
                outputs[feature] = torch.nan_to_num(
                    feature_values,
                    nan=impute_value,
                    posinf=impute_value,
                    neginf=impute_value,
                )
            else:
                outputs[feature] = x[feature]
        return outputs
