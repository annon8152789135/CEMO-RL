import numpy as np
import math 

class AutoPred:
    def __init__(self, num_feats=0, feature_names=None):
        self.num_feats = num_feats
        self.feature_names = feature_names

    def _fmt(self, x):
        if x is None:
            return None
        if isinstance(x, float):
            return f"{x:.{self.num_feats}f}".rstrip("0").rstrip(".")
        return str(x)
    
    def num_predicates(self):
        return self.num_feats

    def _feature_predicate(self, feat_name, bounds):
        # adapt this depending on exact format of cluster.get_bounds(j)
        if bounds is None:
            return None

        # example: bounds = (low, high)
        low, high = bounds

        low = self._fmt(low)
        high = self._fmt(high)

        if low is None and high is None:
            return None
        elif low is None:
            return f"{feat_name} less than {high}"
        elif high is None:
            return f"{feat_name} greater than or equal to {low}"
        elif low == high:
            return f"{feat_name} equal to {low}"
        else:
            return f"{feat_name} between {low} and {high}"

    def translate_cluster(self, cluster_boundary_dict, feature_names_to_use=None):
        preds = []

        if feature_names_to_use is not None:
            cluster_boundary_dict = {
                feat_name: bounds
                for feat_name, bounds in cluster_boundary_dict.items()
                if feat_name in feature_names_to_use
            }

        for feat_name, bounds in cluster_boundary_dict.items():
            p = self._feature_predicate(feat_name, bounds)
            if p is not None:
                preds.append(p)

        if not preds:
            return "(no predicates)"
        return " and ".join(preds)

    def my_translation_algo(self, cluster_boundaries_for_graph, feature_names_to_use=None):
        explanations = []

        if feature_names_to_use is not None:
            if len(cluster_boundaries_for_graph) != len(feature_names_to_use):
                raise ValueError(
                    f"Length mismatch: {len(cluster_boundaries_for_graph)} cluster boundary dicts, "
                    f"but {len(feature_names_to_use)} feature lists"
                )

            for cluster_boundary_dict, selected_features in zip(cluster_boundaries_for_graph, feature_names_to_use):
                explanations.append(self.translate_cluster(cluster_boundary_dict, selected_features))
        else:
            for cluster_boundary_dict in cluster_boundaries_for_graph:
                explanations.append(self.translate_cluster(cluster_boundary_dict))

        return explanations