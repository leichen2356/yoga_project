"""GCN architecture and backend loader shared with notebook 9."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class PoseGCN(nn.Module):
    """Two symmetric-normalized graph convolutions and graph-level angles."""

    def __init__(self, config):
        super().__init__()
        self.node_count = len(config['node_ids'])
        self.coordinate_count = 3 * self.node_count
        self.register_buffer('feature_mean', torch.zeros(config['feature_count']))
        self.register_buffer('feature_scale', torch.ones(config['feature_count']))
        adjacency = torch.eye(self.node_count)
        lookup = {node: i for i, node in enumerate(config['node_ids'])}
        for first, second in config['edges']:
            a, b = lookup[first], lookup[second]
            adjacency[a, b] = adjacency[b, a] = 1
        degree = adjacency.sum(1).pow(-0.5)
        self.register_buffer('adjacency', degree[:, None] * adjacency * degree[None, :])
        hidden = config['hidden_channels']
        self.conv1 = nn.Linear(3, hidden)
        self.conv2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(config['dropout'])
        self.head = nn.Linear(hidden + config['angle_count'], config['class_count'])

    def forward(self, raw_features):
        features = (raw_features - self.feature_mean) / self.feature_scale
        nodes = features[:, :self.coordinate_count].reshape(-1, self.node_count, 3)
        nodes = self.dropout(torch.relu(self.conv1(self.adjacency @ nodes)))
        nodes = self.dropout(torch.relu(self.conv2(self.adjacency @ nodes)))
        pooled = nodes.mean(dim=1)
        return self.head(torch.cat([pooled, features[:, self.coordinate_count:]], dim=1))


def load_gcn(output_dir):
    """Load CPU inference model; normalization is included in its state dict."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / 'model_manifest.json').read_text(encoding='utf-8'))
    model = PoseGCN(manifest['architecture'])
    model.load_state_dict(torch.load(output_dir / manifest['best_model_file'],
                                    map_location='cpu', weights_only=True))
    return model.eval(), manifest


def predict_poses(model, manifest, dataframe):
    """Accept the same keypoint/angle features as training; return labels and scores."""
    values = dataframe.loc[:, manifest['feature_columns']].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Features must be finite.')
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(values)).softmax(dim=1).numpy()
    labels = np.asarray(manifest['classes'])[probabilities.argmax(axis=1)]
    return labels, probabilities
