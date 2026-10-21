import pytest
import torch
from medical_cnn.model import RegularizedCNN


def test_output_shape():
    model = RegularizedCNN(num_classes=4)
    assert model(torch.randn(2, 3, 128, 128)).shape == (2, 4)


def test_invalid_input_channels_raise():
    model = RegularizedCNN(num_classes=2)
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(1, 1, 64, 64))


def test_invalid_class_count_raises():
    with pytest.raises(ValueError, match="num_classes"):
        RegularizedCNN(num_classes=1)


def test_dropout_range_validation():
    with pytest.raises(ValueError, match="dropout"):
        RegularizedCNN(num_classes=2, dropout=1.0)
