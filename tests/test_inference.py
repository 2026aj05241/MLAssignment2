import pytest
from src.inference import load_model, read_metadata

def test_bad_checksum_and_unknown_model_are_rejected():
    read_metadata()
    with pytest.raises(ValueError, match='checksum'):
        load_model('svm', '0' * 64)
    with pytest.raises(ValueError, match='Unknown'):
        load_model('../arbitrary', '0' * 64)
