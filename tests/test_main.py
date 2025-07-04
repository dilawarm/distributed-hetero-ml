from distributed_hetero_ml.main import add


def test_add() -> None:
    """Test for add function"""
    assert add(1, 2) == 3
