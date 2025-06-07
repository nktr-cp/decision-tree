from src.main import gini_impurity, splitter

def test_impurity():
    # Test with a list of labels
    labels = [0, 1, 1, 2, 2, 2]
    result = gini_impurity(labels)
    # 1 - (1/6)^2 - (2/6)^2 - (3/6)^2
    expected = 1 - (1/6)**2 - (2/6)**2 - (3/6)**2
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"

    # Test with an empty list
    result = gini_impurity([])
    expected = 0.0
    assert result == expected, f"Expected {expected}, got {result}"

    # Test with a single class
    labels = [1, 1, 1]
    result = gini_impurity(labels)
    # 1 - (1)^2
    expected = 1 - (1)**2
    assert result == expected, f"Expected {expected}, got {result}"
