import torch

def verify_model(model, batch, device):
    model.freeze()
    output = model.forward(batch)

    assert_never_nan(output)
    assert_never_inf(output)
    return output


def assert_never_nan(in_tensor):
    """Checks against intermediary nan values.
    Args:
        in_tensor: Tensor to be run (can also be training op).
    Raise:
        NanTesnorException: If any value is ever NaN.
    """
    try:
        assert not torch.isnan(in_tensor).byte().any()
    except AssertionError:
        raise NaNTensorException("There was a NaN value in tensor")


def assert_never_inf(in_tensor):
    """Checks against intermediary nan values.
    Args:
        in_tensor: Tensor to be run (can also be training op).
        feed_dict: Feed diction required to obtain in_tensor.
        sess_conf: Session configuration.
        init_op: initialization operation.
    Raise:
        NanTesnorException: If any value is ever NaN.
    """
    try:
        assert torch.isfinite(in_tensor).byte().any()
    except AssertionError:
        raise InfTensorException("There was an Inf value in tensor")
