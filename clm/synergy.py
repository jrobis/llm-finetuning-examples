import torch


def calc_loss_fct(loss_fct, logits, labels):
    r""" Calculates self.loss_fct with logits and labels that are expected to be aligned already.
    """
    _logits = logits.contiguous()
    _labels = labels.contiguous()
    loss = loss_fct(_logits.view(-1, _logits.size(-1)), _labels.view(-1))
    return loss


def scaling_law_loss_to_params(loss):
    r""" (OpenAI scaling laws) Kaplan, Jared, et al. "Scaling laws for neural language models." arXiv:2001.08361 (2020)
    """
    num_params = torch.exp(torch.log(torch.tensor(8.8e13).to(loss.device)) -
                           torch.log(torch.clamp(loss, 1.69)) / 0.076)  # loss lower bound 1.69 is entropy of natural text
    return num_params


def measured_loss(logits, target):
    combined_logits = torch.log(torch.softmax(logits, dim=-1) + 1e-40)
    
    loss_fct = torch.nn.CrossEntropyLoss()
    measured_loss = calc_loss_fct(loss_fct, combined_logits, target)

    return measured_loss


def synergy(expected_loss, measured_loss, scaling_law_power):
    
    loss_diff_share = torch.clamp(expected_loss - measured_loss, 0) / 2

    measured_params = scaling_law_loss_to_params(measured_loss)
    expected_params = scaling_law_loss_to_params(expected_loss)

    # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
    pow_measured_params = torch.pow(measured_params, scaling_law_power)
    pow_expected_params = torch.pow(expected_params, scaling_law_power)

    synergy_share = torch.clamp(pow_measured_params - pow_expected_params, 0) / 2

    return synergy_share


if __name__ == "__main__":

    _exp_loss = 3.0 # e.g. eval_loss
    _meas_loss = 2.9 # measured_loss(logits, target)


    print(
        synergy(
            torch.tensor(3.0), torch.tensor(2.99), 0.5
        )
    )