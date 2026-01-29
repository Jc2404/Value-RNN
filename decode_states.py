import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant
from environments.starkweather import StarkweatherEnv

from agents.drqn import DRQN

from utils import generate_hiddens_and_states, get_run_statistic

from argparse import ArgumentParser

def safelog_torch(x, eps=1e-12):
    return torch.log(torch.clamp(x, min=eps))

class LinearMultinomialProbe(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)  # logits


def fit_state_decoder_torch(
    X_train, y_train,
    standardize=True,
    C=1.0,
    epochs=200,
    lr=1e-2,
    batch_size=1024,
    bias=True,
):
    """
    Faithful to decode_X_from_y_fit:
      - linear multinomial model
      - train on hard labels with CE
      - standardize using train stats
    """
    device = X_train.device
    N, H = X_train.shape

    # ensure y is 1D int64 labels
    if y_train.ndim > 1:
        y_train = y_train.squeeze(-1)
    y_train = y_train.long()

    K = int(torch.max(y_train).item() + 1)

    if standardize:
        mean = X_train.mean(0, keepdim=True)
        std = X_train.std(0, keepdim=True).clamp_min(1e-6)
    else:
        mean, std = None, None

    probe = LinearMultinomialProbe(H, K, bias=bias).to(device)

    # Approximate sklearn's L2 via weight decay (not numerically identical but same intent)
    weight_decay = 1.0 / max(C, 1e-12)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            if standardize:
                xb = (xb - mean) / std

            logits = probe(xb)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return {"probe": probe, "mean": mean, "std": std, "standardize": standardize, "K": K}


@torch.no_grad()
def eval_state_decoder_torch(X, y, state):
    """
    Faithful to decode_X_from_y_eval:
      - pte_hat = softmax(logits)
      - Yh = argmax(pte_hat)
      - LL = mean log p(true_class)
      - pcor = accuracy * 100
      - phat_mean = mean(p_hat | true class)
    """
    probe = state["probe"]

    if y.ndim > 1:
        y = y.squeeze(-1)
    y = y.long()

    if state["standardize"]:
        X = (X - state["mean"]) / state["std"]

    logits = probe(X)
    pte_hat = F.softmax(logits, dim=-1)         # [N, K]
    Yh = torch.argmax(pte_hat, dim=-1)          # [N]

    p_true = pte_hat.gather(1, y.view(-1, 1)).squeeze(1)
    LL = safelog_torch(p_true).mean().item()
    pcor = (Yh == y).float().mean().item() * 100.0

    classes = torch.unique(y)
    phat_mean = torch.stack([
        pte_hat[y == c].mean(dim=0) if (y == c).any() else torch.zeros(state["K"], device=X.device)
        for c in classes
    ], dim=0)

    return {"LL": LL, "pcor": pcor, "phat_mean": phat_mean.detach().cpu(), "classes": classes.detach().cpu()}

def main(args):

    train_args = get_run_statistic(args.train_id)

    # merge for wandb
    config = vars(train_args) | vars(args)

    wandb.init(
        project='belief-softmax',
        name=args.name,
        config=config,
        save_code=True,
    )
    config = wandb.config

    wandb.save('*.py')
    wandb.save('agents/*.py')
    wandb.save('environments/*.py')

    if train_args.environment == 'tmaze':
        environment = TMaze(
            bayes=True,
            length=train_args.length,
            stochasticity=train_args.stochasticity,
        )
    elif train_args.environment == 'hike':
        environment = MountainHike(
            bayes=True,
            variations=train_args.variations,
        )
    elif config.environment == 'starkweather':
        environment = StarkweatherEnv(
            p_omission= train_args.p_omission,
            bin_size = train_args.bin_size,
            iti_hazard = train_args.iti_hazard,
            iti_min = train_args.iti_min,
            nITI_microstates = train_args.nITI_microstates,
        )
    else:
        environment = train_args.environment
        raise NotImplementedError(f'Unknown environment {environment}')

    if train_args.irrelevant != 0:
        environment = Irrelevant(
            environment,
            state_size=train_args.irrelevant,
            bayes=True,
        )

    if train_args.algorithm == 'drqn':
        network_kwargs = {
            'num_layers': train_args.num_layers,
            'hidden_size': train_args.hidden_size,
        }
        agent = DRQN(
            cell=train_args.cell,
            action_size=environment.action_size,
            observation_size=environment.observation_size,
            **network_kwargs,
        )
    else:
        raise NotImplementedError(f'Unknown algorithm {args.algorithm}')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('cpu') 
    else:
        device = torch.device('cpu')

    print('Device:', device)
    print(config.episodes)

    for episode in range(0, config.episodes + 1, args.mine_period):
        
        # load agent checkpoint
        agent.load(args.train_id, episode=episode)
        print('agent loaded')

        # sample hidden states + true states
        hiddens, states = generate_hiddens_and_states(
            agent,
            environment,
            num_samples=args.mine_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )
        hiddens = hiddens.to(device)
        states  = states.to(device)

        # If states are one-hot [N,K], convert to labels [N]
        if states.ndim == 2:
            states = states.argmax(dim=1)

        # shuffle + split
        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        states  = states[perm]

        split = int(N * 0.8)
        X_train, X_test = hiddens[:split], hiddens[split:]
        y_train, y_test = states[:split],  states[split:]

        # fit decoder on train
        dec_state = fit_state_decoder_torch(
            X_train, y_train,
            standardize=True,
            C=args.C,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            bias=True,
        )

        # eval metrics on train/test
        res_test  = eval_state_decoder_torch(X_test,  y_test,  dec_state)
        res_train = eval_state_decoder_torch(X_train, y_train, dec_state)

        wandb.log({
            "train/episode": episode,
            "probe_state/LL": res_test["LL"],
            "probe_state/pcor": res_test["pcor"],
            "probe_state/LL_train": res_train["LL"],
            "probe_state/pcor_train": res_train["pcor"],
        })

        print(f"[episode {episode}] "
            f"LL={res_test['LL']:.4f}, pcor={res_test['pcor']:.2f}% "
            f"(train LL={res_train['LL']:.4f}, train pcor={res_train['pcor']:.2f}%)")

    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Probe RNN hidden state with multinomial state decoder (LL, accuracy)',
    )
    parser.add_argument('name', type=str, nargs='?', default=None)
    parser.add_argument('train_id', type=str)

    # sampling
    parser.add_argument('--mine_num_samples', type=int, default=10000)
    parser.add_argument('--mine_period', type=int, default=100)
    parser.add_argument('--approximate', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.0)

    # decoder training hyperparams
    parser.add_argument('--probe_epochs', type=int, default=300)
    parser.add_argument('--probe_lr', type=float, default=1e-2)
    parser.add_argument('--probe_batch_size', type=int, default=1024)
    parser.add_argument('--C', type=float, default=1.0)  # inverse reg strength analogue

    args = parser.parse_args()
    main(args)