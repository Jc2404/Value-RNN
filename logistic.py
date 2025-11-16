import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant

from agents.drqn import DRQN

from utils import generate_hiddens_and_beliefs, get_run_statistic

from argparse import ArgumentParser



class SoftmaxProbe(nn.Module):
    """Linear probe that outputs log-probabilities."""
    def __init__(self, in_dim, out_dim, add_bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=add_bias)

    def forward(self, x):
        logits = self.linear(x)
        return F.log_softmax(logits, dim=-1)

class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = 128, add_bias=True, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=add_bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim, bias=add_bias)
        )

    def forward(self, X):
        logits = self.net(X)
        return F.log_softmax(logits, dim=-1)

def fit_softmax_probe(
    X,
    Y,
    add_bias=True,
    standardize=True,
    epochs=200,
    lr=1e-2,
    batch_size=1024,
    use_MLP = True
):
    """
    X: [N, H] hidden states
    Y: [N, K] belief probabilities (should sum to 1)
    """
    device = X.device
    N, H = X.shape
    K = Y.shape[1]

    # standardize inputs (like yours)
    if standardize:
        mean = X.mean(0, keepdim=True)
        std = X.std(0, keepdim=True) + 1e-6
        Xn = (X - mean) / std
    else:
        mean, std = None, None
        Xn = X
    
    if use_MLP:
        print("Using MLP for fitting")
        probe = MLPProbe(H, K, add_bias=add_bias).to(device)
    else:
        print("Using Linear fitting")
        probe = SoftmaxProbe(H, K, add_bias=add_bias).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xn[idx]
            yb = Y[idx]

            log_probs = probe(xb)              # [B, K] log-probs
            
            loss = criterion(log_probs, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        if (ep + 1) % 50 == 0 or ep == 0:
            avg_kl = total_loss / max(num_batches, 1)
            print(f"[Epoch {ep+1}/{epochs}] train_KL={avg_kl:.4f}")

    return {
        "probe": probe,
        "mean": mean,
        "std": std,
        "standardize": standardize,
    }


def eval_softmax_probe(X, Y, state):
    """
    Return KL and cross-entropy so we can log both.
    """
    device = X.device
    probe = state["probe"]
    if state["standardize"]:
        Xn = (X - state["mean"]) / state["std"]
    else:
        Xn = X

    with torch.no_grad():
        log_probs = probe(Xn)          # [N, K]
        probs = log_probs.exp()        # convert back to probs

    # KL(p || q): target = Y (probs), input = log_probs
    kl = F.kl_div(log_probs, Y, reduction="batchmean").item()

    # cross-entropy = - E_p [log q]
    ce = -(Y * log_probs).sum(dim=-1).mean().item()

    return kl, ce, probs, log_probs, Y



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

        # sample hidden states + belief tuple
        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent,
            environment,
            num_samples=args.mine_num_samples,
            epsilon=args.epsilon,
            approximate=args.approximate,
        )
        print(f'generated hiddens: {hiddens.shape}, beliefs: {[b.shape for b in beliefs]}')

        # move to device
        hiddens = hiddens.to(device)
        beliefs = tuple(b.to(device) for b in beliefs)

        # shuffle + split
        N = hiddens.size(0)
        perm = torch.randperm(N, device=device)
        hiddens = hiddens[perm]
        beliefs = tuple(b[perm] for b in beliefs)
        split = int(N * 0.8)
        X_train = hiddens[:split]
        X_test = hiddens[split:]

        # loop over belief parts (relevant, irrelevant)
        for part_idx, belief_part in enumerate(beliefs):
            Y_train = belief_part[:split]
            Y_test = belief_part[split:]

            # train probe on train split
            probe_state = fit_softmax_probe(
                X_train,
                Y_train,
                add_bias=True,
                standardize=True,
                epochs=args.probe_epochs,
                lr=args.probe_lr,
                batch_size=args.probe_batch_size,
                use_MLP = args.use_MLP
            )

            kl, ce, probs, log_probs, target = eval_softmax_probe(X_test, Y_test, probe_state)
            # look at sample of it
            print(f"episode {episode}")
            print("x1", X_test[0])
            print("y1", target[0])
            print("yhat1", probs[0])
            print("x2", X_test[1])
            print("y2", target[1])
            print("yhat2", probs[1])
            print("x3", X_test[2])
            print("y3", target[2])
            print("yhat3", probs[2])
            print(f"kl = {kl}, ce = {ce}")

            # log to wandb
            wandb.log({
                'train/episode': episode,
                f'probe/kl-{part_idx}': kl,
                f'probe/ce-{part_idx}': ce,
            })
            print(f"[episode {episode}] belief {part_idx}: KL={kl:.4f}  CE={ce:.4f}")

    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Probe RNN beliefs with softmax + KL',
    )
    parser.add_argument('name', type=str, nargs='?', default=None)
    parser.add_argument('train_id', type=str)

    # how many samples we gather for probing each time
    parser.add_argument('--mine_num_samples', type=int, default=10000)
    parser.add_argument('--use_MLP', type=bool, default=True)
    parser.add_argument('--mine_period', type=int, default=1000)
    parser.add_argument('--approximate', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.0)

    # probe training hyperparams
    parser.add_argument('--probe_epochs', type=int, default=300)
    parser.add_argument('--probe_lr', type=float, default=1e-2)
    parser.add_argument('--probe_batch_size', type=int, default=1024)

    # keep other args for compatibility / unused
    parser.add_argument('--mine_num_layers', type=int, default=2)
    parser.add_argument('--mine_hidden_size', type=int, default=256)
    parser.add_argument('--mine_alpha', type=float, default=0.01)
    parser.add_argument('--mine_num_epochs', type=int, default=100)
    parser.add_argument('--mine_batch_size', type=int, default=1024)
    parser.add_argument('--mine_learning_rate', type=float, default=1e-3)
    parser.add_argument('--mine_lambda', type=float, default=0.0)
    parser.add_argument('--representation_size', type=int, default=16)
    parser.add_argument('--belief_part', type=int, default=None)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--train_set', action='store_true')

    args = parser.parse_args()
    main(args)
