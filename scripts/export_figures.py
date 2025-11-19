"""Read outputs/results.json and draw ROC/loss curves.
Simple plotting utilities.
"""
import json
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--results', default='outputs/results.json')
parser.add_argument('--outdir', default='outputs/figures')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

if not os.path.exists(args.results):
    raise FileNotFoundError(f"Results file not found: {args.results}")

with open(args.results, 'r') as f:
    r = json.load(f)

# Example: plot search history
if isinstance(r.get('search_history'), list) and len(r['search_history']) > 0:
    # support both dicts with 'score' and float directly
    scores = []
    for s in r['search_history']:
        if isinstance(s, dict) and 'score' in s:
            scores.append(s['score'])
        else:
            try:
                scores.append(float(s))
            except Exception:
                pass
    if scores:
        plt.figure()
        plt.plot(scores, marker='o')
        plt.title('Search combined validation score (lower better)')
        plt.xlabel('candidate')
        plt.ylabel('score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'search_history.png'))
        plt.close()

# Plot ROC if available (expect structure {'roc':{'fpr':[], 'tpr':[], 'auc':float}})
if 'roc' in r and isinstance(r['roc'], dict) and 'fpr' in r['roc'] and 'tpr' in r['roc']:
    plt.figure()
    plt.plot(r['roc']['fpr'], r['roc']['tpr'], label=f"AUC={r['roc'].get('auc', 0):.3f}")
    plt.plot([0,1],[0,1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'roc.png'))
    plt.close()

print('Figures saved to', args.outdir)

# Plot training curves if available
if 'train_curves' in r:
    tc = r['train_curves']
    if 'train_loss' in tc:
        plt.figure(); plt.plot(tc['train_loss']); plt.title('Training Loss'); plt.xlabel('step'); plt.ylabel('loss')
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'train_loss.png')); plt.close()
    if 'val_acc' in tc:
        plt.figure(); plt.plot(tc['val_acc']); plt.title('Validation Accuracy'); plt.xlabel('eval step'); plt.ylabel('acc')
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'val_acc.png')); plt.close()

    # Vẽ error curves nếu có
    if isinstance(r.get('search_history'), list) and len(r['search_history']) > 0:
        src_e = [h.get('src_err') for h in r['search_history'] if isinstance(h, dict) and 'src_err' in h]
        tgt_e = [h.get('tgt_err') for h in r['search_history'] if isinstance(h, dict) and 'tgt_err' in h]
        hyb_e = [h.get('hybrid_err') for h in r['search_history'] if isinstance(h, dict) and 'hybrid_err' in h]
        import matplotlib.pyplot as plt, os
        if hyb_e:
            plt.figure(); plt.plot(hyb_e, marker='o'); plt.title('Hybrid validation error'); plt.grid(True)
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'hybrid_val_error.png')); plt.close()
        if src_e and tgt_e:
            plt.figure(); 
            plt.plot(src_e, label='src_err'); plt.plot(tgt_e, label='tgt_err')
            plt.title('Source/Target validation error'); plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'src_tgt_val_error.png')); plt.close()


