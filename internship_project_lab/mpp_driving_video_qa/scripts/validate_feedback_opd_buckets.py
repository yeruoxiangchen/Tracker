#!/usr/bin/env python3
"""Small real NCCL check for the four-rank reducer, including locally absent gradients."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mpp_video_qa.opd import average_gradients
from mpp_video_qa.opd_runtime import GradientBuckets


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group('nccl')
    try:
        params=[torch.nn.Parameter(torch.ones(1024,device=device)) for _ in range(128)]
        old=[torch.nn.Parameter(p.detach().clone()) for p in params]
        reducer=GradientBuckets(params,world,1)
        errors, seconds = [], {'legacy':[], 'bucketed':[]}
        for iteration in range(22):
            for index,(a,b) in enumerate(zip(old,params)):
                grad=None if (rank+index+iteration)%3==0 else torch.full_like(a,rank+index+1.)
                a.grad=grad
                b.grad=None if grad is None else grad.clone()
            for label, reduction in (('legacy',lambda:average_gradients(old,world)), ('bucketed',reducer.reduce)):
                torch.distributed.barrier()
                torch.cuda.synchronize(device)
                started=time.perf_counter()
                reduction()
                torch.cuda.synchronize(device)
                if iteration>=2:
                    seconds[label].append(time.perf_counter()-started)
            errors.append(max(float((a.grad-b.grad).abs().max()) for a,b in zip(old,params)))
        result={'rank':rank,'world_size':world,'passed':max(errors)==0.,'max_abs_gradient_difference':max(errors),
                'collectives_legacy':len(params),'collectives_bucketed':len(reducer.buckets),
                'mean_ms':{k:1000*sum(v)/len(v) for k,v in seconds.items()},
                'interpretation':'Small tensors measure communication launch overhead, not complete OPD throughput.'}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/('nccl_rank%d.json'%rank)).write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        if not result['passed']:
            raise RuntimeError('bucketed NCCL gradients differ')
    finally:
        torch.distributed.destroy_process_group()


if __name__=='__main__':
    main()
